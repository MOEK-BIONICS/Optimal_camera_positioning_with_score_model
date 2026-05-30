"""
score_model.py
─────────────────────────────────────────────────────────────────────────────
Implements the score-based diffusion model that replaces the GP prior in the
LGCP sensor placement framework.

Architecture overview
─────────────────────
•  Forward process  : gradually adds Gaussian noise to clean intensity
                      fields x₀ following a cosine variance schedule.

•  Score U-Net      : conditional denoising network that predicts the noise
                      ε added at timestep t given the noisy field xₜ and
                      the geometric conditioning tensor g (SDF, gradients,
                      room labels).  The network architecture is:

                        Input (xₜ ⊕ g)
                            ↓  Encoder (3× downsample + residual blocks)
                        Bottleneck (residual + spatial attention)
                            ↓  Decoder (3× upsample + skip connections)
                        Output ε̂  (same shape as x₀)

•  Training         : denoising score matching — minimise
                      E_{t,x₀,ε}[ ||ε̂_θ(xₜ,t,g) − ε||² ]

•  Intensity extraction:
                      After training, use Tweedie's formula to recover
                      the posterior mean of x₀ from a noisy observation:
                        x̂₀ = (xₜ − √(1−ᾱₜ)·ε̂) / √ᾱₜ

•  Sampling (DDIM)  : deterministic reverse diffusion over a subset of
                      timesteps, producing K intensity field realisations
                      {λ₁,...,λ_K} for Monte-Carlo void probability.

Mathematical connection to the papers
──────────────────────────────────────
The trained score model s_θ(xₜ,t) ≈ ∇_{xₜ} log p_t(xₜ) captures the
same information as the GP posterior distribution in Kim et al. (2023),
but without the stationarity and smoothness constraints imposed by the
Matérn covariance kernel.  The key identity is:

    log λ(x) ≈ log p(x) + C

where p(x) is the data density learned by the score model and C is a
scalar set by the observed event rate.  Learning the score ≡ learning
how the intensity varies across space.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Compute the cumulative noise schedule ᾱₜ for t = 0, 1, ..., T.

    The cosine schedule (Nichol & Dhariwal, 2021) corrupts data more
    gently than a linear schedule, preserving fine spatial detail at low
    noise levels — important for intensity fields near wall boundaries.

        ᾱₜ = cos²( π/2 · (t/T + s) / (1 + s) )

    Returns
    -------
    alpha_bar : (T+1,) tensor,  ᾱ₀ = 1.0,  ᾱ_T ≈ 0.0
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(math.pi / 2 * (steps / T + s) / (1 + s)) ** 2
    alpha_bar = f / f[0]          # normalise so ᾱ₀ = 1
    return alpha_bar.float()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Forward diffusion
# ─────────────────────────────────────────────────────────────────────────────

def forward_diffuse(x0: torch.Tensor, t: torch.Tensor,
                    alpha_bar: torch.Tensor):
    """
    Sample xₜ from the forward diffusion process q(xₜ | x₀).

        xₜ = √ᾱₜ · x₀  +  √(1−ᾱₜ) · ε,   ε ~ N(0, I)

    Parameters
    ----------
    x0        : (B, 1, H, W) clean intensity fields
    t         : (B,)         integer timestep indices
    alpha_bar : (T+1,)       precomputed cumulative noise schedule

    Returns
    -------
    xt    : (B, 1, H, W) noisy fields
    noise : (B, 1, H, W) the Gaussian noise added (training target)
    """
    noise = torch.randn_like(x0)
    ab = alpha_bar[t].to(x0.device).view(-1, 1, 1, 1)
    xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
    return xt, noise


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Score U-Net building blocks
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar timestep t → dense vector of dimension `dim`.

    Uses alternating sin/cos at geometric frequencies, identical to the
    transformer positional encoding.  Followed by two linear layers
    (MLP) to produce the conditioning vector injected into every
    residual block via Adaptive Group Normalisation (AdaGN).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t : (B,) long  →  (B, dim*4) float"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)
        return self.mlp(emb)   # (B, dim*4)


# ──────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual convolutional block with Adaptive Group Normalisation (AdaGN).

    Structure
    ---------
        GroupNorm → SiLU → Conv2d(3×3)
            ↓  inject timestep embedding (scale + shift)
        GroupNorm → SiLU → Conv2d(3×3)
            +  skip connection (1×1 conv if channels change)

    The timestep embedding is injected via scale-shift:
        h = GroupNorm(h) * (1 + scale) + shift
    where scale and shift are linear projections of the time embedding.
    This allows every spatial feature to be modulated by the noise level.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 n_groups: int = 8):
        super().__init__()
        # Clamp group count so it always divides the channel count
        g_in  = min(n_groups, in_ch)
        g_out = min(n_groups, out_ch)

        self.norm1    = nn.GroupNorm(g_in,  in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(g_out, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act      = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_ch * 2)   # → scale, shift
        self.skip     = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Inject timestep via AdaGN
        proj   = self.time_proj(t_emb)          # (B, out_ch*2)
        scale, shift = proj.chunk(2, dim=-1)     # each (B, out_ch)
        scale  = scale.unsqueeze(-1).unsqueeze(-1)   # (B, out_ch, 1, 1)
        shift  = shift.unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h) * (1.0 + scale) + shift

        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ──────────────────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """
    Multi-head self-attention over the spatial dimensions of a feature map.

    Reshapes (B, C, H, W) → (B, H·W, C) sequence, applies attention,
    reshapes back.  Used in the U-Net bottleneck to capture long-range
    spatial dependencies (e.g. the food court and main entrance are
    correlated attractors even when far apart on the grid).
    """

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        n_heads = min(n_heads, channels)
        # ensure head_dim is valid
        while channels % n_heads != 0:
            n_heads -= 1
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, n_heads,
                                          batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h   # residual


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full Score U-Net
# ─────────────────────────────────────────────────────────────────────────────

class ScoreUNet(nn.Module):
    """
    Conditional U-Net that predicts the noise ε given (xₜ, t, geometry).

    Input channels (N_INPUT_CHANNELS = 5)
    ──────────────────────────────────────
        Ch 0 : xₜ             — noisy intensity field at timestep t
        Ch 1 : SDF            — signed distance to nearest wall (normalised)
        Ch 2 : ∂SDF/∂x        — wall direction x-component
        Ch 3 : ∂SDF/∂y        — wall direction y-component
        Ch 4 : room label     — semantic zone (corridor, retail, food, …)

    Architecture
    ─────────────
        Encoder: 3 downsampling stages, each = ResBlock + AvgPool2d(2)
        Bottleneck: ResBlock + SpatialAttention + ResBlock
        Decoder: 3 upsampling stages, each = Upsample + ResBlock + skip concat
        Output: 1×1 conv → single-channel noise prediction ε̂
    """

    def __init__(self, in_ch: int = C.N_INPUT_CHANNELS,
                 base_ch: int = C.BASE_CHANNELS,
                 time_dim: int = C.TIME_DIM):
        super().__init__()
        ch = base_ch
        td = time_dim * 4   # output size of SinusoidalTimeEmbedding

        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # ── Encoder ───────────────────────────────────────────────────
        self.enc0 = ResBlock(in_ch,  ch,    td)   # (B, ch,   H,   W)
        self.enc1 = ResBlock(ch,     ch*2,  td)   # (B, ch*2, H/2, W/2)
        self.enc2 = ResBlock(ch*2,   ch*4,  td)   # (B, ch*4, H/4, W/4)

        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(2)

        # ── Bottleneck ────────────────────────────────────────────────
        self.mid1 = ResBlock(ch*4, ch*4, td)
        self.mid_attn = SpatialAttention(ch*4, n_heads=4)
        self.mid2 = ResBlock(ch*4, ch*4, td)

        # ── Decoder (skip connections double the input channels) ───────
        # Note: we use F.interpolate with explicit size rather than a fixed
        # 2× upsample, because non-square grids (e.g. 40×50) produce odd
        # intermediate sizes (10×12 after two halvings) that don't round-trip
        # cleanly through AvgPool2d + Upsample(scale_factor=2).
        self.dec2  = ResBlock(ch*4 + ch*4, ch*2, td)
        self.dec1  = ResBlock(ch*2 + ch*2, ch,   td)
        self.dec0  = ResBlock(ch   + ch,   ch,   td)

        # ── Output head ───────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, 1, 1)   # → (B, 1, H, W)

    # ------------------------------------------------------------------
    def forward(self, xt: torch.Tensor, t: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xt       : (B, 1, H, W)  noisy intensity field
        t        : (B,)           integer timestep indices
        geometry : (B, 4, H, W)  [sdf, sdf_dx, sdf_dy, room]

        Returns
        -------
        eps_hat : (B, 1, H, W)  predicted noise
        """
        # Concatenate input channels
        x = torch.cat([xt, geometry], dim=1)   # (B, 5, H, W)

        # Timestep embedding
        t_emb = self.time_embed(t)             # (B, TIME_DIM*4)

        # Encoder
        e0 = self.enc0(x,             t_emb)  # (B, ch,   H,   W)
        e1 = self.enc1(self.down1(e0), t_emb) # (B, ch*2, H/2, W/2)
        e2 = self.enc2(self.down2(e1), t_emb) # (B, ch*4, H/4, W/4)

        # Bottleneck
        h = self.mid1(e2, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)               # (B, ch*4, H/4, W/4)

        # Decoder with skip connections — upsample to exactly match encoder size
        h = self.dec2(torch.cat([
            F.interpolate(h, size=e2.shape[2:], mode='bilinear', align_corners=False),
            e2], dim=1), t_emb)
        h = self.dec1(torch.cat([
            F.interpolate(h, size=e1.shape[2:], mode='bilinear', align_corners=False),
            e1], dim=1), t_emb)
        h = self.dec0(torch.cat([
            F.interpolate(h, size=e0.shape[2:], mode='bilinear', align_corners=False),
            e0], dim=1), t_emb)

        # Output
        return self.out_conv(self.out_act(self.out_norm(h)))  # (B,1,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_score_model(train_dataset, val_dataset,
                      n_epochs: int = C.N_EPOCHS,
                      batch_size: int = C.BATCH_SIZE,
                      lr: float = C.LR_SCORE,
                      device: str = C.DEVICE,
                      verbose: bool = True):
    """
    Train the Score U-Net using denoising score matching.

    Objective
    ---------
        L(θ) = E_{t, x₀, ε} [ ||ε̂_θ(xₜ, t, g) − ε||² ]

    At each step:
        1. Sample a batch of clean intensity fields x₀.
        2. Sample random timesteps t ~ Uniform{1, ..., T}.
        3. Add noise:  xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε.
        4. Predict the noise:  ε̂ = network(xₜ, t, geometry).
        5. Backpropagate  ||ε̂ − ε||².

    Parameters
    ----------
    train_dataset, val_dataset : MallDataset instances
    n_epochs   : number of training epochs
    batch_size : mini-batch size
    lr         : Adam learning rate
    device     : 'cuda' or 'cpu'
    verbose    : print per-epoch loss

    Returns
    -------
    model      : trained ScoreUNet
    alpha_bar  : (T+1,) cosine noise schedule tensor
    history    : dict with 'train_loss' and 'val_loss' lists
    """
    # ── Data loaders ──────────────────────────────────────────────────
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, drop_last=False)

    # ── Noise schedule ────────────────────────────────────────────────
    alpha_bar = cosine_alpha_bar(C.T_DIFFUSION).to(device)

    # ── Model ─────────────────────────────────────────────────────────
    model = ScoreUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[ScoreUNet] {n_params:,} trainable parameters")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr * 0.01
    )

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, n_epochs + 1):
        # ── Training phase ────────────────────────────────────────────
        model.train()
        train_losses = []
        for x0, geom in train_loader:
            x0   = x0.to(device)    # (B, 1, H, W)
            geom = geom.to(device)  # (B, 4, H, W)

            # Random timesteps
            t = torch.randint(1, C.T_DIFFUSION + 1,
                              (x0.shape[0],), device=device)

            # Forward diffusion
            xt, noise = forward_diffuse(x0, t, alpha_bar)

            # Predict noise
            eps_hat = model(xt, t, geom)

            # Denoising score matching loss
            loss = F.mse_loss(eps_hat, noise)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            optimiser.step()

            train_losses.append(loss.item())

        scheduler.step()

        # ── Validation phase ──────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x0, geom in val_loader:
                x0   = x0.to(device)
                geom = geom.to(device)
                t    = torch.randint(1, C.T_DIFFUSION + 1,
                                     (x0.shape[0],), device=device)
                xt, noise = forward_diffuse(x0, t, alpha_bar)
                eps_hat   = model(xt, t, geom)
                val_losses.append(F.mse_loss(eps_hat, noise).item())

        train_l = sum(train_losses) / len(train_losses)
        val_l   = sum(val_losses)   / len(val_losses)
        history['train_loss'].append(train_l)
        history['val_loss'].append(val_l)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"train={train_l:.4f}  val={val_l:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    return model, alpha_bar, history


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Intensity extraction — Tweedie's formula
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_mean_intensity(model: ScoreUNet,
                             geometry: torch.Tensor,
                             alpha_bar: torch.Tensor,
                             n_samples: int = 128,
                             t_eval: int = 50,
                             device: str = C.DEVICE) -> torch.Tensor:
    """
    Estimate the posterior mean intensity field λ̄(x,y) using Tweedie's
    formula.

    At a small noise level t_eval, the posterior mean of the clean data
    given the noisy data is:

        E[x₀ | xₜ] = (xₜ − √(1−ᾱₜ) · ε̂(xₜ,t)) / √ᾱₜ

    We average this over many noisy samples to obtain λ̄.

    Mathematical note
    -----------------
    This is the neural-network equivalent of the INLA posterior mean
    Eλ[λ(s)] used in Kim et al. (2023), but without the GP's Gaussian
    distributional assumption.

    Parameters
    ----------
    model     : trained ScoreUNet
    geometry  : (4, H, W) geometric conditioning
    alpha_bar : (T+1,) noise schedule
    n_samples : number of Monte Carlo samples for averaging
    t_eval    : noise level at which to apply Tweedie (small = less noisy)
    device    : 'cuda' or 'cpu'

    Returns
    -------
    lambda_mean : (1, H, W) float32 tensor — estimated mean intensity
    """
    model.eval()
    geom = geometry.unsqueeze(0).expand(n_samples, -1, -1, -1).to(device)
    t_batch = torch.full((n_samples,), t_eval, dtype=torch.long, device=device)

    ab = alpha_bar[t_eval].to(device)

    # Sample noisy fields from the prior at noise level t_eval
    x_prior = torch.randn(n_samples, 1, C.GRID_H, C.GRID_W, device=device)
    # These are noisy observations; apply Tweedie to recover x̂₀
    eps_hat = model(x_prior, t_batch, geom)

    # Tweedie denoising
    x0_hat = (x_prior - torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(ab)

    # Mean over samples, clamp to non-negative (intensity ≥ 0)
    lambda_mean = x0_hat.mean(dim=0).clamp(min=0.0)   # (1, H, W)
    return lambda_mean.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DDIM sampling — intensity field realisations
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model: ScoreUNet,
                geometry: torch.Tensor,
                alpha_bar: torch.Tensor,
                K: int = C.N_INTENSITY_SAMPLES,
                ddim_steps: int = C.DDIM_STEPS,
                device: str = C.DEVICE) -> torch.Tensor:
    """
    Generate K intensity field realisations using DDIM reverse diffusion.

    DDIM (Song et al., 2021) is a deterministic sampler that sub-samples
    T diffusion steps into `ddim_steps` steps.  Because it is deterministic
    given the initial noise seed, it is differentiable — gradients can flow
    back through the sampler to the sensor location parameters.

    The key DDIM update rule at each step (t → t_prev):

        x̂₀ = (xₜ − √(1−ᾱₜ)·ε̂) / √ᾱₜ          (Tweedie denoised estimate)
        xₜ₋₁ = √ᾱₜ₋₁ · x̂₀ + √(1−ᾱₜ₋₁) · ε̂    (DDIM deterministic step)

    Parameters
    ----------
    model      : trained ScoreUNet
    geometry   : (4, H, W) geometric conditioning
    alpha_bar  : (T+1,) noise schedule
    K          : number of samples to draw
    ddim_steps : number of reverse diffusion steps
    device     : computation device

    Returns
    -------
    lambda_samples : (K, 1, H, W) float32 — K intensity field realisations
                     All values are ≥ 0 (intensity must be non-negative).
    """
    model.eval()

    # Sub-sampled timestep sequence  T → 0
    timesteps = torch.linspace(C.T_DIFFUSION - 1, 0,
                               ddim_steps, dtype=torch.long)

    geom = geometry.unsqueeze(0).expand(K, -1, -1, -1).to(device)

    # Start from pure Gaussian noise
    x = torch.randn(K, 1, C.GRID_H, C.GRID_W, device=device)

    for i, t_val in enumerate(timesteps):
        t_batch = torch.full((K,), t_val.item(), dtype=torch.long, device=device)
        ab_t = alpha_bar[t_val].to(device)

        # Predict noise
        eps_hat = model(x, t_batch, geom)

        # Tweedie: x̂₀ estimate
        x0_hat = (x - torch.sqrt(1.0 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        x0_hat = x0_hat.clamp(-5.0, 5.0)   # numerical stability

        # DDIM step
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            ab_prev = alpha_bar[t_prev].to(device)
        else:
            ab_prev = torch.tensor(1.0, device=device)

        x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps_hat

    # Final samples — clamp to non-negative
    lambda_samples = x.clamp(min=0.0).cpu()
    return lambda_samples   # (K, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Score model smoke test ──────────────────────────────────────")
    device = C.DEVICE
    model = ScoreUNet().to(device)
    alpha_bar = cosine_alpha_bar(C.T_DIFFUSION)

    B = 2
    x0   = torch.randn(B, 1, C.GRID_H, C.GRID_W).to(device)
    geom = torch.randn(B, 4, C.GRID_H, C.GRID_W).to(device)
    t    = torch.randint(1, C.T_DIFFUSION, (B,)).to(device)
    xt, noise = forward_diffuse(x0, t, alpha_bar)
    eps_hat   = model(xt, t, geom)

    print(f"  Input shape  : {xt.shape}")
    print(f"  Output shape : {eps_hat.shape}")
    print(f"  Loss         : {F.mse_loss(eps_hat, noise).item():.4f}")
    print("── Smoke test passed ───────────────────────────────────────────")
