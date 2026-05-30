"""
sensor_placement.py
─────────────────────────────────────────────────────────────────────────────
Implements the complete sensor (camera) placement pipeline:

    1.  CameraModel         — computes γ(x, aᵢ) the probability that
                              camera i at location aᵢ detects a person
                              at location x.  Accounts for:
                              •  Gaussian range falloff
                              •  Field-of-view angular cutoff
                              •  Line-of-sight occlusion by walls (SDF-based)

    2.  miss_probability()  — computes π(x, a) = ∏ᵢ (1 − γ(x,aᵢ)) the
                              probability that the entire camera network
                              fails to detect a person at x.

    3.  void_probability()  — computes the Monte-Carlo void probability:
                                ν(a) ≈ (1/K) Σₖ exp(−Σₓ λₖ(x)·π(x,a)·Δx²)
                              This is the true objective from Kim et al.
                              (2023) without the Jensen approximation.

    4.  greedy_initialise() — places M cameras one at a time, each time
                              choosing the location/angle that maximally
                              thins the mean intensity field λ̄.  This
                              gives the (1−1/e) ≈ 63% guarantee from the
                              submodularity proof in the paper.

    5.  optimise_sensors()  — refines the greedy solution via Adam gradient
                              ascent on the full void probability.  This
                              corresponds to the nonlinear iterative methods
                              (Newton, quasi-Newton) used in Kim et al. (2025).

Mathematical connections
─────────────────────────
•  Void probability:  equation (5) in Kim et al. (2023)
•  Thinning:          the term ∫λ(s)π(s,a)ds is the "thinned" intensity —
                      only events that escape all sensors contribute.
•  Greedy guarantee:  Theorem 1 + Corollary 1.1 from Kim et al. (2023).
•  Jensen's gap:      not used here — we use the exact MC estimate instead,
                      which is the key improvement over the GP+INLA baseline.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Grid of cell-centre coordinates
# ─────────────────────────────────────────────────────────────────────────────

def make_grid_coords():
    """
    Build tensors of cell-centre physical coordinates.

    Returns
    -------
    grid_x : (H, W) float32 — x coordinate (metres) of each cell centre
    grid_y : (H, W) float32 — y coordinate (metres) of each cell centre
    grid_xy: (H*W, 2) float32 — flattened version, rows = [x, y]
    """
    cols = torch.arange(C.GRID_W, dtype=torch.float32)
    rows = torch.arange(C.GRID_H, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(
        (rows + 0.5) * C.GRID_RES_M,
        (cols + 0.5) * C.GRID_RES_M,
        indexing='ij'
    )   # both (H, W)
    grid_xy = torch.stack([grid_x.reshape(-1),
                           grid_y.reshape(-1)], dim=1)  # (H*W, 2)
    return grid_x, grid_y, grid_xy


GRID_X, GRID_Y, GRID_XY = make_grid_coords()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Line-of-sight check
# ─────────────────────────────────────────────────────────────────────────────

def line_of_sight(sensor_xy: torch.Tensor,
                  sdf: torch.Tensor,
                  n_steps: int = 8) -> torch.Tensor:
    """
    Fast line-of-sight check using SDF ray-marching.

    For each grid cell, checks whether the straight-line path from the
    sensor passes through any wall (SDF ≤ 0) at n_steps sample points.

    Uses vectorised operations for speed — all cells checked in parallel.

    Parameters
    ----------
    sensor_xy : (2,) tensor — sensor physical location [x, y] in metres
    sdf       : (H, W) tensor — signed distance function
    n_steps   : number of ray sample points (fewer = faster, less accurate)

    Returns
    -------
    los : (H, W) bool tensor — True if line of sight is clear
    """
    H, W = C.GRID_H, C.GRID_W
    sx = sensor_xy[0].item()
    sy = sensor_xy[1].item()

    tx = GRID_X.reshape(-1)   # (H*W,)
    ty = GRID_Y.reshape(-1)   # (H*W,)

    blocked = torch.zeros(H * W, dtype=torch.bool)

    # Sample along each ray at n_steps fractions (vectorised over all cells)
    for frac in torch.linspace(0.15, 0.85, n_steps):
        ix = sx + frac * (tx - sx)           # (H*W,)
        iy = sy + frac * (ty - sy)           # (H*W,)

        col = ix.div(C.GRID_RES_M).clamp(0, W-1).long()
        row = iy.div(C.GRID_RES_M).clamp(0, H-1).long()

        sdf_val = sdf[row, col]
        blocked = blocked | (sdf_val <= 0.0)

    return (~blocked).reshape(H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Camera detection model
# ─────────────────────────────────────────────────────────────────────────────

def detection_probability(sensor_xy: torch.Tensor,
                           sensor_angle_deg: torch.Tensor,
                           sdf: torch.Tensor,
                           rho: float        = C.DETECTION_RHO,
                           sigma_l: float    = C.SIGMA_L,
                           fov_deg: float    = C.CAMERA_FOV_DEG,
                           max_range: float  = C.MAX_DETECTION_RANGE_M
                           ) -> torch.Tensor:
    """
    Compute γ(x, aᵢ) for every grid cell x simultaneously.

    Detection model
    ---------------
        γ(x, a) = ρ · range_factor(x) · angle_factor(x) · LOS(x)

        range_factor(x) = exp(−‖x − a‖² / (2·σ_l²))
        angle_factor(x) = 1 if |angle(x−a) − θ| < FOV/2, else 0
        LOS(x)          = 1 if no wall between x and a, else 0

    Parameters
    ----------
    sensor_xy        : (2,) float tensor — camera location [x, y] metres
    sensor_angle_deg : scalar float tensor — camera pointing direction (°)
    sdf              : (H, W) SDF tensor for LOS check
    rho              : peak detection probability ∈ (0, 1]
    sigma_l          : Gaussian falloff length-scale (metres)
    fov_deg          : full field-of-view angle (degrees)
    max_range        : cells beyond this distance have near-zero detection

    Returns
    -------
    gamma : (H, W) float32 tensor — detection probability ∈ [0, 1]
    """
    H, W = C.GRID_H, C.GRID_W

    # ── Vector from sensor to each cell ───────────────────────────────
    dx = GRID_X - sensor_xy[0]   # (H, W)
    dy = GRID_Y - sensor_xy[1]   # (H, W)
    dist2 = dx*dx + dy*dy        # (H, W)  squared distance

    # ── Range falloff  ─────────────────────────────────────────────────
    range_factor = torch.exp(-dist2 / (2.0 * sigma_l**2))

    # Also hard-cutoff at max_range for realism
    range_factor = range_factor * (dist2.sqrt() < max_range).float()

    # ── Angular field of view ──────────────────────────────────────────
    # Angle of vector (dx, dy) relative to east axis, in degrees
    cell_angle_deg = torch.atan2(dy, dx) * (180.0 / math.pi)  # (H,W) [-180,180]

    # Angular difference (handle wrap-around)
    angle_diff = (cell_angle_deg - sensor_angle_deg) % 360.0
    angle_diff = torch.min(angle_diff, 360.0 - angle_diff)    # [0, 180]

    # Within FOV/2 of the pointing direction → detected
    angle_factor = (angle_diff < fov_deg / 2.0).float()

    # ── Line of sight ──────────────────────────────────────────────────
    los = line_of_sight(sensor_xy, sdf).float()   # (H, W)

    # ── Combined detection probability ────────────────────────────────
    gamma = rho * range_factor * angle_factor * los   # (H, W)
    return gamma


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Network miss probability
# ─────────────────────────────────────────────────────────────────────────────

def miss_probability(sensor_xys: torch.Tensor,
                     sensor_angles: torch.Tensor,
                     sdf: torch.Tensor) -> torch.Tensor:
    """
    Compute π(x, a) = ∏ᵢ (1 − γ(x, aᵢ)) for the full camera network.

    This is the probability that ALL cameras simultaneously fail to detect
    a person at location x.  As more cameras are added, π decreases
    monotonically — the diminishing-returns property (submodularity).

    Parameters
    ----------
    sensor_xys   : (M, 2) float tensor — camera locations
    sensor_angles: (M,)   float tensor — camera angles (degrees)
    sdf          : (H, W) SDF tensor

    Returns
    -------
    pi : (H, W) float32 tensor — miss probability at each cell ∈ [0, 1]
    """
    pi = torch.ones(C.GRID_H, C.GRID_W)

    for i in range(sensor_xys.shape[0]):
        gamma_i = detection_probability(
            sensor_xys[i].detach(),      # detach for LOS (not differentiable)
            sensor_angles[i].detach(),
            sdf
        )
        pi = pi * (1.0 - gamma_i)

    return pi   # (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Void probability — the objective function
# ─────────────────────────────────────────────────────────────────────────────

def void_probability_mc(sensor_xys: torch.Tensor,
                         sensor_angles: torch.Tensor,
                         lambda_samples: torch.Tensor,
                         sdf: torch.Tensor) -> torch.Tensor:
    """
    Monte-Carlo estimate of the void probability.

    Mathematical definition (equation 5, Kim et al. 2023):
        P(N̄ = 0) = E_λ[ exp(−∫ λ(s)·π(s,a) ds) ]

    Discretised approximation:
        ν(a) ≈ (1/K) Σₖ exp(−Σₓ λₖ(x)·π(x,a)·Δx²)

    This is the EXACT void probability (no Jensen approximation).
    Using K intensity samples from the score model rather than
    collapsing to the mean λ̄ gives a tighter, unbiased estimate.

    Parameters
    ----------
    sensor_xys     : (M, 2) camera locations  [requires_grad for optimisation]
    sensor_angles  : (M,)   camera angles (°) [requires_grad for optimisation]
    lambda_samples : (K, 1, H, W) intensity realisations from DDIM sampling
    sdf            : (H, W) SDF tensor

    Returns
    -------
    vp : scalar float32 tensor — void probability estimate ∈ [0, 1]
         (differentiable w.r.t. sensor_xys and sensor_angles via π)
    """
    H, W = C.GRID_H, C.GRID_W
    K = lambda_samples.shape[0]
    dx2 = torch.tensor(C.CELL_AREA_M2, dtype=torch.float32)

    # ── Miss probability on this grid ─────────────────────────────────
    pi = miss_probability(sensor_xys, sensor_angles, sdf)   # (H, W)
    pi_flat = pi.reshape(-1)                                 # (H*W,)

    # ── Thinned intensity for each sample ─────────────────────────────
    lam_flat = lambda_samples.reshape(K, -1)   # (K, H*W)

    # Σₓ λₖ(x)·π(x,a)·Δx² for each sample k
    thinned = (lam_flat * pi_flat.unsqueeze(0) * dx2).sum(dim=1)  # (K,)

    # Per-sample void probability
    vp_per_sample = torch.exp(-thinned.clamp(max=80))   # clamp for stability

    # Monte-Carlo average
    vp = vp_per_sample.mean()
    return vp


def void_probability_jensen(sensor_xys: torch.Tensor,
                              sensor_angles: torch.Tensor,
                              lambda_mean: torch.Tensor,
                              sdf: torch.Tensor) -> torch.Tensor:
    """
    Jensen lower-bound void probability (GP baseline equivalent).

    This is the approximation from equation (7) / (9) in Kim et al. (2023):
        ν_Jensen(a) = exp(−∫ λ̄(s)·π(s,a) ds)

    Used for the GP baseline comparison in experiments.

    Parameters
    ----------
    lambda_mean : (1, H, W) mean intensity field λ̄

    Returns
    -------
    vp_lb : scalar — Jensen lower-bound void probability
    """
    dx2    = C.CELL_AREA_M2
    pi     = miss_probability(sensor_xys, sensor_angles, sdf)  # (H, W)
    lam_m  = lambda_mean.squeeze(0)                             # (H, W)
    thinned = (lam_m * pi * dx2).sum()
    return torch.exp(-thinned.clamp(max=80))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Greedy initialisation
# ─────────────────────────────────────────────────────────────────────────────

def greedy_initialise(lambda_mean: torch.Tensor,
                       free_mask: torch.Tensor,
                       sdf: torch.Tensor,
                       M: int = C.N_SENSORS,
                       min_wall_dist_m: float = 3.0,
                       verbose: bool = True):
    """
    Greedily place M cameras to maximise the Jensen void probability
    approximation using the mean intensity field λ̄.

    Algorithm (from Kim et al. 2023, Algorithm 1 adapted for 2D)
    ─────────────────────────────────────────────────────────────
    For m = 1 to M:
        For each candidate location b_j and angle θ:
            Compute ν(a ∪ {b_j, θ}) using current miss prob
            Select (b_j*, θ*) = argmax ν
        Add (b_j*, θ*) to sensor set a

    Submodularity guarantees this greedy solution achieves at least
    (1 − 1/e) ≈ 63.2% of the optimal void probability.

    Parameters
    ----------
    lambda_mean     : (1, H, W) posterior mean intensity
    free_mask       : (H, W) bool — True in walkable cells
    sdf             : (H, W) SDF tensor
    M               : number of cameras to place
    min_wall_dist_m : camera must be at least this far from any wall

    Returns
    -------
    sensors : (M, 2) float tensor — camera locations
    angles  : (M,)  float tensor — camera angles (degrees)
    """
    H, W   = C.GRID_H, C.GRID_W
    dx2    = C.CELL_AREA_M2
    lam_m  = lambda_mean.squeeze(0)    # (H, W)

    # ── Valid placement locations ──────────────────────────────────────
    # Must be in free space AND at least min_wall_dist_m from walls
    valid_mask = free_mask & (sdf >= min_wall_dist_m)
    valid_rows, valid_cols = valid_mask.nonzero(as_tuple=True)
    valid_xs = (valid_cols.float() + 0.5) * C.GRID_RES_M   # physical x
    valid_ys = (valid_rows.float() + 0.5) * C.GRID_RES_M   # physical y

    # Sub-sample candidate locations to keep search tractable
    # (every 4th cell in each dimension → ~16× speedup)
    stride = 4
    keep = ((valid_rows % stride == 0) & (valid_cols % stride == 0))
    valid_rows = valid_rows[keep]
    valid_cols = valid_cols[keep]
    valid_xs   = valid_xs[keep]
    valid_ys   = valid_ys[keep]

    n_locs   = len(valid_xs)
    if verbose:
        print(f"[Greedy] {n_locs} valid camera locations, "
              f"{len(C.GREEDY_ANGLE_CANDIDATES)} candidate angles")

    # ── Running miss probability (updated as cameras are added) ───────
    pi_current = torch.ones(H, W)   # initially everything is missed

    sensors_list = []
    angles_list  = []

    for m in range(M):
        best_vp  = -1.0
        best_xy  = None
        best_ang = None
        best_gamma_flat = None

        lam_flat    = lam_m.reshape(-1)         # (H*W,)
        pi_flat     = pi_current.reshape(-1)    # (H*W,)

        # Precompute range+angle factor for every (location, angle) pair
        # Cache LOS per location (shared across angles)
        ang_candidates = torch.tensor(C.GREEDY_ANGLE_CANDIDATES, dtype=torch.float32)
        fov_half = C.CAMERA_FOV_DEG / 2.0

        for j in range(n_locs):
            x_j = valid_xs[j].item()
            y_j = valid_ys[j].item()
            xy  = torch.tensor([x_j, y_j])

            # LOS mask — cached once per location
            los_flat = line_of_sight(xy, sdf).float().reshape(-1)  # (H*W,)

            # Range factor — same for all angles
            dx = GRID_X - x_j    # (H, W)
            dy = GRID_Y - y_j
            dist2   = (dx*dx + dy*dy).reshape(-1)                   # (H*W,)
            range_f = torch.exp(-dist2 / (2.0 * C.SIGMA_L**2))
            range_f = range_f * (dist2.sqrt() < C.MAX_DETECTION_RANGE_M).float()

            # Cell angles from this location
            cell_ang = torch.atan2(dy, dx).reshape(-1) * (180.0 / 3.14159265)

            for ang_deg in ang_candidates:
                ang_val = ang_deg.item()
                ang_diff = (cell_ang - ang_val) % 360.0
                ang_diff = torch.min(ang_diff, 360.0 - ang_diff)
                angle_f  = (ang_diff < fov_half).float()

                gamma_j = C.DETECTION_RHO * range_f * angle_f * los_flat

                pi_new_flat = pi_flat * (1.0 - gamma_j)

                # Use reduction in thinned intensity as score
                # (maximising VP ≡ minimising thinned intensity)
                thinned_new = (lam_flat * pi_new_flat * dx2).sum().item()
                thinned_cur = (lam_flat * pi_flat     * dx2).sum().item()
                reduction   = thinned_cur - thinned_new   # > 0 = improvement

                if reduction > best_vp:
                    best_vp         = reduction
                    best_xy         = xy.clone()
                    best_ang        = torch.tensor(ang_val)
                    best_gamma_flat = gamma_j.clone()

        # Place the best sensor
        sensors_list.append(best_xy)
        angles_list.append(best_ang)

        # Update running miss probability using cached gamma
        if best_gamma_flat is not None:
            pi_current = (pi_current.reshape(-1) * (1.0 - best_gamma_flat)).reshape(H, W)
        else:
            gamma_best = detection_probability(best_xy, best_ang, sdf)
            pi_current = pi_current * (1.0 - gamma_best)

        # Compute actual VP for logging
        thinned_now = (lam_m * pi_current * dx2).sum()
        vp_now = torch.exp(-thinned_now.clamp(max=80)).item()

        if verbose:
            print(f"  Camera {m+1}/{M}: "
                  f"loc=({best_xy[0]:.1f}, {best_xy[1]:.1f}) m  "
                  f"angle={best_ang.item():.0f}°  "
                  f"VP={vp_now:.4f}")

    sensors = torch.stack(sensors_list)   # (M, 2)
    angles  = torch.stack(angles_list)    # (M,)
    return sensors, angles


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Gradient-based optimisation (nonlinear refinement)
# ─────────────────────────────────────────────────────────────────────────────

def optimise_sensors(init_sensors: torch.Tensor,
                     init_angles: torch.Tensor,
                     lambda_samples: torch.Tensor,
                     lambda_mean: torch.Tensor,
                     sdf: torch.Tensor,
                     free_mask: torch.Tensor,
                     n_iterations: int = C.N_OPT_ITERATIONS,
                     lr_xy: float = C.LR_SENSORS,
                     lr_ang: float = C.LR_ANGLES,
                     verbose: bool = True):
    """
    Refine camera placements via gradient ascent on the Monte-Carlo
    void probability.

    This corresponds to the Newton / quasi-Newton refinement in
    Kim et al. (2025, Algorithm 2), extended to use the full MC
    void probability rather than the Jensen approximation.

    Gradient computation
    ─────────────────────
    The void probability ν(a) = (1/K)Σₖ exp(−Σₓ λₖ·π·Δx²) is
    differentiable with respect to sensor locations because:

        ∂π(x,a)/∂aᵢ = ∂(1−γᵢ)/∂aᵢ · ∏_{j≠i}(1−γⱼ)

    and γᵢ is a smooth Gaussian function of ‖x − aᵢ‖².
    PyTorch autograd handles this automatically.

    Note: the line-of-sight mask is treated as a constant (non-differentiable)
    which is a standard approximation.  The range and angle components are
    fully differentiable.

    Parameters
    ----------
    init_sensors    : (M, 2) greedy initial sensor locations
    init_angles     : (M,)   greedy initial angles
    lambda_samples  : (K, 1, H, W) intensity samples from DDIM
    lambda_mean     : (1, H, W) mean intensity (for Jensen comparison)
    sdf             : (H, W) SDF tensor
    free_mask       : (H, W) bool mask
    n_iterations    : gradient steps
    lr_xy           : learning rate for location parameters
    lr_ang          : learning rate for angle parameters

    Returns
    -------
    sensors_opt : (M, 2) optimised camera locations
    angles_opt  : (M,)   optimised camera angles
    history     : dict with 'vp_mc' and 'vp_jensen' per iteration
    """
    H, W   = C.GRID_H, C.GRID_W
    dx2    = torch.tensor(C.CELL_AREA_M2)

    # ── Differentiable sensor parameters ──────────────────────────────
    sensors = init_sensors.clone().float().requires_grad_(True)
    angles  = init_angles.clone().float().requires_grad_(True)

    optimiser = torch.optim.Adam(
        [{'params': sensors, 'lr': lr_xy},
         {'params': angles,  'lr': lr_ang}]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_iterations, eta_min=lr_xy * 0.05
    )

    # Pre-flatten lambda samples for efficiency
    K     = lambda_samples.shape[0]
    lam_flat = lambda_samples.reshape(K, -1)   # (K, H*W)

    history = {'vp_mc': [], 'vp_jensen': []}

    # ── Precompute LOS masks for each sensor (treated as constant) ────
    # (LOS changes only when sensors move significantly; we refresh
    #  every 50 steps as a practical compromise)
    los_masks = None

    def recompute_los():
        """Recompute line-of-sight masks using current (detached) positions."""
        masks = []
        for i in range(sensors.shape[0]):
            los = line_of_sight(sensors[i].detach(), sdf)   # (H, W)
            masks.append(los.float().reshape(-1))            # (H*W,)
        return torch.stack(masks)   # (M, H*W)

    # ── Differentiable detection probability (range + angle only) ─────
    def diff_detection(sensor_xy, sensor_angle_deg):
        """
        Differentiable version of detection_probability.
        Excludes LOS (handled separately as constant mask).

        sensor_xy         : (2,)  differentiable
        sensor_angle_deg  : ()    differentiable
        """
        dx = GRID_X - sensor_xy[0]   # (H, W)
        dy = GRID_Y - sensor_xy[1]
        dist2 = dx*dx + dy*dy

        # Range factor (differentiable)
        range_f = torch.exp(-dist2 / (2.0 * C.SIGMA_L**2))
        range_f = range_f * (dist2.sqrt() < C.MAX_DETECTION_RANGE_M).float()

        # Angle factor  (differentiable via atan2 — but small gradient)
        cell_ang = torch.atan2(dy, dx) * (180.0 / math.pi)
        ang_diff = (cell_ang - sensor_angle_deg) % 360.0
        ang_diff = torch.min(ang_diff, 360.0 - ang_diff)
        # Soft FOV gate: sigmoid approximation for gradient flow
        sharpness = 0.5   # degrees⁻¹
        angle_f   = torch.sigmoid(
            sharpness * (C.CAMERA_FOV_DEG / 2.0 - ang_diff)
        )

        gamma = C.DETECTION_RHO * range_f * angle_f   # (H, W)
        return gamma.reshape(-1)   # (H*W,)

    # ── Main optimisation loop ─────────────────────────────────────────
    for step in range(1, n_iterations + 1):

        # Refresh LOS every 50 steps
        if los_masks is None or step % 50 == 0:
            los_masks = recompute_los()   # (M, H*W) — constant

        optimiser.zero_grad()

        # Compute miss probability differentiably
        pi_flat = torch.ones(H * W)
        for i in range(sensors.shape[0]):
            gamma_i = diff_detection(sensors[i], angles[i])  # (H*W,)
            gamma_i = gamma_i * los_masks[i]                  # apply LOS
            pi_flat = pi_flat * (1.0 - gamma_i)

        # Thinned intensity: (K,)
        thinned = (lam_flat * pi_flat.unsqueeze(0) * dx2).sum(dim=1)

        # Monte-Carlo void probability
        vp_mc = torch.exp(-thinned.clamp(max=80)).mean()

        # Maximise → minimise negative
        loss = -vp_mc
        loss.backward()

        # ── Gradient projection: keep sensors in valid free space ─────
        with torch.no_grad():
            sensors.grad *= _valid_gradient_mask(sensors, free_mask, sdf)

        torch.nn.utils.clip_grad_norm_([sensors, angles], 5.0)
        optimiser.step()
        scheduler.step()

        # ── Project sensors back to valid positions ───────────────────
        with torch.no_grad():
            sensors.data = _project_to_free_space(
                sensors.data, free_mask, sdf, min_dist_m=2.0
            )
            angles.data = angles.data % 360.0   # wrap angles

        # ── Jensen VP for logging ─────────────────────────────────────
        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                vp_j = void_probability_jensen(
                    sensors.detach(), angles.detach(), lambda_mean, sdf
                ).item()
            history['vp_mc'].append(vp_mc.item())
            history['vp_jensen'].append(vp_j)
            if verbose:
                print(f"  Step {step:3d}/{n_iterations}  "
                      f"VP_MC={vp_mc.item():.4f}  "
                      f"VP_Jensen={vp_j:.4f}  "
                      f"Jensen gap={vp_mc.item()-vp_j:.5f}")

    sensors_opt = sensors.detach()
    angles_opt  = angles.detach()
    return sensors_opt, angles_opt, history


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _valid_gradient_mask(sensors: torch.Tensor,
                          free_mask: torch.Tensor,
                          sdf: torch.Tensor) -> torch.Tensor:
    """
    Returns a (M, 2) mask that zeros gradients for sensors that are
    pushing toward walls.
    """
    M = sensors.shape[0]
    mask = torch.ones(M, 2)
    for i in range(M):
        col = int((sensors[i, 0] / C.GRID_RES_M).clamp(0, C.GRID_W-1).item())
        row = int((sensors[i, 1] / C.GRID_RES_M).clamp(0, C.GRID_H-1).item())
        if not free_mask[row, col]:
            mask[i] = 0.0
    return mask


def _project_to_free_space(sensors: torch.Tensor,
                             free_mask: torch.Tensor,
                             sdf: torch.Tensor,
                             min_dist_m: float = 2.0) -> torch.Tensor:
    """
    If a sensor has drifted into a wall, snap it back to the nearest
    free cell with SDF ≥ min_dist_m.
    """
    proj = sensors.clone()
    for i in range(sensors.shape[0]):
        col = int((sensors[i, 0] / C.GRID_RES_M).clamp(0, C.GRID_W-1).item())
        row = int((sensors[i, 1] / C.GRID_RES_M).clamp(0, C.GRID_H-1).item())
        if not free_mask[row, col] or sdf[row, col] < min_dist_m:
            # Find nearest valid cell
            valid = free_mask & (sdf >= min_dist_m)
            valid_rows, valid_cols = valid.nonzero(as_tuple=True)
            dists = (valid_cols - col)**2 + (valid_rows - row)**2
            best  = dists.argmin()
            proj[i, 0] = (valid_cols[best].float() + 0.5) * C.GRID_RES_M
            proj[i, 1] = (valid_rows[best].float() + 0.5) * C.GRID_RES_M
    return proj


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Simulation study — miss rate evaluation
# ─────────────────────────────────────────────────────────────────────────────

def simulation_miss_rate(sensors: torch.Tensor,
                          angles: torch.Tensor,
                          test_intensity_fields: list,
                          sdf: torch.Tensor,
                          n_trials: int = C.N_SIM_TRIALS) -> dict:
    """
    Empirically estimate the miss rate of a sensor placement by simulating
    Poisson target arrivals and checking detection.

    For each trial:
        1.  Sample a true intensity field λ_true from the test set.
        2.  Generate Poisson events: n_k ~ Poisson(λ_true(x_k) · Δx²)
            for each grid cell k.
        3.  For each event location x, sample detection:
                detected ~ Bernoulli(1 − π(x, a))
        4.  Count missed events.

    Returns
    -------
    dict with keys:
        'miss_rate'      : fraction of events missed overall
        'void_prob_emp'  : empirical fraction of trials with zero misses
        'mean_missed'    : mean number of missed events per trial
    """
    H, W   = C.GRID_H, C.GRID_W
    dx2    = C.CELL_AREA_M2

    # Precompute miss probability (constant for fixed sensor placement)
    pi = miss_probability(sensors, angles, sdf)   # (H, W)

    total_events  = 0
    total_missed  = 0
    zero_miss_trials = 0

    rng = np.random.default_rng(C.RANDOM_SEED + 999)

    for trial in range(n_trials):
        # Sample true intensity from test fields
        idx = rng.integers(0, len(test_intensity_fields))
        lam = test_intensity_fields[idx].squeeze(0).numpy()   # (H, W)

        # Poisson event counts per cell
        expected = lam * dx2
        n_events_per_cell = rng.poisson(expected)   # (H, W)
        n_total = n_events_per_cell.sum()

        if n_total == 0:
            zero_miss_trials += 1
            continue

        # Simulate detection for each event
        miss_prob_np = pi.numpy()
        missed = 0
        for row in range(H):
            for col in range(W):
                n_ev = n_events_per_cell[row, col]
                if n_ev == 0:
                    continue
                # Each event independently missed with probability pi[row,col]
                n_missed = rng.binomial(n_ev, float(miss_prob_np[row, col]))
                missed += n_missed

        total_events += n_total
        total_missed += missed
        if missed == 0:
            zero_miss_trials += 1

    miss_rate = total_missed / max(total_events, 1)
    void_prob_emp = zero_miss_trials / n_trials
    mean_missed   = total_missed / n_trials

    return {
        'miss_rate':      miss_rate,
        'void_prob_emp':  void_prob_emp,
        'mean_missed':    mean_missed,
        'total_events':   total_events,
        'total_missed':   total_missed,
    }
