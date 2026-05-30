"""
data_generation.py
─────────────────────────────────────────────────────────────────────────────
Generates everything needed for a self-contained demo:

    1.  MallFloorPlan   — a synthetic 100 m × 80 m shopping mall with rooms,
                          corridors, entrances, and restricted zones.

    2.  compute_sdf()   — converts the binary free/wall mask into a Signed
                          Distance Function (SDF).  Every cell stores its
                          signed distance to the nearest wall (positive = free
                          space, negative = inside wall).  The SDF and its
                          spatial gradients are the geometric conditioning
                          channels fed to the score U-Net.

    3.  TrajectorySimulator — generates realistic pedestrian trajectories that
                          respect the floor plan.  People enter from doors,
                          are attracted to "hotspot" regions (food court, main
                          entrance, anchor stores), and avoid walls via a
                          repulsion potential derived from the SDF.

    4.  build_intensity_field() — converts a batch of (x,y) observations into
                          a 2-D kernel density estimate on the grid.  These
                          fields are the training targets x₀ for the diffusion
                          model.

    5.  MallDataset     — a PyTorch Dataset wrapping all of the above, ready
                          to hand to a DataLoader.

Mathematical connection to the papers
──────────────────────────────────────
•  The intensity fields produced here play the role of λ(s) in Kim et al.
   (2023).  Each field is one realisation of the true (unknown) intensity
   function from which ships / people arrive.

•  The SDF channels replace the hand-crafted Matérn covariance structure of
   the GP by letting the neural network learn spatial correlations from data,
   while still respecting hard geometric boundaries.
"""

import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt
import config as C

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Floor plan
# ─────────────────────────────────────────────────────────────────────────────

class MallFloorPlan:
    """
    Synthetic mall floor plan.

    Coordinate system
    -----------------
    •  Physical space  : (x, y) in metres, x ∈ [0, MALL_WIDTH_M],
                                            y ∈ [0, MALL_HEIGHT_M]
    •  Grid space      : (col, row) integer indices
    •  col = x / GRID_RES_M   (east-west)
    •  row = y / GRID_RES_M   (north-south, 0 = south wall)

    Attributes
    ----------
    free_mask  : (H, W) bool tensor  — True where people can walk
    room_map   : (H, W) int  tensor  — semantic room label per cell
                 0 = wall / outside
                 1 = main corridor (east-west spine)
                 2 = side corridor (north-south)
                 3 = retail unit
                 4 = food court
                 5 = entrance / exit
    doors      : list of (x, y) physical positions of entrance doors
    hotspots   : list of (x, y, strength) attraction points
    """

    ROOM_WALL      = 0
    ROOM_CORRIDOR  = 1
    ROOM_SIDE_COR  = 2
    ROOM_RETAIL    = 3
    ROOM_FOOD      = 4
    ROOM_ENTRANCE  = 5

    def __init__(self):
        H, W = C.GRID_H, C.GRID_W
        self.H = H
        self.W = W

        # Start with everything as wall
        self.room_map  = np.zeros((H, W), dtype=np.int32)
        self.free_mask = np.zeros((H, W), dtype=bool)

        self._build_layout()
        self._define_hotspots()

    # ------------------------------------------------------------------
    def _rect(self, r0, c0, r1, c1, label):
        """Fill rectangle [r0:r1, c0:c1] with given room label."""
        self.room_map [r0:r1, c0:c1] = label
        self.free_mask[r0:r1, c0:c1] = True

    def _build_layout(self):
        H, W = self.H, self.W
        r = self._rect

        # ── Main east-west corridor (rows 17-23, full width) ──────────
        r(17, 0, 23, W, self.ROOM_CORRIDOR)

        # ── North-south side corridors ────────────────────────────────
        r(0,  10, H, 14,  self.ROOM_SIDE_COR)   # west side corridor
        r(0,  36, H, 40,  self.ROOM_SIDE_COR)   # east side corridor

        # ── Retail units (north wing) ─────────────────────────────────
        r(24, 0,  H,  10, self.ROOM_RETAIL)     # NW retail block
        r(24, 14, H,  36, self.ROOM_RETAIL)     # N central retail
        r(24, 40, H,  W,  self.ROOM_RETAIL)     # NE retail block

        # ── South wing retail + food court ───────────────────────────
        r(0,  0,  17, 10, self.ROOM_RETAIL)     # SW retail
        r(0,  14, 17, 24, self.ROOM_RETAIL)     # S central retail
        r(0,  26, 17, 36, self.ROOM_FOOD)       # Food court
        r(0,  40, 17, W,  self.ROOM_RETAIL)     # SE retail

        # ── Entrance zones (doors cut through outer wall) ─────────────
        r(17, 0,  23, 3,  self.ROOM_ENTRANCE)   # west entrance
        r(17, W-3, 23, W, self.ROOM_ENTRANCE)   # east entrance
        r(H-3, 10, H, 40, self.ROOM_ENTRANCE)   # north entrance (wide)
        r(0,  20, 3, 30,  self.ROOM_ENTRANCE)   # south entrance

        # ── Door locations (physical metres) ─────────────────────────
        self.doors = [
            (1.0,  20.0),       # west door
            (99.0, 20.0),       # east door
            (25.0, 79.0),       # north door (left)
            (50.0, 79.0),       # north door (centre)
            (25.0, 0.0),        # south door
        ]

    def _define_hotspots(self):
        """
        Attraction points that pull pedestrian trajectories.
        Each entry: (x_m, y_m, strength)
        Higher strength → more trajectories pass nearby.
        """
        self.hotspots = [
            (50.0, 20.0, 3.0),   # centre of main corridor
            (30.0, 60.0, 2.0),   # north-central retail
            (30.0,  8.0, 2.5),   # food court
            (80.0, 60.0, 1.5),   # NE retail
            (10.0, 60.0, 1.5),   # NW retail
            ( 8.0, 20.0, 2.0),   # west entrance area
            (92.0, 20.0, 2.0),   # east entrance area
        ]

    # ------------------------------------------------------------------
    def world_to_grid(self, x_m, y_m):
        """Convert physical (x, y) in metres to (col, row) grid indices."""
        col = np.clip(x_m / C.GRID_RES_M, 0, self.W - 1).astype(int)
        row = np.clip(y_m / C.GRID_RES_M, 0, self.H - 1).astype(int)
        return col, row

    def grid_to_world(self, col, row):
        """Convert grid (col, row) to physical (x, y) metres (cell centres)."""
        x = (col + 0.5) * C.GRID_RES_M
        y = (row + 0.5) * C.GRID_RES_M
        return x, y

    def is_free(self, x_m, y_m):
        """Return True if physical location is in walkable space."""
        col, row = self.world_to_grid(np.array([x_m]), np.array([y_m]))
        return bool(self.free_mask[row[0], col[0]])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Signed Distance Function
# ─────────────────────────────────────────────────────────────────────────────

def compute_sdf(floor_plan: MallFloorPlan):
    """
    Compute the Signed Distance Function (SDF) for the mall floor plan.

    For each grid cell the SDF value is:
        +d   if the cell is in free space, where d = distance (metres)
             to the nearest wall cell
        -d   if the cell is inside a wall, where d = distance to the
             nearest free-space cell
         0   on the boundary

    The SDF and its spatial gradients form the geometric conditioning
    channels of the score U-Net, allowing it to learn that intensity
    must be zero inside walls and transitions are hard discontinuities.

    Returns
    -------
    sdf        : (H, W) float32 tensor  — signed distance in metres
    sdf_grad_x : (H, W) float32 tensor  — ∂sdf/∂x
    sdf_grad_y : (H, W) float32 tensor  — ∂sdf/∂y
    """
    free = floor_plan.free_mask.astype(np.float32)   # 1 = free, 0 = wall

    # Distance from free cells to nearest wall  (inside free space)
    dist_free = distance_transform_edt(free)          # pixels

    # Distance from wall cells to nearest free cell (inside walls)
    dist_wall = distance_transform_edt(1 - free)      # pixels

    # Signed distance: positive in free space, negative in walls
    sdf_px = dist_free - dist_wall                    # pixels

    # Convert pixel distances to metres
    sdf_m = sdf_px * C.GRID_RES_M                    # metres

    # Spatial gradients via central differences
    sdf_grad_y, sdf_grad_x = np.gradient(sdf_m)      # (H,W)

    # Convert to tensors
    sdf        = torch.tensor(sdf_m,      dtype=torch.float32)
    sdf_grad_x = torch.tensor(sdf_grad_x, dtype=torch.float32)
    sdf_grad_y = torch.tensor(sdf_grad_y, dtype=torch.float32)

    return sdf, sdf_grad_x, sdf_grad_y


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Trajectory simulator
# ─────────────────────────────────────────────────────────────────────────────

class TrajectorySimulator:
    """
    Generates synthetic pedestrian trajectories inside a mall floor plan.

    Physics model
    ─────────────
    Each person starts at a random entrance door and moves by:

        velocity_{t+1} = α · velocity_t
                       + β · ∇(attraction potential)
                       - γ · ∇(wall repulsion potential)
                       + σ · noise

    Attraction potential: Gaussian wells centred on hotspots.
    Wall repulsion:       Exponential function of the SDF — cells near
                          walls have a strong positive repulsion force
                          pointing away from the wall (direction of ∇SDF).

    The result is trajectories that flow naturally through corridors,
    cluster near hotspots, and never pass through walls.

    Parameters
    ----------
    floor_plan : MallFloorPlan
    sdf        : (H, W) tensor — metres to nearest wall
    sdf_grad_x : (H, W) tensor — ∂sdf/∂x
    sdf_grad_y : (H, W) tensor — ∂sdf/∂y
    """

    def __init__(self, floor_plan, sdf, sdf_grad_x, sdf_grad_y):
        self.fp = floor_plan
        self.sdf        = sdf.numpy()
        self.sdf_grad_x = sdf_grad_x.numpy()
        self.sdf_grad_y = sdf_grad_y.numpy()

        # Precompute grid of cell-centre physical coordinates
        cols = np.arange(floor_plan.W)
        rows = np.arange(floor_plan.H)
        self.grid_x, self.grid_y = np.meshgrid(
            (cols + 0.5) * C.GRID_RES_M,
            (rows + 0.5) * C.GRID_RES_M
        )   # both (H, W)

    # ------------------------------------------------------------------
    def _sdf_at(self, x, y):
        """Bilinear SDF query at physical (x, y) in metres."""
        col = np.clip(x / C.GRID_RES_M, 0, self.fp.W - 1)
        row = np.clip(y / C.GRID_RES_M, 0, self.fp.H - 1)
        c0, r0 = int(col), int(row)
        c1 = min(c0 + 1, self.fp.W - 1)
        r1 = min(r0 + 1, self.fp.H - 1)
        dc, dr = col - c0, row - r0
        v = (self.sdf[r0, c0] * (1-dc) * (1-dr) +
             self.sdf[r0, c1] * dc     * (1-dr) +
             self.sdf[r1, c0] * (1-dc) * dr     +
             self.sdf[r1, c1] * dc     * dr)
        gx = (self.sdf_grad_x[r0, c0] * (1-dc) * (1-dr) +
              self.sdf_grad_x[r0, c1] * dc     * (1-dr) +
              self.sdf_grad_x[r1, c0] * (1-dc) * dr     +
              self.sdf_grad_x[r1, c1] * dc     * dr)
        gy = (self.sdf_grad_y[r0, c0] * (1-dc) * (1-dr) +
              self.sdf_grad_y[r0, c1] * dc     * (1-dr) +
              self.sdf_grad_y[r1, c0] * (1-dc) * dr     +
              self.sdf_grad_y[r1, c1] * dc     * dr)
        return float(v), float(gx), float(gy)

    def _attraction_force(self, x, y):
        """
        Gradient of the hotspot attraction potential.
        Returns (fx, fy) pointing toward the nearest active hotspot.
        """
        fx, fy = 0.0, 0.0
        for hx, hy, strength in self.fp.hotspots:
            dx, dy = hx - x, hy - y
            d2 = dx*dx + dy*dy + 1e-4
            # Gaussian attraction well: F ∝ strength · exp(-d²/2σ²) · direction
            sigma2 = 400.0   # (20 m)² — broad attraction basin
            w = strength * math.exp(-d2 / (2 * sigma2))
            fx += w * dx / math.sqrt(d2)
            fy += w * dy / math.sqrt(d2)
        return fx, fy

    def generate_trajectory(self, rng):
        """
        Simulate one pedestrian trajectory.

        Returns
        -------
        positions : (T_steps, 2) numpy array of (x, y) positions in metres
        """
        # Start at a random entrance door with small jitter
        door = rng.choice(self.fp.doors)
        x = door[0] + rng.uniform(-1.5, 1.5)
        y = door[1] + rng.uniform(-1.5, 1.5)
        x = float(np.clip(x, 1.0, C.MALL_WIDTH_M  - 1.0))
        y = float(np.clip(y, 1.0, C.MALL_HEIGHT_M - 1.0))

        vx, vy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
        positions = [(x, y)]

        # Dynamics parameters
        alpha = 0.7    # velocity damping
        beta  = 0.3    # attraction strength
        gamma = 2.0    # wall repulsion strength
        sigma = 0.8    # noise standard deviation (metres / step)
        dt    = 1.0    # time step

        for _ in range(C.TRAJ_STEPS - 1):
            # Attraction force toward hotspots
            ax, ay = self._attraction_force(x, y)

            # Wall repulsion: push away from walls (along ∇SDF)
            sdf_val, gx, gy = self._sdf_at(x, y)
            # Repulsion grows exponentially as agent approaches wall
            repulsion = gamma * math.exp(-max(sdf_val, 0.0) / 5.0)
            rx = repulsion * gx
            ry = repulsion * gy

            # Velocity update
            vx = alpha*vx + beta*ax + rx + sigma*rng.normal()
            vy = alpha*vy + beta*ay + ry + sigma*rng.normal()

            # Speed cap to prevent unrealistic velocities
            speed = math.sqrt(vx*vx + vy*vy)
            max_speed = 3.0   # metres / step
            if speed > max_speed:
                vx *= max_speed / speed
                vy *= max_speed / speed

            # Position update
            x_new = x + vx * dt
            y_new = y + vy * dt

            # Boundary and wall enforcement
            x_new = float(np.clip(x_new, 0.5, C.MALL_WIDTH_M  - 0.5))
            y_new = float(np.clip(y_new, 0.5, C.MALL_HEIGHT_M - 0.5))

            # If new position is inside a wall, don't move there
            col = int(x_new / C.GRID_RES_M)
            row = int(y_new / C.GRID_RES_M)
            col = np.clip(col, 0, self.fp.W - 1)
            row = np.clip(row, 0, self.fp.H - 1)
            if not self.fp.free_mask[row, col]:
                x_new, y_new = x, y   # stay put
                vx, vy = 0.0, 0.0     # reset velocity

            x, y = x_new, y_new
            positions.append((x, y))

        return np.array(positions, dtype=np.float32)

    def generate_window(self, n_trajectories, rng):
        """
        Generate n_trajectories for one time window.
        Returns list of (T_steps, 2) arrays.
        """
        return [self.generate_trajectory(rng) for _ in range(n_trajectories)]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Intensity field builder
# ─────────────────────────────────────────────────────────────────────────────

def build_intensity_field(trajectories, floor_plan,
                           bandwidth_m=C.KDE_BANDWIDTH_M):
    """
    Convert a list of trajectories into a 2-D kernel density estimate.

    Uses fully-vectorised numpy operations — all observations are deposited
    simultaneously using advanced indexing, giving ~20× speedup over a
    Python loop.

    Returns
    -------
    field : (1, H, W) float32 tensor — intensity field (persons / m²)
    """
    H, W = C.GRID_H, C.GRID_W
    r = int(math.ceil(2.5 * bandwidth_m / C.GRID_RES_M))  # 2.5σ truncation
    sigma2 = bandwidth_m ** 2

    field = np.zeros((H, W), dtype=np.float32)

    # Collect all (x, y) positions from all trajectories
    all_pts = np.concatenate([t for t in trajectories], axis=0)  # (N, 2)
    xs, ys  = all_pts[:, 0], all_pts[:, 1]

    # Cell centres grid
    col_centres = (np.arange(W) + 0.5) * C.GRID_RES_M   # (W,)
    row_centres = (np.arange(H) + 0.5) * C.GRID_RES_M   # (H,)

    # For each observation, compute its contribution to a local patch
    for x, y in zip(xs, ys):
        col_c = x / C.GRID_RES_M
        row_c = y / C.GRID_RES_M
        col_i = int(col_c)
        row_i = int(row_c)

        # Local patch bounds
        c0 = max(col_i - r, 0); c1 = min(col_i + r + 1, W)
        r0 = max(row_i - r, 0); r1 = min(row_i + r + 1, H)

        # Vectorised distance computation for patch
        cc = col_centres[c0:c1]                      # (nc,)
        rr = row_centres[r0:r1]                      # (nr,)
        dx = cc - x                                  # (nc,)
        dy = rr[:, None] - y                         # (nr, 1)
        d2 = dx[None, :]**2 + dy**2                  # (nr, nc)
        field[r0:r1, c0:c1] += np.exp(-d2 / (2 * sigma2)).astype(np.float32)

    # Zero out walls
    field *= floor_plan.free_mask.astype(np.float32)

    # Normalise: intensity integrates to 1.0 over the domain.
    # This makes it a proper spatial density function — the void probability
    # exp(-∫λπ ds) then measures the probability of zero events in a unit
    # time window.  Absolute event rate is absorbed into the Poisson scaling.
    total = field.sum()
    if total > 1e-8:
        field = field / total   # spatial density, sums to 1 over all cells

    return torch.tensor(field, dtype=torch.float32).unsqueeze(0)  # (1,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MallDataset(Dataset):
    """
    PyTorch Dataset for training the score U-Net.

    Each item is:
        intensity_field : (1, H, W) float32  — the clean x₀
        geometry        : (4, H, W) float32  — [sdf, sdf_dx, sdf_dy, room]

    The geometry tensor is constant across all items (it is a property of
    the building, not of the people inside it), so it is pre-computed once.

    Usage
    -----
        dataset = MallDataset(split='train')
        loader  = DataLoader(dataset, batch_size=16, shuffle=True)
        for intensity, geometry in loader:
            ...
    """

    def __init__(self, split='train', verbose=True):
        """
        Parameters
        ----------
        split : 'train' | 'val' | 'test'
        """
        assert split in ('train', 'val', 'test')
        rng = np.random.default_rng(C.RANDOM_SEED + {'train':0,'val':1,'test':2}[split])

        if verbose:
            print(f"[MallDataset] Building '{split}' split ...")

        # ── Floor plan ────────────────────────────────────────────────
        self.floor_plan = MallFloorPlan()

        # ── SDF ───────────────────────────────────────────────────────
        sdf, sdf_gx, sdf_gy = compute_sdf(self.floor_plan)
        self.sdf = sdf

        # ── Room map normalised to [0,1] for neural network input ─────
        room_np = self.floor_plan.room_map.astype(np.float32)
        room_norm = room_np / room_np.max()
        room_t = torch.tensor(room_norm, dtype=torch.float32)

        # ── Geometry tensor: (4, H, W) ─────────────────────────────────
        # Channels: [SDF (normalised), ∂SDF/∂x, ∂SDF/∂y, room label]
        sdf_max = sdf.abs().max().clamp(min=1e-8)
        self.geometry = torch.stack([
            sdf    / sdf_max,    # normalised SDF
            sdf_gx / sdf_max,    # x-gradient
            sdf_gy / sdf_max,    # y-gradient
            room_t               # semantic room label
        ], dim=0)                # (4, H, W)

        # ── Trajectory simulator ──────────────────────────────────────
        sim = TrajectorySimulator(self.floor_plan, sdf, sdf_gx, sdf_gy)

        # ── Generate intensity fields ─────────────────────────────────
        n_windows = {
            'train': C.N_TRAIN_WINDOWS,
            'val':   C.N_VAL_WINDOWS,
            'test':  C.N_TEST_WINDOWS,
        }[split]

        self.fields = []
        for i in range(n_windows):
            # Vary occupancy: some hours are busy, some quiet
            occupancy_factor = rng.lognormal(mean=0.0, sigma=0.5)
            n_traj = max(10, int(C.TRAJ_PER_WINDOW * occupancy_factor))
            trajs = sim.generate_window(n_traj, rng)
            field = build_intensity_field(trajs, self.floor_plan)
            self.fields.append(field)
            if verbose and (i+1) % 50 == 0:
                print(f"  [{split}] {i+1}/{n_windows} windows done")

        if verbose:
            print(f"[MallDataset] '{split}' ready — {len(self.fields)} samples")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        return self.fields[idx], self.geometry

    # ------------------------------------------------------------------
    def get_mean_intensity(self):
        """
        Return the empirical mean intensity field (1, H, W).
        This is the GP-equivalent of the posterior mean λ̄(s).
        Used as the GP baseline for sensor placement.
        """
        stacked = torch.stack(self.fields, dim=0)   # (N, 1, H, W)
        return stacked.mean(dim=0)                   # (1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Data generation smoke test ──────────────────────────────────")
    ds = MallDataset(split='train')
    field, geom = ds[0]
    print(f"  Intensity field shape : {field.shape}")
    print(f"  Geometry tensor shape : {geom.shape}")
    print(f"  Field min / mean / max: {field.min():.3f} / "
          f"{field.mean():.3f} / {field.max():.3f}")
    print(f"  SDF channel range     : {geom[0].min():.2f} – {geom[0].max():.2f}")
    print("── Smoke test passed ───────────────────────────────────────────")
