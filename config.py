"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Score-Based LGCP Sensor Placement system.

All physical dimensions are in METRES.  All angles are in DEGREES.
Change values here; every other module imports from this file.

Research context
────────────────
This system extends the Log-Gaussian Cox Process (LGCP) sensor placement
framework from Kim et al. (2023, 2025) by replacing the Gaussian Process
prior with a score-based diffusion model.  The key motivation is that
real indoor environments (malls, shipping terminals) have non-Gaussian,
geometrically constrained intensity fields that GP+Matérn kernels cannot
capture faithfully.

Pipeline overview
─────────────────
1.  Toy data generation  →  synthetic mall floor plan + trajectories
2.  SDF computation      →  signed distance function from floor plan
3.  Intensity fields     →  hourly KDE maps from trajectory observations
4.  Score model          →  conditional U-Net trained on intensity fields
5.  Intensity sampling   →  DDIM reverse diffusion → λ realisations
6.  Sensor placement     →  greedy init + gradient ascent on void prob
7.  Validation           →  NLL, calibration, simulation miss-rate
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Mall floor plan ───────────────────────────────────────────────────────────
MALL_WIDTH_M   = 100.0   # metres east-west
MALL_HEIGHT_M  = 80.0    # metres north-south
GRID_RES_M     = 2.0     # metres per grid cell
GRID_W         = int(MALL_WIDTH_M  / GRID_RES_M)   # 50 cells
GRID_H         = int(MALL_HEIGHT_M / GRID_RES_M)   # 40 cells
CELL_AREA_M2   = GRID_RES_M ** 2                   # 4 m²

# ── Toy dataset ───────────────────────────────────────────────────────────────
N_TRAIN_WINDOWS   = 80     # hourly intensity-field snapshots for training
N_VAL_WINDOWS     = 20     # validation snapshots
N_TEST_WINDOWS    = 20     # test snapshots
TRAJ_PER_WINDOW   = 50     # mean number of person-trajectories per hour
TRAJ_STEPS        = 20     # time-steps per trajectory
RANDOM_SEED       = 42

# ── Diffusion hyperparameters ────────────────────────────────────────────────
T_DIFFUSION       = 300    # total forward diffusion steps (kept small for demo)
DDIM_STEPS        = 30     # reverse steps for DDIM sampling (subset of T)
NOISE_SCHEDULE    = "cosine"

# ── Score U-Net architecture ─────────────────────────────────────────────────
BASE_CHANNELS     = 32     # feature channels at first encoder layer
TIME_DIM          = 64     # sinusoidal time-embedding dimension
N_INPUT_CHANNELS  = 5      # [λ_t, sdf, sdf_dx, sdf_dy, room_label]

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE        = 8
N_EPOCHS          = 30     # kept short for demo; use 300+ for a paper
LR_SCORE          = 3e-4
GRAD_CLIP         = 1.0

# ── Intensity extraction ──────────────────────────────────────────────────────
KDE_BANDWIDTH_M   = 4.0    # Gaussian KDE bandwidth for building x₀ fields
N_INTENSITY_SAMPLES = 32   # K samples for Monte-Carlo void probability

# ── Camera (sensor) model ────────────────────────────────────────────────────
N_SENSORS         = 5      # number of cameras to place
MAX_DETECTION_RANGE_M = 20.0   # maximum camera range
CAMERA_FOV_DEG    = 90.0       # field-of-view half-angle → total 90°
DETECTION_RHO     = 0.95       # peak detection probability ρ
SIGMA_L           = 8.0        # Gaussian falloff length-scale (metres)

# ── Sensor optimisation ───────────────────────────────────────────────────────
GREEDY_ANGLE_CANDIDATES = list(range(0, 360, 45))   # 8 candidate angles (fast demo)
N_OPT_ITERATIONS  = 100
LR_SENSORS        = 0.5
LR_ANGLES         = 2.0

# ── Validation / simulation ───────────────────────────────────────────────────
N_SIM_TRIALS      = 500   # Monte-Carlo trials for miss-rate study
CALIBRATION_LEVELS = [50, 80, 95]   # % credible-interval levels to check
