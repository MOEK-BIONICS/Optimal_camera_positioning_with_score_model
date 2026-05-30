"""
visualisation.py
─────────────────────────────────────────────────────────────────────────────
All plotting routines for the paper figures.

Figure list
───────────
  Fig 1 — Mall floor plan (free mask + room labels + SDF)
  Fig 2 — Sample trajectories overlaid on floor plan
  Fig 3 — Training loss curves (train vs val)
  Fig 4 — Intensity field comparison (true / GP baseline / score model)
  Fig 5 — Sensor placement visualisation (both methods side-by-side)
  Fig 6 — Void probability optimisation convergence
  Fig 7 — Jensen gap over sensor count
  Fig 8 — Calibration plot (expected vs actual coverage)
  Fig 9 — Simulation miss-rate comparison bar chart
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch
import config as C

# ── Custom colour maps ────────────────────────────────────────────────────────
# Intensity: white → amber → dark red (matches Kim et al. heat maps)
INTENSITY_CMAP = LinearSegmentedColormap.from_list(
    'intensity', ['#ffffff', '#FAC775', '#D85A30', '#26215C'], N=256
)

ROOM_COLORS = {
    0: '#2C2C2A',   # wall  — dark grey
    1: '#E6F1FB',   # main corridor — light blue
    2: '#EAF3DE',   # side corridor — light green
    3: '#FAEEDA',   # retail — light amber
    4: '#FAECE7',   # food court — light coral
    5: '#EEEDFE',   # entrance — light purple
}

ROOM_LABELS = {0:'Wall', 1:'Main Corridor', 2:'Side Corridor',
               3:'Retail', 4:'Food Court', 5:'Entrance'}


def _room_rgb(room_map: np.ndarray) -> np.ndarray:
    """Convert integer room map to (H, W, 3) RGB array."""
    H, W = room_map.shape
    rgb = np.zeros((H, W, 3))
    for label, hexcol in ROOM_COLORS.items():
        r = int(hexcol[1:3], 16) / 255.0
        g = int(hexcol[3:5], 16) / 255.0
        b = int(hexcol[5:7], 16) / 255.0
        mask = room_map == label
        rgb[mask] = [r, g, b]
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Floor plan
# ─────────────────────────────────────────────────────────────────────────────

def plot_floor_plan(floor_plan, sdf: torch.Tensor, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ext = [0, C.MALL_WIDTH_M, 0, C.MALL_HEIGHT_M]

    # Panel A — room labels
    ax = axes[0]
    rgb = _room_rgb(floor_plan.room_map)
    ax.imshow(rgb, origin='lower', extent=ext, interpolation='nearest')
    patches = [mpatches.Patch(color=ROOM_COLORS[k], label=ROOM_LABELS[k])
               for k in sorted(ROOM_COLORS.keys())]
    ax.legend(handles=patches, loc='upper right', fontsize=6, framealpha=0.9)
    for (dx, dy) in floor_plan.doors:
        ax.plot(dx, dy, 'r^', ms=6, zorder=5)
    ax.set_title('Room layout', fontsize=10)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

    # Panel B — free mask
    ax = axes[1]
    ax.imshow(floor_plan.free_mask.astype(float), origin='lower', extent=ext,
              cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title('Free-space mask', fontsize=10)
    ax.set_xlabel('x (m)')

    # Panel C — SDF
    ax = axes[2]
    sdf_np = sdf.numpy()
    im = ax.imshow(sdf_np, origin='lower', extent=ext,
                   cmap='RdYlGn', interpolation='bilinear',
                   vmin=sdf_np.min(), vmax=sdf_np.max())
    plt.colorbar(im, ax=ax, label='SDF (m)')
    ax.contour(sdf_np, levels=[0], colors='k', linewidths=1.5,
               origin='lower', extent=ext)
    ax.set_title('Signed Distance Function', fontsize=10)
    ax.set_xlabel('x (m)')

    fig.suptitle('Mall Floor Plan', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Sample trajectories
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(floor_plan, trajectories: list, save_path: str,
                      max_show: int = 40):
    fig, ax = plt.subplots(figsize=(8, 6))
    ext = [0, C.MALL_WIDTH_M, 0, C.MALL_HEIGHT_M]

    # Background room map
    rgb = _room_rgb(floor_plan.room_map)
    ax.imshow(rgb, origin='lower', extent=ext, interpolation='nearest')

    # Trajectories
    cmap = plt.cm.cool
    for i, traj in enumerate(trajectories[:max_show]):
        colour = cmap(i / max(max_show - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], '-', color=colour,
                alpha=0.7, linewidth=1.0)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=colour, ms=3)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=colour, ms=3)

    ax.set_xlim(0, C.MALL_WIDTH_M)
    ax.set_ylim(0, C.MALL_HEIGHT_M)
    ax.set_title(f'Sample trajectories (n={min(len(trajectories), max_show)})',
                 fontsize=11)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Training loss
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_loss(history: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train loss', color='#3266AD')
    ax.plot(epochs, history['val_loss'],   label='Val loss',   color='#D85A30',
            linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss')
    ax.set_title('Score model training', fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Intensity field comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_intensity_comparison(true_field: torch.Tensor,
                               gp_field: torch.Tensor,
                               score_field: torch.Tensor,
                               floor_plan,
                               save_path: str):
    fields = [
        (true_field,  'True intensity (empirical mean)'),
        (gp_field,    'GP baseline (empirical mean λ̄)'),
        (score_field, 'Score model (Tweedie estimate)'),
    ]
    ext = [0, C.MALL_WIDTH_M, 0, C.MALL_HEIGHT_M]
    vmax = max(f.max().item() for f, _ in fields)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (field, title) in zip(axes, fields):
        f_np = field.squeeze(0).numpy()
        # Overlay wall as grey
        wall = ~floor_plan.free_mask
        f_show = f_np.copy()
        im = ax.imshow(f_show, origin='lower', extent=ext,
                       cmap=INTENSITY_CMAP, vmin=0, vmax=vmax,
                       interpolation='bilinear')
        # Wall overlay
        wall_rgba = np.zeros((C.GRID_H, C.GRID_W, 4))
        wall_rgba[wall] = [0.17, 0.17, 0.16, 0.8]
        ax.imshow(wall_rgba, origin='lower', extent=ext,
                  interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Intensity (persons/m²)', shrink=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

    fig.suptitle('Intensity field comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Sensor placement
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensor_placement(floor_plan,
                           lambda_mean: torch.Tensor,
                           sensors_gp: torch.Tensor,
                           angles_gp: torch.Tensor,
                           sensors_score: torch.Tensor,
                           angles_score: torch.Tensor,
                           sdf: torch.Tensor,
                           save_path: str):
    from sensor_placement import detection_probability, miss_probability

    ext = [0, C.MALL_WIDTH_M, 0, C.MALL_HEIGHT_M]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (sensors, angles, title) in zip(axes, [
        (sensors_gp,    angles_gp,    'GP baseline placement'),
        (sensors_score, angles_score, 'Score model placement'),
    ]):
        # Background: mean intensity
        lam = lambda_mean.squeeze(0).numpy()
        wall = ~floor_plan.free_mask
        im = ax.imshow(lam, origin='lower', extent=ext,
                       cmap=INTENSITY_CMAP, interpolation='bilinear')
        wall_rgba = np.zeros((C.GRID_H, C.GRID_W, 4))
        wall_rgba[wall] = [0.17, 0.17, 0.16, 0.85]
        ax.imshow(wall_rgba, origin='lower', extent=ext,
                  interpolation='nearest')
        plt.colorbar(im, ax=ax, label='λ̄ (persons/m²)', shrink=0.85)

        # Coverage overlay
        pi = miss_probability(sensors, angles, sdf).numpy()
        coverage = 1.0 - pi
        cov_rgba = np.zeros((C.GRID_H, C.GRID_W, 4))
        cov_rgba[:, :, 2] = 0.4   # blue channel
        cov_rgba[:, :, 3] = coverage * 0.35
        ax.imshow(cov_rgba, origin='lower', extent=ext,
                  interpolation='bilinear')

        # Camera markers with FOV wedges
        colours = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9']
        for i, (xy, ang) in enumerate(zip(sensors, angles)):
            x_, y_ = xy[0].item(), xy[1].item()
            c = colours[i % len(colours)]
            ax.plot(x_, y_, '*', color=c, ms=14, zorder=10,
                    markeredgecolor='white', markeredgewidth=0.5)
            # FOV wedge
            ang_rad = math.radians(ang.item())
            fov_rad = math.radians(C.CAMERA_FOV_DEG / 2)
            for sign in [-1, 1]:
                edge_ang = ang_rad + sign * fov_rad
                ex = x_ + C.MAX_DETECTION_RANGE_M * math.cos(edge_ang)
                ey = y_ + C.MAX_DETECTION_RANGE_M * math.sin(edge_ang)
                ax.plot([x_, ex], [y_, ey], '-', color=c, alpha=0.5,
                        linewidth=1.0)
            ax.annotate(f'C{i+1}', (x_, y_), fontsize=7, color='white',
                        ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points')

        ax.set_xlim(0, C.MALL_WIDTH_M); ax.set_ylim(0, C.MALL_HEIGHT_M)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

    fig.suptitle('Camera Placement Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Optimisation convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_optimisation_convergence(history: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = [25 * i for i in range(len(history['vp_mc']))]
    if steps and steps[0] == 0:
        steps[0] = 1
    ax.plot(steps, history['vp_mc'],     label='VP (MC — score model)',
            color='#3266AD', linewidth=2)
    ax.plot(steps, history['vp_jensen'], label='VP (Jensen — GP approx)',
            color='#D85A30', linewidth=2, linestyle='--')
    ax.fill_between(steps, history['vp_jensen'], history['vp_mc'],
                    alpha=0.15, color='#3266AD', label='Jensen gap')
    ax.set_xlabel('Optimisation step')
    ax.set_ylabel('Void probability')
    ax.set_title('Void probability during sensor placement optimisation',
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Calibration
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(calibration: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    levels = sorted(calibration.keys())
    actual = [calibration[lv] for lv in levels]
    ideal  = [lv / 100.0 for lv in levels]

    ax.plot(ideal, ideal, 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(ideal, actual, 'o-', color='#3266AD', linewidth=2,
            markersize=8, label='Score model')
    for lv, act, idl in zip(levels, actual, ideal):
        ax.annotate(f'{lv}%  ({act:.2f})',
                    (idl, act), textcoords='offset points',
                    xytext=(8, 0), fontsize=8)

    ax.set_xlabel('Nominal coverage probability')
    ax.set_ylabel('Empirical coverage probability')
    ax.set_title('Credible interval calibration', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Miss rate comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_miss_rate_comparison(sim_results: dict, save_path: str):
    """
    sim_results : dict mapping method name → simulation_miss_rate output dict
    """
    methods   = list(sim_results.keys())
    miss_rates = [sim_results[m]['miss_rate'] * 100 for m in methods]
    void_probs = [sim_results[m]['void_prob_emp'] * 100 for m in methods]
    colors     = ['#73726c', '#3266AD'][:len(methods)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Miss rate bar
    ax = axes[0]
    bars = ax.bar(methods, miss_rates, color=colors, edgecolor='white',
                  linewidth=0.5)
    for bar, val in zip(bars, miss_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Miss rate (%)')
    ax.set_title('Event miss rate (lower = better)', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Empirical void probability
    ax = axes[1]
    bars = ax.bar(methods, void_probs, color=colors, edgecolor='white',
                  linewidth=0.5)
    for bar, val in zip(bars, void_probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Empirical void probability (%)')
    ax.set_title('P(zero misses) (higher = better)', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Simulation study: sensor placement comparison',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
