"""
main.py
─────────────────────────────────────────────────────────────────────────────
Complete pipeline runner for Score-Based LGCP Sensor Placement.

Run this file to execute the full experiment:

    python main.py

What happens
────────────
  Step 0  — Build toy mall dataset (floor plan, SDF, trajectories, fields)
  Step 1  — Train score U-Net on intensity fields
  Step 2  — Extract mean intensity via Tweedie + sample K fields via DDIM
  Step 3  — GP baseline: empirical mean intensity field
  Step 4  — Greedy sensor initialisation (both methods)
  Step 5  — Gradient-based sensor refinement (score model only)
  Step 6  — Validation: NLL, calibration, Jensen gap, simulation
  Step 7  — Generate all paper figures
  Step 8  — Print results table

Outputs (written to ./outputs/)
────────────────────────────────
  fig1_floor_plan.png
  fig2_trajectories.png
  fig3_training_loss.png
  fig4_intensity_comparison.png
  fig5_sensor_placement.png
  fig6_convergence.png
  fig7_calibration.png
  fig8_miss_rate.png
  results_summary.txt
"""

import os
import time
import torch
import numpy as np

import config as C
from data_generation   import MallDataset, MallFloorPlan, compute_sdf, \
                              TrajectorySimulator, build_intensity_field
from score_model       import (ScoreUNet, cosine_alpha_bar,
                               train_score_model,
                               estimate_mean_intensity, ddim_sample)
from sensor_placement  import (greedy_initialise, optimise_sensors,
                               void_probability_mc, void_probability_jensen,
                               simulation_miss_rate)
from validation        import (compute_nll, compute_calibration,
                               compute_jensen_gap, print_results_table)
from visualisation     import (plot_floor_plan, plot_trajectories,
                               plot_training_loss, plot_intensity_comparison,
                               plot_sensor_placement,
                               plot_optimisation_convergence,
                               plot_calibration, plot_miss_rate_comparison)

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def out(fname):
    return os.path.join(OUT_DIR, fname)


# ─────────────────────────────────────────────────────────────────────────────
def banner(text):
    width = 70
    print()
    print("─" * width)
    print(f"  {text}")
    print("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(C.RANDOM_SEED)
    np.random.seed(C.RANDOM_SEED)
    device = C.DEVICE
    print(f"\n{'='*70}")
    print("  Score-Based LGCP Sensor Placement  —  Full Pipeline")
    print(f"  Device: {device}")
    print(f"  Grid:   {C.GRID_H}×{C.GRID_W} cells  ({C.GRID_RES_M}m/cell)")
    print(f"{'='*70}")

    # ── Step 0: Data ──────────────────────────────────────────────────────────
    banner("Step 0 — Building toy mall dataset")
    t0 = time.time()

    train_ds = MallDataset(split='train', verbose=True)
    val_ds   = MallDataset(split='val',   verbose=True)
    test_ds  = MallDataset(split='test',  verbose=True)

    floor_plan = train_ds.floor_plan
    geometry   = train_ds.geometry          # (4, H, W)
    sdf        = train_ds.sdf               # (H, W)
    free_mask  = torch.tensor(floor_plan.free_mask)

    # Compute SDF again to get gradients for visualisation
    sdf_t, sdf_gx, sdf_gy = compute_sdf(floor_plan)

    # Build a few sample trajectories for Fig 2
    rng = np.random.default_rng(0)
    sim = TrajectorySimulator(floor_plan, sdf_t, sdf_gx, sdf_gy)
    sample_trajs = sim.generate_window(60, rng)

    print(f"  Dataset built in {time.time()-t0:.1f}s")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Fig 1 & 2
    banner("Step 0b — Figures 1 & 2")
    plot_floor_plan(floor_plan, sdf_t, out("fig1_floor_plan.png"))
    plot_trajectories(floor_plan, sample_trajs, out("fig2_trajectories.png"))

    # ── Step 1: Train score model ─────────────────────────────────────────────
    banner("Step 1 — Training Score U-Net")
    t1 = time.time()

    model, alpha_bar, history = train_score_model(
        train_ds, val_ds,
        n_epochs   = C.N_EPOCHS,
        batch_size = C.BATCH_SIZE,
        lr         = C.LR_SCORE,
        device     = device,
        verbose    = True,
    )

    print(f"  Training complete in {time.time()-t1:.1f}s")
    plot_training_loss(history, out("fig3_training_loss.png"))

    # Save model checkpoint
    ckpt_path = out("score_model.pt")
    torch.save({'model': model.state_dict(),
                'alpha_bar': alpha_bar}, ckpt_path)
    print(f"  Model saved → {ckpt_path}")

    # ── Step 2: Intensity extraction ──────────────────────────────────────────
    banner("Step 2 — Intensity extraction (Tweedie + DDIM)")
    t2 = time.time()

    # Mean intensity via Tweedie's formula
    lambda_score_mean = estimate_mean_intensity(
        model, geometry, alpha_bar,
        n_samples=128, t_eval=50, device=device
    )   # (1, H, W)

    # K intensity realisations for Monte-Carlo void probability
    print(f"  Sampling {C.N_INTENSITY_SAMPLES} intensity fields via DDIM ...")
    lambda_samples = ddim_sample(
        model, geometry, alpha_bar,
        K=C.N_INTENSITY_SAMPLES, ddim_steps=C.DDIM_STEPS, device=device
    )   # (K, 1, H, W)

    print(f"  Intensity extraction done in {time.time()-t2:.1f}s")
    print(f"  Score mean λ̄  range: [{lambda_score_mean.min():.3f}, "
          f"{lambda_score_mean.max():.3f}]")

    # ── Step 3: GP baseline (empirical mean) ──────────────────────────────────
    banner("Step 3 — GP baseline (empirical mean intensity)")
    # The empirical mean over all training windows is the GP baseline
    # equivalent — it is what INLA would produce given perfect data.
    lambda_gp_mean = train_ds.get_mean_intensity()   # (1, H, W)
    print(f"  GP mean λ̄  range: [{lambda_gp_mean.min():.3f}, "
          f"{lambda_gp_mean.max():.3f}]")

    # True intensity: average over test fields (ground truth approximation)
    test_fields = [test_ds[i][0] for i in range(len(test_ds))]
    lambda_true = torch.stack(test_fields).mean(dim=0)   # (1, H, W)

    plot_intensity_comparison(lambda_true, lambda_gp_mean, lambda_score_mean,
                              floor_plan, out("fig4_intensity_comparison.png"))

    # ── Step 4: Greedy initialisation ─────────────────────────────────────────
    banner("Step 4 — Greedy sensor initialisation")

    print("\n  [GP baseline — greedy on λ̄_GP]")
    t4 = time.time()
    sensors_gp, angles_gp = greedy_initialise(
        lambda_gp_mean, free_mask, sdf_t,
        M=C.N_SENSORS, verbose=True
    )
    print(f"  GP greedy done in {time.time()-t4:.1f}s")

    print("\n  [Score model — greedy on λ̄_score]")
    t4b = time.time()
    sensors_score_init, angles_score_init = greedy_initialise(
        lambda_score_mean, free_mask, sdf_t,
        M=C.N_SENSORS, verbose=True
    )
    print(f"  Score greedy done in {time.time()-t4b:.1f}s")

    # ── Step 5: Gradient-based refinement (score model) ───────────────────────
    banner("Step 5 — Gradient-based sensor refinement (Score model)")
    t5 = time.time()

    sensors_score, angles_score, opt_history = optimise_sensors(
        sensors_score_init, angles_score_init,
        lambda_samples, lambda_score_mean,
        sdf_t, free_mask,
        n_iterations = C.N_OPT_ITERATIONS,
        lr_xy        = C.LR_SENSORS,
        lr_ang       = C.LR_ANGLES,
        verbose      = True,
    )
    print(f"  Refinement done in {time.time()-t5:.1f}s")

    plot_optimisation_convergence(opt_history, out("fig6_convergence.png"))
    plot_sensor_placement(
        floor_plan, lambda_score_mean,
        sensors_gp, angles_gp,
        sensors_score, angles_score,
        sdf_t, out("fig5_sensor_placement.png")
    )

    # ── Step 6: Validation ────────────────────────────────────────────────────
    banner("Step 6 — Validation")

    # NLL
    nll_gp    = compute_nll(lambda_gp_mean,    test_fields)
    nll_score = compute_nll(lambda_score_mean, test_fields)
    print(f"  NLL — GP: {nll_gp:.3f}   Score: {nll_score:.3f}")

    # Calibration
    print("  Computing calibration ...")
    cal = compute_calibration(lambda_samples, test_fields,
                               levels=C.CALIBRATION_LEVELS)
    print(f"  Calibration: {cal}")
    plot_calibration(cal, out("fig7_calibration.png"))

    # Void probabilities
    vp_gp_jensen = void_probability_jensen(
        sensors_gp, angles_gp, lambda_gp_mean, sdf_t).item()

    vp_score_mc  = void_probability_mc(
        sensors_score, angles_score, lambda_samples, sdf_t).item()

    vp_score_jensen = void_probability_jensen(
        sensors_score, angles_score, lambda_score_mean, sdf_t).item()

    gap_gp    = compute_jensen_gap(vp_gp_jensen,    vp_gp_jensen)     # GP has no MC
    gap_score = compute_jensen_gap(vp_score_mc,     vp_score_jensen)
    print(f"\n  VP results:")
    print(f"    GP     (Jensen):         {vp_gp_jensen:.4f}")
    print(f"    Score  (MC):             {vp_score_mc:.4f}")
    print(f"    Score  (Jensen):         {vp_score_jensen:.4f}")
    print(f"    Jensen gap (score):      {gap_score['absolute_gap']:.5f}  "
          f"({gap_score['relative_gap_pct']:.2f}%)")

    # Simulation study
    print(f"\n  Running simulation study ({C.N_SIM_TRIALS} trials each) ...")
    sim_gp = simulation_miss_rate(
        sensors_gp, angles_gp, test_fields, sdf_t, n_trials=C.N_SIM_TRIALS)
    sim_score = simulation_miss_rate(
        sensors_score, angles_score, test_fields, sdf_t, n_trials=C.N_SIM_TRIALS)

    print(f"\n  Simulation results:")
    print(f"    GP    miss rate: {sim_gp['miss_rate']*100:.3f}%   "
          f"VP_emp: {sim_gp['void_prob_emp']:.4f}")
    print(f"    Score miss rate: {sim_score['miss_rate']*100:.3f}%   "
          f"VP_emp: {sim_score['void_prob_emp']:.4f}")

    sim_results = {'GP baseline': sim_gp, 'Score model': sim_score}
    plot_miss_rate_comparison(sim_results, out("fig8_miss_rate.png"))

    # ── Step 7: Results table ─────────────────────────────────────────────────
    banner("Step 7 — Results table")

    results = {
        'GP baseline': {
            'nll':             nll_gp,
            'coverage_50':     cal.get(50, float('nan')),  # GP has no CI
            'coverage_80':     cal.get(80, float('nan')),
            'coverage_95':     cal.get(95, float('nan')),
            'vp_mc':           vp_gp_jensen,  # GP uses Jensen as proxy
            'vp_jensen':       vp_gp_jensen,
            'jensen_gap_pct':  0.0,
            'miss_rate':       sim_gp['miss_rate'],
            'void_prob_emp':   sim_gp['void_prob_emp'],
        },
        'Score model': {
            'nll':             nll_score,
            'coverage_50':     cal.get(50, float('nan')),
            'coverage_80':     cal.get(80, float('nan')),
            'coverage_95':     cal.get(95, float('nan')),
            'vp_mc':           vp_score_mc,
            'vp_jensen':       vp_score_jensen,
            'jensen_gap_pct':  gap_score['relative_gap_pct'],
            'miss_rate':       sim_score['miss_rate'],
            'void_prob_emp':   sim_score['void_prob_emp'],
        }
    }
    print_results_table(results)

    # ── Step 8: Save text summary ─────────────────────────────────────────────
    summary_path = out("results_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Score-Based LGCP Sensor Placement — Results Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("CONFIG\n")
        f.write(f"  Grid:            {C.GRID_H}×{C.GRID_W} at {C.GRID_RES_M}m/cell\n")
        f.write(f"  Mall dimensions: {C.MALL_HEIGHT_M}m × {C.MALL_WIDTH_M}m\n")
        f.write(f"  Train windows:   {C.N_TRAIN_WINDOWS}\n")
        f.write(f"  Epochs:          {C.N_EPOCHS}\n")
        f.write(f"  Cameras (M):     {C.N_SENSORS}\n")
        f.write(f"  DDIM samples K:  {C.N_INTENSITY_SAMPLES}\n\n")

        f.write("SENSOR LOCATIONS\n")
        f.write("  GP baseline:\n")
        for i, (xy, ang) in enumerate(zip(sensors_gp, angles_gp)):
            f.write(f"    Camera {i+1}: ({xy[0]:.1f}, {xy[1]:.1f}) m  "
                    f"angle={ang.item():.0f}°\n")
        f.write("  Score model:\n")
        for i, (xy, ang) in enumerate(zip(sensors_score, angles_score)):
            f.write(f"    Camera {i+1}: ({xy[0]:.1f}, {xy[1]:.1f}) m  "
                    f"angle={ang.item():.0f}°\n")

        f.write("\nMETRICS\n")
        for method, metrics in results.items():
            f.write(f"\n  {method}:\n")
            for k, v in metrics.items():
                f.write(f"    {k:<22} {v:.5f}\n")

        f.write("\nJENSEN GAP ANALYSIS\n")
        f.write(f"  {gap_score['interpretation']}\n")

    print(f"\n  Results summary saved → {summary_path}")

    banner("DONE")
    print(f"  All outputs in: {OUT_DIR}/")
    print("  Files: fig1–fig8 PNGs, score_model.pt, results_summary.txt\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
