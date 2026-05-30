"""
validation.py
─────────────────────────────────────────────────────────────────────────────
Implements all four validation levels described in the research plan:

    Level 1 — Intensity estimation quality
        •  Negative log-likelihood (NLL) of the LGCP
        •  Visual comparison of true vs estimated intensity

    Level 2 — Uncertainty quantification
        •  Calibration: does the p% credible interval contain the true
           value p% of the time?

    Level 3 — Sensor placement quality
        •  Void probability (MC estimate vs Jensen approximation)
        •  Jensen gap analysis

    Level 4 — Simulation study
        •  miss_rate, void_prob_empirical, mean_missed_per_trial
        (implemented in sensor_placement.py, called from main)

All metrics are returned as dicts for easy logging / comparison.
"""

import math
import torch
import numpy as np
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — Negative log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

def compute_nll(lambda_pred: torch.Tensor,
                test_fields: list) -> float:
    """
    Compute the average negative log-likelihood of the LGCP on the test set.

    LGCP log-likelihood for intensity field λ̂ and observed field λ_true:
        log p ≈ Σₓ [λ_true(x) · log λ̂(x) − λ̂(x)] · Δx²

    This is the Poisson log-likelihood treating each cell as an independent
    Poisson observation with mean λ̂(x)·Δx².

    Lower NLL = better intensity estimate.

    Parameters
    ----------
    lambda_pred : (1, H, W) predicted mean intensity field
    test_fields : list of (1, H, W) test intensity fields

    Returns
    -------
    mean_nll : float — average NLL per observation
    """
    dx2 = C.CELL_AREA_M2
    lam_p = lambda_pred.squeeze(0).clamp(min=1e-8)   # (H, W)
    log_lam = torch.log(lam_p)

    nll_list = []
    for lam_true in test_fields:
        lam_t = lam_true.squeeze(0)   # (H, W)
        # Poisson log-likelihood
        ll = (lam_t * log_lam - lam_p).sum() * dx2
        nll_list.append(-ll.item())

    return float(np.mean(nll_list))


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — Calibration
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(lambda_samples: torch.Tensor,
                         test_fields: list,
                         levels: list = C.CALIBRATION_LEVELS) -> dict:
    """
    Compute the empirical coverage of credible intervals.

    For a well-calibrated model, the p% credible interval should contain
    the true value exactly p% of the time at each location.

    Parameters
    ----------
    lambda_samples : (K, 1, H, W) intensity samples from DDIM
    test_fields    : list of (1, H, W) true intensity fields
    levels         : list of integer percentages [50, 80, 95]

    Returns
    -------
    calibration : dict mapping level → empirical coverage (float ∈ [0,1])
                  Ideal values: 0.50, 0.80, 0.95
    """
    K = lambda_samples.shape[0]
    samples_flat = lambda_samples.squeeze(1).reshape(K, -1)   # (K, H*W)

    coverage = {lv: [] for lv in levels}

    for lam_true in test_fields:
        true_flat = lam_true.reshape(-1)   # (H*W,)

        for lv in levels:
            alpha = (1.0 - lv / 100.0) / 2.0
            lower = torch.quantile(samples_flat, alpha,      dim=0)
            upper = torch.quantile(samples_flat, 1.0 - alpha, dim=0)
            # Fraction of cells where true value is inside the interval
            inside = ((true_flat >= lower) & (true_flat <= upper)).float()
            coverage[lv].append(inside.mean().item())

    return {lv: float(np.mean(coverage[lv])) for lv in levels}


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — Jensen gap analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_jensen_gap(vp_mc: float, vp_jensen: float) -> dict:
    """
    Compute and interpret the Jensen gap.

    The Jensen gap is:
        J = E_λ[exp(−Λ̃(a))] − exp(−E_λ[Λ̃(a)])
          = VP_true − VP_Jensen

    A smaller gap means the Jensen approximation is tighter, i.e. using
    the mean intensity λ̄ instead of sampling full fields loses less
    information about the void probability.

    With the score model we compute VP_true directly (no Jensen needed),
    so the gap measures how much the GP baseline loses.

    Parameters
    ----------
    vp_mc     : float — true void probability (MC estimate)
    vp_jensen : float — Jensen lower-bound void probability

    Returns
    -------
    dict with 'absolute_gap', 'relative_gap_pct', 'interpretation'
    """
    abs_gap = vp_mc - vp_jensen
    rel_gap = abs_gap / max(vp_mc, 1e-8) * 100.0
    return {
        'vp_mc':           vp_mc,
        'vp_jensen':       vp_jensen,
        'absolute_gap':    abs_gap,
        'relative_gap_pct': rel_gap,
        'interpretation': (
            f"Jensen approximation underestimates VP by {rel_gap:.2f}% "
            f"(absolute: {abs_gap:.5f}). "
            + ("Tight approximation." if rel_gap < 5 else
               "Significant approximation error — score MC is preferred.")
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: dict):
    """
    Pretty-print a comparison table of validation metrics.

    Parameters
    ----------
    results : dict mapping method_name → metrics_dict
              Each metrics_dict should have keys:
              'nll', 'coverage_50', 'coverage_80', 'coverage_95',
              'vp_mc', 'vp_jensen', 'jensen_gap_pct',
              'miss_rate', 'void_prob_emp'
    """
    methods = list(results.keys())
    print()
    print("=" * 90)
    print("VALIDATION RESULTS")
    print("=" * 90)

    # Header
    col_w = 18
    header = f"{'Metric':<28}"
    for m in methods:
        header += f"{m:>{col_w}}"
    print(header)
    print("-" * 90)

    metrics = [
        ("NLL ↓",              "nll",            ".3f"),
        ("Coverage 50% (→0.50)", "coverage_50",  ".3f"),
        ("Coverage 80% (→0.80)", "coverage_80",  ".3f"),
        ("Coverage 95% (→0.95)", "coverage_95",  ".3f"),
        ("Void Prob MC ↑",     "vp_mc",          ".4f"),
        ("Void Prob Jensen ↑", "vp_jensen",      ".4f"),
        ("Jensen Gap % ↓",     "jensen_gap_pct", ".2f"),
        ("Miss Rate ↓",        "miss_rate",      ".4f"),
        ("Emp. VP ↑",          "void_prob_emp",  ".4f"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<28}"
        for m in methods:
            val = results[m].get(key, float('nan'))
            row += f"{val:{col_w}{fmt}}"
        print(row)

    print("=" * 90)
    print()
