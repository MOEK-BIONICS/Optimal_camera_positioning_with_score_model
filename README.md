# Score-Based LGCP Sensor Placement

## Overview
Complete implementation of the score-based diffusion model extension to the
Log-Gaussian Cox Process (LGCP) sensor placement framework from Kim et al. (2023, 2025).

## Files
| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters — edit here to change grid size, epochs, etc. |
| `data_generation.py` | Toy mall floor plan, SDF, trajectory simulator, intensity fields |
| `score_model.py` | Score U-Net architecture, training loop, Tweedie extraction, DDIM sampling |
| `sensor_placement.py` | Camera model, void probability (MC + Jensen), greedy init, gradient optimisation |
| `validation.py` | NLL, calibration, Jensen gap metrics |
| `visualisation.py` | All 8 paper figures |
| `main.py` | Full pipeline runner |

## How to run
```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn tqdm
python main.py
```

## Demo vs paper settings
The default config runs in ~3 minutes on CPU. For paper-quality results:
```python
# In config.py:
N_TRAIN_WINDOWS   = 1000   # more data
N_EPOCHS          = 300    # longer training
N_INTENSITY_SAMPLES = 128  # more MC samples
N_OPT_ITERATIONS  = 500   # more refinement
N_SIM_TRIALS      = 5000  # more simulation trials
```

## Pipeline steps
1. **Data** — Build synthetic mall: floor plan → SDF → trajectories → intensity fields
2. **Train** — Score U-Net learns ∇ log p(intensity field) with geometric conditioning
3. **Extract** — Tweedie's formula → mean intensity λ̄; DDIM sampling → K field samples
4. **Greedy** — Place M cameras sequentially maximising void probability reduction
5. **Refine** — Adam gradient ascent on Monte-Carlo void probability
6. **Validate** — NLL, calibration, Jensen gap, simulation miss-rate
7. **Figures** — 8 publication-ready PNG figures

## Key equations implemented
- Forward diffusion: xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε  (cosine schedule)
- Training loss: E[||ε̂_θ(xₜ,t,g) − ε||²]
- Tweedie: x̂₀ = (xₜ − √(1−ᾱₜ)·ε̂) / √ᾱₜ
- Void probability (MC): ν = (1/K)Σₖ exp(−Σₓ λₖ(x)·π(x,a)·Δx²)
- Void probability (Jensen): ν_J = exp(−Σₓ λ̄(x)·π(x,a)·Δx²)
- Miss probability: π(x,a) = ∏ᵢ(1 − γ(x,aᵢ))
- Detection: γ(x,a) = ρ·exp(−‖x−a‖²/2σ²)·angle_factor·LOS
