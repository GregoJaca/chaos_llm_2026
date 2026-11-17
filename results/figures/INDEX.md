# Figure index

DeepSeek-1.5B, FP32, window 100, prompt `interstellar_travel`, 300 generated tokens.
All A1 figures use the same 20 simplex directions. “Generated step” means token index after the prompt.

## A1 / response

- **a1_response_individual.png** — 3 individual directions: last-layer `||δh_t||` vs generated step, for r=1e-3, 1e-2, 0.1, 0.3. Shows the transient rise and decay.
- **a1_response_aggregate.png** — median and IQR over 20 directions for the same radii. The aggregate is smooth; the individual curves are noisy.
- **margins.png** — baseline top1–top2 margin vs step. The smallest margins are at t=123 (0.0202) and t=143 (0.0204).
- **response_vs_r.png** — median last-layer response vs r at selected steps. Shows a numerical floor below r≈1e-4 and near-linear scaling above r≈1e-3.

## Thresholds / flux tube

- **threshold_validation.png** — (left) `r_c(t,u)` for 3 directions, clipped at 100; (middle) `r_c` at t=16/123/143 across 20 directions; (right) predicted vs observed first-flip counts at r=0.1 and 0.3. Main result: no flips at r=0.1; at r=0.3 one direction flips at t=16, matching prediction.
- **predicted_T.png** — predicted first-flip time from teacher forcing at r=0.03/0.1/0.3.
- **rstar_distribution.png** — distribution/ECDF of the flux-tube radius `r_* = min_t r_c(t,u)` over directions.
- **rc_heatmap.png** — `log10 r_c(t,u)` over steps and directions, sorted by `r_*`. Mostly saturated (large thresholds) after t≈150; early steps and a few directions are fragile.

## FSLE

- **fsle_time.png** — time-axis FSLE from A1 separations; `λ≈0.043` for `ρ=2`, with censoring for `ρ=4`. This is a transient growth rate, not sustained exponential growth.
- **fsle_individual.png** — 3 individual separation curves at several r; shows rise then decay.
- **fsle_depth.png** — depth-axis separation `d(layer)/d(0)` from existing single-token hidden states. The first layer gives a huge jump (~200×), then the curves are flat/decaying. FSLE is trivial (τ=1) because the growth is one layer.

## Depth map / spectrum

- **depth_individual.png** — top-3 cumulative depth exponents vs layer for 3 tokens. They start high (~3) and decay to ~0.1; the first layer dominates.
- **depth_aggregate.png** — median/IQR top exponent vs layer.
- **depth_spectrum.png** — top-8 exponents at the final layer for several tokens. All positive, ~0.08–0.18.
- **depth_local_gain.png** — mean local layer gain `||J_l q||` vs layer. Layer 0 gain 10–20; later layers ≈1.1.
- **h_depth.png** — `h_depth = Σ_{λ_i>0} λ_i` per token, a lower bound from the top 8 directions.

## D2 / RQA

- **d2_curves.png** — correlation sum and local slope for one prompt (first, last, vocab). The local slope is a peak, not a plateau.
- **d2_vs_N.png** — D2 vs N for 3 prompts. First-layer D2 is ~20 and flat; last-layer decreases for some prompts but not consistently.
- **d2_distribution.png** — D2 across 12 prompts for first/last layer and the vocabulary null.
- **d2_theiler.png** — local slope and Theiler-window sensitivity. Theiler 0/10/50 does not create a plateau.

## Non-normality

- **nonnorm_mlp.png** — MLP `σ_max`, `ρ`, `κ` vs layer for several tokens.
- **nonnorm_full.png** — full-block `σ_max`, `ρ`, `κ` at selected layers/tokens.
- **nonnorm_scatter.png** — `σ_max` vs `ρ`; points above the diagonal are non-normal.
- **nonnorm_angles.png** — principal angles between top singular subspaces of consecutive layers.
