# Figure index

DeepSeek-1.5B, FP32, window 100 unless stated; prompt `interstellar_travel`; 300 generated tokens (margins 600).
“Generated step” is after the prompt. All A1 runs use in-batch `r=0`.

## A1 / response / scaling

- **a1_response_individual.png** — 3 directions, last-layer `||δh_t||` for several r; gray dotted lines are t=16/123/143 near-ties.
- **a1_response_aggregate.png** — median/IQR over 20 directions.
- **a1_collapse_by_layer.png** — raw (left) and `δh/r` (right) for layers 1/2/3/7/14/27; curves collapse on the right for deep layers.
- **a1_scaling_collapse.png** — raw vs `/r` vs `/√r` vs `/r^α(t)`; `/r` is the winner.
- **a1_exponent_vs_time.png** — fitted exponent α(t) per layer; α≈0.95–0.98 for deep layers, lower late/early.
- **a1_exponent_vs_layer.png** — α at selected times vs layer.
- **a1_alpha_heatmap.png** — α(t,ℓ) heatmap.
- **a1_layers_fixed_r.png** — fixed r=0.1/0.3, response vs token for layers 0/1/2/3/7/14/21/27.
- **a1_layers_r0.1.png** — compact layer comparison at r=0.1.
- **a1_direction_corr.png** — pairwise correlation of prompt-direction curves; median 0.92, no strong groups.
- **margins.png** — baseline margin vs step (window 100).
- **response_vs_r.png** — median response vs r at selected steps; floor below r≈1e-4.

## Margins / attention window

- **margins_windows.png** — 2×2 margin traces for windows 50/100/200/500; minima annotated.
- **margins_vs_window.png** — min margin, count(m<0.05), median margin vs window.
- **window_comparison.png** — observed flips and prediction vs observation for windows 100/200.

## Thresholds / flux tube

- **threshold_geometry.png** — (left) margin vs gain with iso-r_c lines; (middle) r_* vs controlling near-tie; (right) predicted-vs-observed confusion matrix.
- **threshold_validation.png** — r_c for 3 directions and predicted-vs-observed counts.
- **rstar_distribution.png** — distribution/ECDF of r_* over directions.
- **rc_heatmap.png** — log10 r_c(t,u) over steps/directions, clipped; mostly robust except near-ties.
- **large_r_summary.png** — flip fraction at r=0.5/1/2 and token-diversity proxy; text remains non-degenerate at 300 tokens.

## FSLE / depth

- **fsle_time.png** — time-axis FSLE; λ≈0.043 for ρ=2, transient.
- **fsle_individual.png** — 3 separation curves at several r.
- **fsle_depth.png** — depth separation `d(layer)/d(0)`; first-layer jump then flat.
- **fsle_depth_detail.png** — separate y axes per r; shows the jump is one layer.
- **depth_local_gain_existing.png** — local depth gain; numerical-floor spikes, diagnostic only.
- **depth_individual.png**, **depth_aggregate.png**, **depth_spectrum.png**, **depth_local_gain.png**, **h_depth.png** — depth spectrum k=8; top exponents ~0.08–0.18, h_depth 0.6–1.4.
- **depth_w100_k16** (k=16) — all 16 exponents positive, h_depth 1.2–2.7.

## Non-normality

- **nonnorm_mlp_big.png** — MLP σ_max, ρ, κ vs selected layers, 6 tokens.
- **nonnorm_full_big.png** — full-block σ_max, ρ, κ at 5 layers × 6 tokens; layer 0 huge, layers 7–21 ~3, layer 27 6–9.
- **nonnorm_scatter_big.png** — σ_max vs ρ; points above diagonal are non-normal.
- **nonnorm_angles_big.png** — principal angles between consecutive top singular subspaces; median 82–85°.

## State / R(t,s)

- **state_response.png** — input-embedding injection at s=16/50/100; δh_t and R(t,s).
- **state_directions.png** — 32 directions per s, median/IQR.
- **state_direction_corr.png** — pairwise direction correlation; median 0.94–0.98.
- **state_shift.png** — R(t,s) vs lag τ; near-tie bumps at different lags.
- **state_mid_compare.png** — input embedding vs layer-14 residual injection; residual is ~10× weaker.

## D2 / RQA

- **d2_curves.png** — C(ε) and local slope; peak, not plateau.
- **d2_vs_N.png** — D2 vs N.
- **d2_distribution.png** — D2 across 12 prompts.
- **d2_theiler.png** — Theiler sensitivity.
- **d2_scaling_examples.png** — clean scaling examples.
- **d2_controls.png** — raw vs shuffled vs phase vs vocab; raw≈shuffled.
- **d2_raw_vs_shuffled.png** — all 12 prompts: raw D2 ≈ shuffled D2.
- **d2_delay_twonn.png** — delay-embedding D2 and TwoNN.
