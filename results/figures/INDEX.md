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

- **threshold_margin_gain.png** — margin vs signed gain, iso-r_c lines, observed flips; best mechanism figure.
- **threshold_control_map.png** — signed `r_c⁻(t,u)` heatmaps, rows sorted by danger; which near-tie controls each direction (t=44 for w≥200, t=123 for w100).
- **threshold_ecdf.png** — signed `r_*⁻` ECDF per window; median 1.7 (w100) vs 0.26 (w200/500).
- **threshold_window_spectrum.png** — margin traces per window; floor collapses 50× while trajectories stay close.
- **threshold_prediction_scorecard.png** — observed vs predicted class table and diagonal plot; 19–20/20 at r≤0.1, breakdown at r≥1.
- **threshold_linearity.png** — gain growth and threshold drift with r; small-r thresholds do not extrapolate.
- **threshold_tubes_cases.png** — 3 individual threshold tubes (fragile, typical, resistant).
- **threshold_geometry.png** — earlier symmetric-|δm| version; kept for comparison.
- **threshold_validation.png** — earlier r_c and counts.
- **large_r_summary.png** — flip fraction at r=0.5/1/2 and token-diversity proxy.
- **prompt_generality.png** — flip fraction at r=0.1/0.3 for interstellar/childhood/cybersecurity; controlling near-ties annotated.

Subagent report: `results/subagent_threshold/REPORT.md`.

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

## A1 subagent deep dive

- **a1_universal_curve.png** — `G(t)=δh/r` universal curve; 7 radii collapse (CV 2–3%); 3 directions; direction gain differs.
- **a1_window_causality.png** — w100 jump ×1.73 at t=35, w200 ×1.51 at t=135, wider windows byte-identical controls; window-driven regimes.
- **a1_regime_fits.png** — rise/plateau/surge/tail fits; flush times; exponential vs power-law.
- **a1_depth_layers.png** — depth sets gain/rise time/memory; gain `A(ℓ)~ℓ^0.64`, front-loaded.
- **a1_direction_gain.png** — directions: same shape, scalar gain ×1.13–18.97; dir10 strongest.
- **a1_collapse_L27.png** — raw vs /r at L27; r≥0.5 peels up.
- **a1_large_r_subagent.png** — large-r shape/superlinearity; no saturation, bounded.
- **a1_collapse_quality.png** — collapse CV by layer/time.
- **a1_direction_corr.png** — pairwise prompt-direction correlation.

## Non-normality / depth: scale-normalized

- **depth_scale_normalization.png** — headline: absolute growth vs state-scale growth; scale-normalized gain <1 at every block; FSLE ×206 absolute → ×0.46 relative.
- **nonnorm_gain_budget.png** — component σ_max, misalignment model, measured aligned gain, scale growth; chained bounds.
- **nonnorm_mechanism.png** — toy non-normal rotation mechanism.
- **depth_gain_heatmaps.png** — layer/token gain structure.
- **depth_growth_subagent.png** — depth growth.
- **nonnorm_big_report.png** — σ_max/ρ/κ and angles.

## State subagent deep dive

- **state_response_field.png** — (s,t) response field: causal wedge, ×13 gain variation, fast-drop/slow-tail, bumps.
- **state_lag_view.png** — lag-coordinate field and collapse failure (×1.58).
- **state_direction_corr_subagent.png** — direction correlation heatmaps; no clusters.
- **state_mid_vs_emb.png** — input vs layer-14 injection; mid is 9–36× weaker, flatter tail.
- **state_collapse_subagent.png** — all collapse attempts and failure factors.
- **state_s_dependence.png**, **state_r_scaling_subagent.png** — gain/decay vs s and r proxy.
- **state_r_scaling.png**, **state_r_decay.png** — new state r-scan (1e-3…3e-2); response linear in r.
- **attention_entropy.png** — attention entropy vs token/margin for windows 50/100/200; weak/mixed relation to near-ties (exploratory negative).

## Signed threshold (best threshold set)

- **threshold_margin_gain.png** — margin vs signed gain, iso-r_c lines, observed flips.
- **threshold_control_map.png** — signed `r_c⁻` heatmaps, rows sorted by danger; controlling near-tie.
- **threshold_prediction_scorecard.png** — observed vs predicted classes; 19–20/20 at r≤0.1.
- **threshold_tubes_cases.png** — 3 individual threshold tubes (fragile/typical/resistant).
- **threshold_ecdf.png**, **threshold_window_spectrum.png**, **threshold_linearity.png**.

## D2 / RQA

- **d2_loops_scan.png** — state distance vs lag; cyber/future/great collapse at P=53/74/69 (repetition loops); non-loop prompts do not. Reframes the dataset.
- **d2_scaling_subagent.png** — C(ε) on a shared normalized axis for N=1024/2048/full + shuffled overlay; no plateau.
- **d2_slopes_subagent.png** — local slopes vs ε, Theiler variants; slopes decline monotonically.
- **d2_synthloop_C.png** — statics-only synthetic reproduces the loop C(ε) curve.
- **d2_synthloop_pred.png** — trivial cycle+drift synthetic reproduces NSE≈0.999; statics alone predicts nothing.
- **d2_intrinsic_dim.png** — TwoNN/MLE/PCA vs nulls; locally thin, globally spread; finite-N bias.
- **d2_prediction.png**, **d2_continuation.png** — non-loop dynamics: launch NSE gap, no continuation sharpness; interstellar null.
- **d2_curves.png**, **d2_vs_N.png**, **d2_distribution.png**, **d2_theiler.png**, **d2_controls.png**, **d2_raw_vs_shuffled.png**, **d2_delay_twonn.png** — earlier D2/RQA diagnostics.

## Subagent reports

- `results/subagent_a1/REPORT.md`
- `results/subagent_threshold/REPORT.md`
- `results/subagent_state/REPORT.md`
- `results/subagent_nonnorm/REPORT.md`
- `results/subagent_d2/REPORT.md`

## Round 3 (new, after user feedback)

- **a1_window_fixed_inputs.png** — same prompt AND same tokens, five windows; each curve is bit-identical to the no-clipping control until `window - prompt_len`, then departs; panel (c) shows the law on two prompts (65- and 62-token prompts): 35/85/135/235 and 38/138.
- **a1_window_natural_experiment.png** — windows 100/200/500 with their own generated text: the response ratio is exactly 1.000 until the clip step; generated text splits 4-6 tokens later; departure-step panel.
- **a1_window_regimes.png** — four windows, separate y axis per panel, clip steps marked, nothing clipped.
- **tf_is_exact_until_flip.png** — one teacher-forced pass vs the free run: 29/29 flips exact, 0 false positives, 0 misses in 140 cases; token timeline shows the shared first flip.
- **state_current_field.png** — single-token kick: (s,t) field, decay vs lag, immediate size x13 across positions, linear in r (slope 0.99-1.00).
- **state_current_depth.png** — kick amplification with depth; relative response; which directions change a token; latency 107 tokens (kick at s=16 acts at t=123).
- **state_kick_latent_validated.png** — autoregressive validation: 5/5 exact, same directions, same steps; cumulative agreement stays 1.0 until t=123.
- **flip_is_a_close_call.png** — two kinds of flips: 17 immediate at the response peak, 10 delayed at t=116/123/143 when the response has faded.
- **directions_are_not_equivalent.png** — signed margin change at the same close call: 7 directions shrink the gap, 13 widen it; flips are stratified (3/20 at r=0.5, from the hot trio).
- **depth_angle_test.png** — the perturbation is 86-87 degrees from the state (nearly perpendicular), so "radial drift" was wrong; the perpendicular part dominates.
- **depth_plain_language.png** — plain-language schematic: state length vs direction change, and why the output stage is length-blind.
- **threshold_explainer.png** — what is predicted, built from real data (gap trace, shrink, overtake line, worked example).
- **threshold_rule_scorecard_clean.png** — signed rule vs sign-blind rule on the same 320 direction-runs (99.4% vs 89.1%).
- **threshold_scorecard_full.png** — predicted vs observed word for all 75 flips + outcome strips for 380 runs.
- **threshold_value_survival.png** — value: 0/75 flips ever re-sync, 2.7% agreement afterwards; surviving fraction = first-flip step / 300.
- **threshold_danger_ladder.png** — replaces the control map: one row per direction, threshold bar, controlling word, sign-blind correction.
- **threshold_when_it_breaks.png** — noise floor / linear window / superlinear regimes, and what that does to extrapolation.
- **threshold_three_cases.png** — three individual directions (fragile, near-tie, resistant).
- **d2_artifact_gallery.png** — mechanisms that produce "good-looking" correlation sums on non-chaotic data (loops, duplicates, cdist self-pairs, linear eps grid).
- **d2_delay_embedding.png** — m=2 delay embeddings look textbook but match white noise.
- **d2_shape_vs_N.png** — shape vs N and eps normalization.
- **directions_gain_consistency.png** — gain is a stable property of a direction across different continuations.
