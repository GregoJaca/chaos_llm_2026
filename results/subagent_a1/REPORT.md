# A1 perturbation response: collapse, regimes, depth, directions, large r

Data: `results/local_data/w100_layers/a1.npz` (primary; 12 radii 1e-3..2.0, 20 simplex
directions, layers {0,1,2,3,7,14,21,27}, t=0..299) plus `w200/a1.npz`, `w500/a1.npz`
for window-size tests. All analysis: CPU, `s1..s6*.py` in this folder rerun everything
(`common.py` holds the loader; each script prints its numbers and writes the CSV/txt
of the same name).

**Data semantics (verified, worth knowing before any further plot):**
- `dh_norm[r,dir,l,t]` = `||h_base(l,t)|| * sqrt(1 - cos^2)` — the *absolute* L2
  deviation of the hidden state in units of the embedding norm (checked against the
  `cos` key to <1.5% via the constant 1.219 = ||h_emb||). It is **not** normalized
  per layer, so it is directly comparable across depth. Layer 0 is identically zero
  (the perturbation enters after the embedding output).
- Noise floor: float32, layer-dependent, ~7e-7 * ||h_base(l)||: 1.5e-5 (L1) .. 3.9e-4
  (L27) in dh_norm units. r=1e-3 hits it from t~52 (L1-2) / t~135-246 (L3-27)
  (11-51% of times); r >= 0.03 is clean everywhere. All fits below mask 3x floor.

## Robust findings

1. **The vertical shifts are exactly r: one exponent, alpha = 1.**
   For r <= 0.3 the response is *linear*: `dh(t,l;r) = r * G(t,l)` with collapse CV of
   dh/r = **2.0-3.3% (median)**, p90 <= 11%, at *every* layer and *every* t
   (`collapse_stats.csv`, fig1, fig2a). Per direction too (CV 2.6-6.4%). The fitted
   exponent over r in [1e-3, 0.3] is 1.00 +- 0.02 at all (l,t). **No time-dependent
   or layer-dependent exponent is needed in the linear regime** — the curves the
   user saw as "vertically shifted" are literally `r x` one universal curve; dividing
   by r collapses them (fig1 right column).

2. **The "two regimes" are set by the sliding attention window, not by intrinsic
   dynamics.** Shallow layers (L1-3) go *bitwise zero* (exact float32 zero, not a
   floor) at t* = 92, 191, 290 — spacing 99 = window-1: the perturbed KV entries are
   evicted one window-length later per layer (fig5e). Deep layers keep a residual
   tail (self-perpetuating perturbed KV) decaying at -0.011..-0.013 /token.
   Causality is confirmed by two *unconfounded clip events* (fig8): w100's response
   jumps **x1.73 at t=35** (the step where context exceeds 100 and the oldest prompt
   token is dropped) while w200/w500 stay byte-identical; w200 jumps **x1.51 at
   t=135** with w500 (not yet clipping) as control. The free-run token streams
   diverge only *after* clipping (t=40, t=142), so these events are clean.
   Tail decay rates fall with window size (-0.0100, -0.0075, -0.0009 /token for
   w100/w200/w500) — consistent, but confounded after t=40 (different streams).

3. **Rise vs decay asymmetry.** "Rise then decay" is real but layer-dependent:
   shallow layers peak at t=0 and decay slowly (-0.002..-0.021 /token); only deep
   layers rise (L27: +0.0225/token to a peak at t=17, x1.5) and they keep *growing*
   (+0.019/token) into a **pre-flush surge** (x1.9-2.5 at t~70-90, all layers >= L3)
   before the flush-driven decay starts. |rise| ~ 2x |tail decay| but the decay lasts
   >200 tokens vs 17 — the decay is not the time-reverse of the rise
   (`regime_rates.csv`, fig4b). Tail fits: neither law is decisive — exponential
   R^2 = 0.95 vs power-law (t^-2.2) R^2 = 0.97 at L27; power slightly better at
   l >= 14, exponential slightly better at L7 (fig4c,d).

4. **Depth is the second separator: A(l) ~ l^0.64.** Instantaneous gain
   A(l) = dh(t=0,l)/r grows 0.155 (L1) -> 1.875 (L27): power law l^0.636 fits better
   (R^2 = 0.95) than exponential e^{0.076 l} (R^2 = 0.89) because growth is
   *front-loaded*: marginal gain 1.90/layer at L1->2, ~1.04 mid-network, 1.12 again
   at L21->27 (`depth_stats.csv`, fig5c,d).

5. **Directions: a near-orthogonal simplex, but gains span x16.** The 20 directions
   have pairwise cosine ~ -1/19 (regular simplex), yet the model amplifies them
   1.13..18.97 at L27 (dir 10 ~ 10x median; dirs 1, 11 next; max/min ratio
   12.9-16.8 at *every* layer — depth-stable anisotropy, fig6a,c). To first order a
   direction contributes a *scalar gain* on a shared envelope (72% of log-variance
   in one mode), but token-level oscillations are direction-specific: pairwise
   correlation 0.58 (t<85) vs 0.88 (t<250) — the response is **not rank-1** in
   perturbation direction (fig6b,d).

6. **Large r: no saturation, no new time regime; superlinear with direction-
   dependent exponent.** For r >= 0.5 deep layers grow superlinearly:
   alpha(l, t=0) = 0.96 (L1) -> 1.45 (L27); the exponent drifts mildly and
   non-monotonically in t (L27: 1.45 -> 1.10 by t=200). The *shape* survives
   (median CV of dh(t)/dh(0) across r in {0.1..2}: 13%), and the largest deviations
   sit inside the pre-flush surge: max dh = 362 at r=2 (L27, dir 3, t=78), ~0.7x the
   base norm, cos -> 0.81 — big but bounded; no runaway (fig7). Per direction the
   large-r exponent spans **0.47 (saturating, dir 11) to 2.0 (dir 0)**,
   uncorrelated with the linear gain (Pearson -0.37).

7. **Prediction flips are a margin property, not a response-magnitude property.**
   Flips (pred_T) onset at r~0.2 (5% of directions), 35% at r=0.5, 95% at r >= 1.5,
   and every observed flip time {0, 2, 14, 16, 123, 143} has base margin <= 0.33
   (median margin ~1.9); the two smallest margins (0.020 at t=123, 143) flip first
   and most often (fig7d). Response magnitude only weakly tracks margins
   (corr(log dh, log margin) ~ -0.38).

## Negative results / failed collapses (quantified)

- **A single global power law in r fails.** At L27 over all 12 radii the best single
  exponent is alpha = 1.077 with max residual **0.30 decades** (factor ~2), while
  the regime split (alpha=1 for r<=0.3; per-layer alpha for r>=0.5) leaves residuals
  0.03-0.05 decades (fig2b,c). `dh * r^const`-type normalizations cannot fix a
  *multiplicative* shift and were not pursued further.
- **A time-dependent exponent is not supported** in the linear regime (alpha = 1.00
  +- 0.02 at all t); in the nonlinear regime alpha(l,t) drifts by ~0.3 with no clear
  temporal structure (fig2d) — a per-layer alpha(l) is the parsimonious description.
- **No single decay law**: exponential vs power-law tails differ by Delta R^2 ~
  0.02 (both >= 0.83) — do not claim either.
- **Cross-window tail-rate comparison is confounded** (streams diverge at t=40);
  only the clip-event jumps (t=35, t=135) are clean evidence.
- Shallow layers are slightly *sublinear* at large r (L1: alpha = 0.96 at t=0,
  0.79 at t=50) — "linear up to r=2" is false at the first layers.

## Recommended figures (ranked)

1. `fig1_collapse_L27.png` — the direct answer to the "vertical shifts" complaint:
   raw (left) vs /r (right) for the final layer; aggregate + dirs 0/6/10; the 7
   radii <= 0.3 collapse to one curve, r >= 0.5 peels off.
2. `fig8_windows.png` — *the* mechanism figure: the response changes regime exactly
   when the window starts clipping (t=35, t=135), controls included.
3. `fig5_depth.png` — depth dependence done properly: G(t) by layer, (l,t) heatmap
   with the flush front, A(l) with fits, flush times, per-layer gains.
4. `fig7_large_r.png` — large-r regime + flips-vs-margins.
5. Supporting: `fig2_collapse_quality.png` (collapse error + exponent analysis),
   `fig3_universal_curve.png` / `fig4_regime_fits.png` (regimes + fits),
   `fig6_directions.png` (anisotropy).

## Files

- Figures: `fig1_collapse_L27.png`, `fig2_collapse_quality.png`,
  `fig3_universal_curve.png`, `fig4_regime_fits.png`, `fig5_depth.png`,
  `fig6_directions.png`, `fig7_large_r.png`, `fig8_windows.png`.
- Scripts (rerun in this order): `common.py`, `s1_collapse.py`, `s2_regimes.py`,
  `s3_layers.py`, `s4_directions.py`, `s5_large_r.py`, `s6_windows.py`.
- Numbers: `collapse_stats.csv`, `regime_rates.csv`, `depth_stats.csv`,
  `direction_stats.csv`, `large_r_stats.txt`, `window_stats.txt`.
- Note: I could not visually inspect the rendered PNGs (no image input in this
  session); axis ranges, legends and annotations were set from the underlying
  numbers, but a quick human glance before publication is prudent.

## One-paragraph summary for the main agent

The A1 response has a clean separable description in its linear regime:
`dh(t,l;r,j) ~ r * A(l) * g_j * Phi(t,l)` for r <= 0.3 — exactly one exponent in r
(collapse CV 2-3% at every layer/time/direction; the user's "vertically shifted"
curves are one universal curve times r, so normalizing by r is the right and
sufficient fix), a depth gain A(l) ~ l^0.64 that is front-loaded (x12 over 27
layers, first 3 layers do most of it), and a direction gain g_j spanning x16
within a near-orthogonal simplex (so "random direction" assumptions fail; direction
is a scalar gain on a shared envelope plus direction-specific token oscillations).
The temporal "rise then decay" is governed by the finite attention window: deep
layers rise (t_peak ~ 17) and surge ~x2 just before the injected KV approaches the
window edge, shallow layers flush *bitwise* at t* = 92 + 99(l-1), and the tail
decays at ~ -0.012/token (power-law vs exponential undecidable, Delta R^2 ~ 0.02);
two clean clip events (response jumps x1.73 at t=35 for w100 and x1.51 at t=135 for
w200, with larger windows as untouched controls) prove the window, not the
perturbation or intrinsic chaos, sets the decay regime. Above r ~ 0.5 the response
stays shape-preserving but becomes superlinear with a depth- and direction-
dependent exponent (aggregate alpha up to 1.45 at L27; per direction 0.47-2.0,
uncorrelated with the linear gain), reaching ~0.7x the base norm at the r=2 surge
with no saturation and no runaway; prediction flips (onset r~0.2, 95% by r~1.5)
occur exactly at the base run's low-margin tokens and are therefore a property of
the decision geometry, not of the perturbation size. Recommended headline figures:
`fig1_collapse_L27.png`, `fig8_windows.png`, `fig5_depth.png`.
