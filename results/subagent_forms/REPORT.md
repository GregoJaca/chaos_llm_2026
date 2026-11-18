# Functional forms of perturbation growth — DeepSeek-1.5B (w100 teacher-forced + free runs)

**Scope.** All results from local data in `results/local_data/` (no GPU, no ssh): `w100_layers/a1.npz` (12 radii × 20 directions × 8 layers × 300 tokens), `window_fixed/w{100,150,200,300,1000}/a1.npz` (same token stream, five attention windows, T=400, 8 dirs, layers {7,14,21,27}), `w200/`, `w500/`, and free-run flip records `fr_*.npz`, `free_*.json`. Fits in log space (log a(t)) with OLS / bounded least squares; uncertainties are parameter SEs and/or 2000-replicate bootstrap over the 20 (or 8) directions; AIC/BIC with parameter count +1. n quoted per fit. Layers 0–2 excluded from curve fits (dh ≡ 0 there); layer 3 tail (t ≥ 290) masked.

---

## Bullets

1. **Every response curve is a hump, and only piecewise forms describe it.** Over 700 individual curves (r ≤ 0.3, 20 dirs, layers {3,7,14,21,27}, w100) the full-curve BIC winner is the piecewise-power-in-log-t model in 504/700 cases and the continuous two-exponential in 196/700; single exponential, power, exp+floor, and log-normal win **0/700** (median R² of the winner 0.91; fig `fig_fits_examples.png`). Single smooth functional forms are rejected outright.

2. **The decay is (nearly) exponential and strongly identified.** Decay-segment fits (t ∈ [t\*+5, 299], layer 27, r = 0.1): λ_decay = −0.0121 [−0.0126, −0.0115] per token (n = 217 points/curve, per-curve |λ/SE| median ≈ 50, R² = 0.91–0.95 across all radii). Exp + constant floor is BIC-better than pure exponential in **78%** of curves (ΔBIC < −2; floor τ = 46.5 [42.7, 57.8] tokens with a positive floor C): the decay *slows down* at late t. Decay-model winners: exp+floor 45%, power 34%, pure exp 21%.

3. **The rise is intermittent, not a clean exponential — honest identifiability caveat.** Rise-segment fits give λ_rise = +0.0129 [+0.0110, +0.0164] per token (L=27, r=0.1, median over 20 dirs), but R² ≈ 0.31 (L=27) and 0.02–0.11 (L=7/14), and the per-curve |λ/SE| median is only 3.2 (≈25% of curves not individually significant at 95%). The exponential rise is an **envelope description of a jump process** (plateaus + spikes near t ≈ 16, 55–90), not a law of single curves. Rise exponentials *are* identifiable at the population level, never per-curve.

4. **Finite-time rate λ(t) = ln(a(t)/a(1))/t (r=0.1, L=27).** The 95% bootstrap CI of the smoothed median is entirely positive for **t ∈ [42, 44] and [57, 98]** (the mean gives [57,99]); at t = 90 all 20 directions have λ_d > 0. Outside those windows λ(t) < 0: the all-time average is −0.0033. λ(t) is a two-phase transient: growth for ~60–100 tokens, contraction before and after (fig `fig_lambda.png`).

5. **Envelope model comparison on median λ(t):** two-phase (λ₁(1−1/t) → −λ₂ + C/t) wins BIC (R² = 0.31, λ₁ = 0.0097, λ₂ = 0.0141, consistent with the segment fits); A·exp(−t/τ) R² = 0.22; C/t − λ₂ R² = 0.13; constant R² = 0; **λ(t) = A − B log t is rejected** (R² ≈ 0.000, B ≈ 0). No smooth envelope captures more than ~1/3 of the variance — the median curve wiggles at ±0.005 on a 0.01 scale (fig `fig_lambda_env.png`).

6. **Aggregation (Jensen).** λ-of-the-mean-curve − mean-of-λ_d is +0.007 at t = 5, +0.003 at t = 30 (the mean is dominated by spiking directions), |gap| ≤ 0.001 elsewhere, and the two aggregations give the same positive window [57,98]. The physically meaningful object is the **mean (or median) of per-direction λ_d** — the ensemble-averaged log-stretch, i.e. the finite-time Lyapunov exponent; the log-of-mean is amplitude-dominated and biased high at spike times (concavity: E[ln X] ≤ ln E[X]). The median is the robust equivalent (per-direction spread: IQR of λ_d at t=70 is [+0.0006, +0.0073], entirely positive).

7. **Radius dependence: linear regime through r = 2.** Peak time t\* = 77 tokens at *every* radius; λ_rise = +0.0125 → +0.0146 and λ_decay = −0.0097 → −0.0127 across r ∈ [10⁻³, 2] with overlapping bootstrap CIs (table in `t5_radius_table.json`); λ(t) curves collapse across r (median spread across radii 0.0011 < CI width 0.0035, t ∈ [10,200]). The decay-time scaling τ ~ r^β has β = −0.028 [−0.040, −0.017] (bootstrap over dirs, r ≤ 0.3, R² = 0.55): *statistically* nonzero, *practically* flat (±15% over a 300× amplitude range) — no meaningful τ(r) scaling (fig `fig_radius.png`).

8. **Depth: gain A(1,l) ~ l^p with p ≈ 0.7, steepening with r.** Early gain A(1) = median a(t=1, l): p = +0.70 [0.54, 0.91] for r ∈ {0.01, 0.1, 0.3} (R² ≈ 0.78; p ≈ +1.02 [L3 excluded], R² ≈ 0.79), rising to +0.91 [0.74, 1.14] at r = 1 and +1.11 [0.88, 1.26] at r = 2 — the same power law does **not** hold at all radii; p drifts upward in the nonlinear regime. The rise *rate* also grows with depth as a clean power law: λ_rise(l) ~ l^p with p = +1.59 [1.10, 3.79] (0.0016 → 0.0129 from L=7 to L=27, r = 0.1, R² = 0.998). In window_fixed the t=1 gain exponent is p = 1.20 at *every* W (identical — window inactive) and steepens by t = 150 (p ≈ 1.7–2.0); the rise-rate depth exponent is +4.8 at W = 100 and +1.9 at W = 150, while for W ≥ 200 the rise rates are negative so no positive power law exists (fig `fig_depth.png`).

9. **Attention window (controlled: same tokens, T=400): the growth phase is an attention-boundary effect.** The rise exists **only when the sliding window actually truncates**: λ_rise = +0.028…+0.032 [CI > 0] at W = 100, versus −0.0054…−0.0058 [CI < 0] for W ≥ 150 (identical early segments — the window is inactive until t = W − 65). The decay weakens monotonically with W: λ_decay(L=27) = −0.0078 (100) → −0.0051 (150) → −0.0042 (200) → −0.0004 (300) → **+0.0010 [0.0005, 0.0023] at W = 1000**. With full context the perturbation *contracts* early (λ_early = −0.0135 [−0.0175, −0.0116]) and mildly grows only late; with truncation (W=100) it grows from t ≈ 35–55 (truncation onset + O(20) tokens). On the independent w200/w500 token streams (stream-confounded) the rise is not significant (w200: −0.0019 [−0.0045, +0.0017]; w500: +0.0019 [−0.0030, +0.0279]), while the w100 stream gives +0.0130 [+0.0112, +0.0159]. **The "chaotic" growth phase of the w100 dataset is largely a window-truncation artifact; λ_early does not grow with W — it is W=100 that is special** (fig `fig_windows.png`).

10. **Free-run divergence of token streams: exponential-to-floor with a large floor.** f_ident(t) (fraction of directions whose token still matches the reference, n = 20): exp-to-floor wins BIC in 8/9 divergent cases (R² 0.71–0.85; power-to-floor statistically indistinguishable, ΔBIC ≈ 5–10; linear loses except one). Floors are large (C = 0.24–0.95): streams frequently re-align (repetitive text). Survival S(t) = P(no flip yet): Weibull with k < 1 (decreasing hazard) usually beats constant-hazard exponential; τ_div: r=2 → 232 [96, 690] tokens; r=1 → 695 [317, 3051]; childhood r=0.3 → 224 [127, 402]; r=0.5 → ≥ 1000 (**not identifiable within T = 300**) (fig `fig_divergence.png`).

11. **Window dependence of free-run flips (different token streams per window — stream-confounded).** Flip fraction by t=300 rises with W: r = 0.1: 0/20 (W=100) vs 3/20 (W=200, W=500); r = 0.3: 1/20 (W=100) vs 11/20 (W=200 and W=500; binomial CIs clearly separated). The free-run stream under larger windows passes closer to its own decision boundaries more often — the opposite sign from the controlled teacher-forced result (bullet 9), so the two effects must not be conflated.

12. **Spikes vs margins.** Individual early spikes occasionally coincide with near-zero-margin tokens (t = 16: margin percentile 0.02), but the systematic burst t ∈ [55, 90] (per-curve t\* median 77, IQR [62, 83]) sits at *typical* margins (median percentile 0.405) — the burst is truncation-driven (bullet 9), not decision-driven; only isolated spikes are margin events.

---

## Table of fits (all uncertainties 95% bootstrap CIs over directions unless marked SE)

| # | object | model | parameters ± CI / SE | R² | n | verdict |
|---|--------|-------|----------------------|----|---|---------|
| 1 | a(t), 700 curves (r≤0.3, w100) | pw power in log t | per-curve; t\* free | 0.91 (winner median) | 299 pts × 700 | wins 504/700 (BIC) |
| 2 | " | two-exponential (cont.) | per-curve; t\* free | 0.89 | " | wins 196/700 |
| 3 | " | exp / power / exp+floor / lognormal | — | 0.82 / 0.54 / 0.82 / 0.78 (medians) | " | **0/700 wins** |
| 4 | rise segment, L=27, r=0.1 | exp | λ = +0.0129 [+0.0110, +0.0164]/tok; per-curve SE ≈ λ/3.2 | 0.31 | 77 pts × 20 dirs | envelope only; intermittent |
| 5 | decay segment, L=27, r=0.1 | exp | λ = −0.0121 [−0.0126, −0.0115]/tok (τ = 83) | 0.91 | 217 pts × 20 | strongly identified |
| 6 | decay segment | exp + floor | τ = 46.5 [42.7, 57.8], floor C > 0; ΔBIC vs exp < −2 in 78% of curves | 0.95 | " | decay slows; best decay form (45%) |
| 7 | median λ(t), t∈[3,299] | two-phase (rise→decay) | λ₁ = 0.0097, λ₂ = −0.0141, t\* free (k=3) | 0.31 | 297 | BIC winner |
| 8 | " | A − B log t | B ≈ 0 | **0.000** | 297 | **rejected** |
| 9 | λ(t) sign window | bootstrap of smoothed median | λ > 0 for t ∈ [57, 98] (CI); frac dirs > 0: 1.00 at t=90 | — | 20 dirs × 2000 boots | positive growth window |
| 10 | Jensen | λ(mean) − mean λ_d | +0.007 (t=5), +0.003 (t=30), ≤0.001 else | — | 20 × 2000 | mean-of-λ = FTLE object |
| 11 | τ_exp(r), L=27, r≤0.3 | τ ~ r^β | β = −0.028 [−0.040, −0.017] | 0.55 (log-log) | 7 radii × 20 | flat for practical purposes |
| 12 | A(1,l), w100, r≤0.3 | A ~ l^p | p = +0.70 [+0.54, +0.91] (+1.02 excl. L3) | 0.78 | 5 layers × 20 | supported; r=1: +0.91, r=2: +1.11 |
| 12b | A(l), window_fixed (all W) | A ~ l^p | p(t=1) = +1.20 (all W, identical); p(t=150) = +1.69…+2.01 | 0.86–0.99 | 4 layers × 8 | same law every window at t=1 |
| 13 | λ_rise(l), r=0.1 | λ ~ l^p | p = +1.59 [+1.10, +3.79] (0.0016 → 0.0129, L=7→27) | 0.998 | 4 layers × 20 | depth-steepening of growth |
| 14 | window_fixed λ_rise, L=27 | — | W=100: +0.032 [+0.018, +0.037]; W≥150: −0.0055 [CI<0] | — | 8 dirs × 3 r × 2000 | growth only under truncation |
| 15 | window_fixed λ_decay, L=27 | trend vs W | −0.0078 → −0.0051 → −0.0042 → −0.0004 → +0.0010 (W=100…1000) | monotone | 8 dirs × 3 r | window controls contraction |
| 16 | f_ident(t), 9 divergent cases | C + A e^{−t/τ} | τ = 13–679 by case; C = 0.24–0.95 | 0.71–0.85 | 20 dirs, 300 pts | wins 8/9; power ≈ tied |
| 17 | survival S(t) | Weibull | k = 0.68 [0.30, 0.96] (r=2), k = 1.25 [0.96, 1.44] (childhood 0.3); τ_exp(r=2) = 232 [96, 690] | 0.81–0.85 | 20 × 1000 boots | decreasing hazard |
| 18 | window flips | binomial | r=0.3: 1/20 (W=100) vs 11/20 (W=200/500) | — | 20/window | stream-confounded |

---

## Which functional forms are supported, which are not

**Supported.**
- Two-phase (rise-then-decay) descriptions of a(t); piecewise-power-in-log-t preferred by BIC, continuous two-exponential close behind; the segment decomposition gives the meaningful rates.
- Exponential decay with a positive floor (better than pure exponential in 78% of curves); pure exponential is adequate at shallow layers (L=7 wins more often).
- A positive finite-time stretching window t ∈ [57, 98] at L=27 (r = 0.1, w100), with the whole direction population positive at t = 90; λ_rise ≈ +0.013/token, λ_decay ≈ −0.012/token, t\* ≈ 77.
- Amplitude-independence (linear regime) through r = 2: rates, t\* and λ(t) shape do not depend on r; τ(r) effectively flat (β = −0.03).
- Depth gain A(1,l) ~ l^p, p ≈ 0.7 (r ≤ 0.3), steepening with r; rise rate grows with depth (~l^{+1.6}, R² = 0.998).
- exp-to-floor divergence of free-run token agreement with a large re-alignment floor; Weibull (k<1) survival for flip times.
- The growth phase is produced by attention-window truncation (controlled same-token comparison), weakening with W and absent for W ≥ 150.

**Not supported.**
- Single exponential, single power law, exp+floor, or log-normal as *full-curve* forms (0/700 wins).
- A clean per-curve exponential rise (low R², weak per-curve identifiability; it is an intermittent jump process).
- λ(t) = A − B log t (R² ≈ 0) and any smooth envelope capturing the median λ(t) beyond R² ≈ 0.3.
- τ ~ r^β with appreciable β (|β| ≲ 0.04).
- Constant-hazard (pure exponential) survival for weakly-diverging free runs (k < 1) — and τ_div is not identifiable within T = 300 whenever divergence is rare (r = 0.5 case).
- Identifying any exponential law in the *free-run* window-agreement curves without a floor: the floor (re-alignment) is essential and large.

**Honest caveats.** (i) Exponential rise rates are identifiable as population statistics, not per-curve. (ii) The window comparison in `window_fixed` is controlled (same tokens) but uses 8 directions — CIs are correspondingly wider. (iii) The w200/w500 free-run comparison uses different token streams per window; it cannot isolate the attention effect. (iv) Layer 0–2 carry no signal (dh ≡ 0), so depth laws start at L = 3 (or 7 if L3 is excluded).

---

## Figures
- `fig_fits_examples.png` — 3 example full-curve fits + BIC winner counts (Task 1).
- `fig_lambda.png` — per-direction λ_d(t) + CI, radii collapse, depth structure, Jensen (Tasks 2–3).
- `fig_lambda_env.png` — envelope model comparison + decay-segment forms (Task 2).
- `fig_windows.png` — λ_early / λ_rise / λ_decay vs W, same tokens (Task 4).
- `fig_radius.png` — λ_rise, λ_decay, τ vs r with CIs (Task 5).
- `fig_depth.png` — A(1,l) fits, p with CIs, λ_rise(l) per window (Task 6).
- `fig_divergence.png` — f_ident(t) fits, survival curves, window flip fractions (Task 7).

## Raw outputs
`t1_summary.json`, `t1_per_curve.npy`, `t2_summary.json`, `t2_results.npz`, `t456_summary.json`, `t5_radius_table.json`, `t7_summary.json`, `t7_boot.json`, `depth_quick.json`, `lam_iqr.json`, `tpeak_margin.json` (this directory).

---

## Summary for the main agent

We fitted every perturbation-growth curve we have (700 teacher-forced response curves, 5 attention windows, 12 radii, 5 layers, plus 4 free-run datasets) with six functional forms and model comparison, and the picture is consistent: the response a(t) is a two-phase transient — an intermittent, spike-dominated rise (t\* ≈ 77 tokens, envelope rate λ_rise = +0.013 [0.011, 0.016]/token, only ~31% of per-curve log-variance explained; population-level only) followed by a strongly identified near-exponential decay (λ_decay = −0.012 [−0.0126, −0.0115]/token, exp+floor preferred in 78% of curves) — and only piecewise models ever win (504/700 piecewise-power, 196/700 two-exponential; exp/power/lognormal/exp+floor never). The finite-time rate λ(t) is positive with 95% bootstrap confidence only for t ∈ [57, 98] (all 20 directions positive at t = 90) and negative elsewhere; λ(t) = A − B log t is rejected, and the correct envelope is the two-phase form. Aggregation matters little here (Jensen gap ≤ 0.003), but the FTLE object is the mean/median of per-direction λ_d, not the λ of the mean curve. Rates are amplitude-independent through r = 2 (t\* and λ flat; τ ~ r^−0.03), and the gain grows with depth as A(l) ~ l^0.7 (steepening with r) with rise rate ~l^{+1.6} (R² = 0.998). The most important structural finding is that the *growth phase itself is an attention-boundary artifact*: in the controlled same-token comparison the rise exists only at W = 100 where the sliding window truncates (λ_rise = +0.032 vs −0.0055 for W ≥ 150), the decay weakens monotonically with W and flips sign at W = 1000, and the full-context model contracts early (−0.0135) — so any "chaotic Lyapunov exponent" quoted from the w100 runs should be labeled as a truncation-driven finite-time rate, while free-run token divergence instead *increases* with window (1/20 → 11/20 flips at r = 0.3, stream-confounded) and follows an exponential-to-floor law with a large re-alignment floor (τ_div = 232 [96, 690] tokens at r = 2, Weibull k < 1).
