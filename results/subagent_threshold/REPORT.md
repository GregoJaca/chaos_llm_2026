# Threshold / flux-tube analysis — conclusive figures

**Scope.** Local CPU analysis of the DeepSeek-1.5B perturbation data in
`results/local_data/`: `w100_layers/a1.npz`, `w200/a1.npz`, `w500/a1.npz`
(sustained kicks in a 20-direction token-simplex subspace, 20 dirs x 300 steps x
radii up to 12 values), free-run flip records (`free_w100_large.json`,
`free_w200.json`, `free_w500.json`, `free_large.json`), and baseline margin traces
(`margins/w{50..4096}.npz`, T=600). All scripts + figures live in this folder.
Data model used throughout: `r_c(d,t) = r·m_t/|δm_r,d,t|` (verified exactly against
the stored arrays), `pred_T` = first t with `r_c ≤ r` (the *adversarial* rule).

## Robust findings

1. **The margin–gain plane is the right picture, and it is readable**
   (`fig1_margin_gain_scatter.png`). Every observed flip sits above the iso-`r_c`
   line `g_t = m_t/r` at a deep margin valley; non-flipping directions sit below
   their weakest tie. No direction-index axis needed: axes are physical
   (margin vs gain), iso-lines carry the radius, and markers carry the outcome.
2. **A single near-tie controls almost every direction** (`fig3_control_map.png`).
   For window 200/500 the tie at t=44 (m = 0.0031, token 13 vs 311) controls
   13–14/20 directions; secondary ties at t=55, 183/165. For window 100 the tie
   at t=123 (m = 0.0202) controls 12/20, with t=143 (4/20) and t=16 (3/20).
   Rows sorted by threshold, not by direction index.
3. **The sign of δm decides who flips, not just |δm|** (biggest refinement).
   A flip needs δm_t ≤ −m_t. At w200's t=44 tie, 18/20 directions are
   *destabilizing*; at w100's t=123 tie only 7/20 are. So w200's tie is fragile
   both because its margin is 6.5× smaller and because the whole kick subspace
   pushes downhill through it. The *signed* threshold `r_c^- = r·m/max(−δm,0)`
   fixes flip-step attribution: e.g. w200 d10 is predicted at t=44 by the |δm|
   rule but flips at t=55, exactly where `r_c^-`(55) = 0.045 first crosses r=0.1.
4. **Thresholds are heavy-tailed and heterogeneous** (`fig2_threshold_ecdf.png`).
   Median per-direction signed threshold r_*⁻: ≈1.7 (w100) vs ≈0.26 (w200, w500);
   12/20 w200/w500 directions are below 0.3 (vs 3/20 for w100). Pooled over
   (d,t), >95% of local thresholds exceed 1 — the danger is concentrated in
   <1% of steps (frac r_c⁻ ≤ 0.3: 0.1% w100, 0.7% w200).
5. **Prediction vs observation: sharp where it should be, fuzzy in a ±30%
   borderline band** (`fig5_pred_vs_obs.png`). Exact flip-step agreement:
   20/20 and 19/20 for r ≤ 0.1; 17–18/20 at r=0.3; 15/20 at r=0.5; 9–12/20 for
   r = 1–2. All r=0.3 errors have r_*⁻ in [0.26, 0.40] (flippers up to 0.396,
   non-flippers down to 0.258) — a soft threshold zone, not a wrong model.
6. **The attention window reshapes the margin spectrum before it changes the
   trajectory** (`fig4_window_spectrum.png`). Truncation only departs from the
   full-context run at t = 36 (w100), 136 (w200), 436 (w500); w1000 ≡ w4096.
   Yet the margin floor collapses ~50×: min m = 0.167 (w50) → 0.020@123 (w100) →
   0.0031@44 (w200+), and near-ties (m<0.05 per 600 steps) grow 0 → 2 → 8–12.
   w200 and w500 share the trajectory up to t≈136, so their observed flips are
   literally identical (d1@44, d10@55, d11@44 at r=0.1).
7. **Small-r thresholds do NOT extrapolate to large r** (`fig6_large_r_linearity.png`,
   the key failure quantification). Gain |δm|/r is flat (±10% at 25–75%) for
   r ≤ 0.1, then grows superlinearly: median gain 0.0085 → 0.0132 (1.5×) at r=2.
   The fraction of (d,t) pairs whose r_c(r) drifts outside [0.5, 2]×r_c(0.01)
   grows 11% (r=0.1) → 16% (0.5) → 20% (2), with 5% tails reaching 20× either way.
   Consequently the linear |δm| rule over-predicts flips at large r: predicted
   15/20 vs observed 8/20 at r=1, and 19/20 vs 16/20 at r=2. Below r≈0.01,
   responses drown in float32 noise (hidden norms ~500 ⇒ δm ~ 1e-5 is at
   machine epsilon).
8. **Observed flip fractions (free runs, 20 directions each)**: w200/w500:
   15% at r=0.1, 55% at r=0.3. w100: 0% at 0.03/0.1, 5% at 0.3, 15% at 0.5,
   40% at r=1, 80% at r=2. No text was generated locally, so flip-fraction ↔
   text-quality correspondence is not assessed here.
9. **Per-token attribution is ill-posed** — the 20 simplex directions are rank-19
   (cond ≈ 1e8), so δm cannot be decomposed into per-token sensitivities;
   "which token controls the tie" is not identifiable from these directions.
   The meaningful geometry is the sign structure + threshold ranking instead.
10. **Caveat**: `free_w100_large.json` comes from a twin run of the w100_layers
    npz (both have margin min ≈ 0.0201–0.0202 at t=123); its obs/pred pair is
    internally consistent, and the overlay on the w100_layers geometry is safe
    to ~1e-4 in margin.

## Figures (all in this folder)

| file | shows | verdict |
|---|---|---|
| `fig1_margin_gain_scatter.png` | m_t vs gain, iso-r_c lines, observed flips marked | **best mechanism figure** |
| `fig2_threshold_ecdf.png` | r_*⁻ ECDF per window + pooled threshold field | distribution story |
| `fig3_control_map.png` | r_c⁻ heatmaps, rows sorted by danger, controlling ties marked | **which near-tie controls — best map** |
| `fig4_window_spectrum.png` | margin traces per window + floor/near-tie summaries | window effect |
| `fig5_pred_vs_obs.png` | annotated obs\|pred class table + diagonal plot | **prediction scorecard** |
| `fig6_large_r_linearity.png` | flip fractions vs r, r_c(r)/r_c(0.01) bands, gain growth | linearity failure quantified |
| `fig7_cases.png` | 3 individual directions: d10 (fragile), d1 (typical), d13 (resistant) | per-case intuition |

Supporting scripts: `common.py` (data model), `s0_audit.py` → `audit.txt`,
`s1_signed.py` → `signed_check.txt` (signed vs |δm| predictor tables),
`s2_qa.py` (figure QA + cross-checks), `f1..f7_*.py` (one per figure).

## Recommended presentation order

1. `fig7` (three threshold tubes) → "a kick flips at the first step where the
   tube dips below r" — instant intuition.
2. `fig3` → "the dips are shared: one tie at t=44 (w≥200) or t=123 (w100)
   controls most directions; sign of δm splits fragile from safe."
3. `fig1` → the geometric rule in one glance.
4. `fig5` → scorecard: 95–100% within the linear regime, soft band at
   r ≈ r_*, breakdown r ≥ 1.
5. `fig4` → windows move the margin floor 50× while trajectories stay close.
6. `fig6` → close with the honest limit: linear thresholds valid for
   r ≲ 0.1–0.3; superlinear gain (1.5× median by r=2) breaks extrapolation
   bidirectionally.

**Avoid in future plots**: direction index as an x-axis (meaningless — use
threshold-sorted rows), bar charts for 20-direction comparisons (use annotated
tables/heatmaps), and any prediction claimed for r ≥ 1 without the nonlinearity
caveat.

## Summary for the main agent

The threshold/flux-tube story is now conclusive and each claim has a figure: a
sustained kick of radius r flips the argmax at the first step t where the signed
local threshold r_c⁻(d,t) = m_t·r/max(−δm_t,0) falls below r; for windows ≥200
that is almost always the shared t=44 near-tie (m=0.0031, 13–14/20 directions,
18/20 destabilizing), for window 100 the t=123 tie (m=0.020, only 7/20
destabilizing), so the window controls fragility by collapsing the margin floor
~50× (0.167→0.0031) long before truncated and full-context trajectories diverge
(t=36/136/436). Predicted first flips match free-run observations at 19–20/20
for r ≤ 0.1, 17–18/20 at 0.3 (all misses in a ±30% borderline band r_*⁻ ∈
[0.26, 0.40]), and fail at r ≥ 1 (9–12/20) because gain grows superlinearly
(median |δm|/r up 1.5× by r=2, 16–20% of thresholds drift outside [0.5, 2]×,
over-predicting flips: 15 predicted vs 8 observed at r=1); the |δm|-based
r_c in the shipped data is the adversarial version and the signed version is
the physically correct one (it fixes flip-step attribution, e.g. w200 d10:
t=55 not t=44). Best presentation: fig7 (three threshold tubes) → fig3
(control map) → fig1 (margin–gain scatter) → fig5 (obs|pred table) → fig4
(window spectrum) → fig6 (linearity failure); drop direction-index axes and
per-direction bars entirely, and don't extrapolate small-r thresholds beyond
r ≈ 0.3 without the superlinearity caveat.
