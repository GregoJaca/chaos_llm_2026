# Current-state perturbation and the two-time response R(t,s)

**Scope.** Local CPU-only re-analysis of the existing current-state perturbation
data (no new model runs). Input-embedding injection (`results/local_data/state.npz`,
s=16/50/100/150, 32 Gaussian directions, seeds 1000+i, r=1e-3, measured at layer 27,
teacher-forced, 300 tokens) and layer-14 residual injection
(`results/local_data/state_mid.npz`, s=16/50/100, 8 directions, seeds 2000+i).
Scripts in `scripts/`, figures in `figures/`, machine-readable numbers in `stats/`.
Reproduce with `python scripts/run_all.py` (needs only numpy/scipy/matplotlib).

---

## 0. Data audit (do this before interpreting anything)

Three findings that change how the data should be read:

1. **`results/local_data/a1.npz` is NOT the baseline of the state runs.** Its md5
   matches `results/a1_ds_w200_large/a1.npz` bit-for-bit. The state runs used
   `results/a1_ds_w100/a1.npz` (per `state_w100_32dirs/meta.json`), whose margins
   match `margins/w100.npz` to float32 noise (corr 0.99999999998, min margin
   0.0202 at t=123). Any margin/near-tie interpretation must use the w100
   trajectory; using `local_data/a1.npz` (w200: min margin 0.0031 at t=44, a
   different near-tie spectrum) would be wrong.
2. **The teacher-forced forwards were effectively full-attention.** Under a hard
   100-token sliding window, dh(t≥s+100) would be *exactly* 0. Instead, median dh
   at lags 100–120 is 0.00096–0.016 and the max is 0.041 (`stats/audit.json`,
   `window_check`). So the "w100" label describes how the *token sequence was
   generated*, not the attention span during the perturbation measurement.
   Consequence: the decay of R(t,s) below is **not** a window-cutoff artifact —
   but it is also not the response of the windowed system.
3. **r=1e-3 is deep in the linear regime at the logit level.** Max |dm| over all
   s, directions, tokens is 3.7e-3 = 18% of the *smallest* baseline margin
   (0.0202); median |dm| is 1.5e-5. No top-1 flips occur anywhere. Everything
   reported below is infinitesimal perturbation dynamics; no threshold crossing
   is involved.

Also verified: dh(t<s) = 0 exactly (clean causality), and median response
floors (7.6e-4 … 3.6e-3 for emb; ~2.9e-4 for mid) sit above fp32 noise
(dh(t<s)≡0 rules out nondeterministic noise), so no floor contamination.

## 1. Direction groups: mostly similar, no groups, no special directions

Per s, correlations of log-response curves across the 32 directions
(`stats/direction_groups.json`, `figures/fig01_dir_corr_heatmaps.png`):

| s | median r | p10 | frac > 0.9 | min | PC1 share | resid std (dex) |
|---|---|---|---|---|---|---|
| 16 | 0.975 | 0.953 | 1.00 | 0.913 | 0.974 | 0.089 |
| 50 | 0.942 | 0.896 | 0.88 | 0.792 | 0.937 | 0.075 |
| 100 | 0.942 | 0.905 | 0.92 | 0.826 | 0.941 | 0.062 |
| 150 | 0.958 | 0.903 | 0.90 | 0.825 | 0.953 | 0.077 |

Refinements beyond the earlier "median 0.94–0.98, 88–100% > 0.9" summary:

- **Part of the similarity is the shared initial transient.** Excluding lags 0–2,
  the median is still 0.914–0.973 but the fraction >0.9 drops to 0.66 at s=100
  (0.84–1.00 elsewhere). So the "directions look alike" statement is strongest
  at early lags and weakest, though still true in the median, in the tail.
- **Magnitude of direction effects:** after removing the per-s median curve,
  residual std is 0.06–0.09 dex, i.e. direction idiosyncrasies are a **×1.15–1.23**
  modulation on top of a shared response (p90 |residual| 0.10–0.15 dex).
- **No groups:** hierarchical clustering gives 1 cluster at the r>0.8 level for
  every s; at r>0.9 there are 1–2 groups, i.e. no meaningful partition.
- **No special directions:** no direction has median correlation to the others
  below 0.85 at any s. The worst pairs differ by s ((4,16), (0,5), (4,5), (7,20)).
- **No persistent direction identity:** correlating each direction's deviation
  from the median curve across different s gives median per-direction
  correlations of −0.13 … +0.07. A direction that deviates at one s does *not*
  deviate the same way at another s. The idiosyncrasies are s-specific noise,
  not a direction property.

**Bottom line (user's suspicion, quantified):** directions are essentially
equivalent up to a ×1.2 log-curve modulation; one shared shape (PC1) explains
94–97% of the log-curve variance; no clusters, no outliers, no direction you
could single out and study again. Averaging over directions is safe; hunting
for "special directions" at r=1e-3 is not supported by this data.

## 2. Time-shift / collapse: R(t,s) is NOT a function of lag

All overlap attempts, quantified as the spread factor across the four s curves
(std over s of log10 median R at fixed lag; 1.0 = perfect collapse), on the
common lag range τ∈[5,149] (`stats/shift_collapse.json`,
`figures/fig02_collapse.png`):

| normalization | RMS spread factor | max spread factor |
|---|---|---|
| R = dh(t)/dh(s) | 1.63 | 2.54 |
| R = dh(t)/peak(lag≤3) | 1.56 | 2.41 |
| R = dh(t)/dh(s+5) | 1.59 | 2.53 |

- Best normalization (by the early peak) still leaves a **×1.56** RMS
  disagreement. Over the first 50 lags the spread is already ×1.5–1.8.
  Normalizing the lag axis by s does not help. Aligning each curve at its first
  prominent bump (t = 37, 137, 117, 153 for s = 16, 50, 100, 150) does not
  collapse the rest of the curve. **Every variant fails**; this is a robust
  negative, not a plotting problem.
- **Why it fails:** the four curves differ in kind, not just scale. s=16 starts
  from a ×10 larger injection response (0.124) and is bump-rich (13 bumps
  >0.15 dex); s=50 and s=100 are smooth decays; s=150 is a slow bumpy plateau.
  Time-translation invariance does not hold — expected, since the system is
  non-autonomous (token-indexed, state-dependent gains).
- **Bumps vs near-ties (honest negative):** bumps do **not** sit on the margin
  near-ties (t = 2, 14, 16, 40, 48, 69, 116, 123, 135, 143, 147 for margin<0.2).
  The ratio of dh at near-tie tokens to the local 9-token median is 0.97–1.0.
  A more sensitive lagged correlation of the bump residual against
  −log margin gives best values of only r = 0.35 (s=16, margin leads by 1),
  0.44 (s=150, dh leads margin dips by ~3), 0.16 (s=50) and 0.07 (s=100) —
  real but weak, and with s-dependent sign of phase. The bump structure is
  *loosely* coupled to low-margin regions, not pinned to them.
- **Separability (partial yes):** an additive model log dh(t,s) ≈ a(s) + b(t)
  fits the common grid t≥150 with residual std 0.066 dex (**×1.17**), but on
  wider grids the pairwise residual is ×1.16–1.51. The response field is
  *approximately* separable — "gain set at injection × slow decay in t" is a
  decent first approximation late in the sequence, with s-specific bumps as
  the non-separable remainder (`figures/fig02_separability.png`).

## 3. s-dependence: non-monotonic gain, two-timescale decay, no clean regime change

(`stats/s_dependence.json`, `figures/fig03_s_dependence.png`)

| s | dh(s) median | gain dh(s)/r | margin(s) | envelope slope, lags 10–50 (dex/tok) | R² | half-life (tokens) | bumps >0.15 dex |
|---|---|---|---|---|---|---|---|
| 16 | 0.1237 | 124 | 0.120 | +0.0013 (no fit) | 0.01 | 2 | 13 |
| 50 | 0.0096 | 10 | 7.17 | −0.0143 | 0.89 | 8 | 1 |
| 100 | 0.0126 | 13 | 0.81 | −0.0035 | 0.41 | 2 | 1 |
| 150 | 0.0321 | 32 | 0.87 | −0.0083 | 0.97 | 6 | 2 |

- **Immediate gain varies ×13 across s (10–124) and is not explained by the
  local margin alone:** s=100 and s=150 have nearly equal margins (0.81 vs
  0.87) but gains 13 vs 32. s=16 is a near-tie token (margin 0.12) with a ×10
  gain — the only case where the margin clearly matters.
- **Decay is two-timescale:** a fast initial drop (half-life 2–8 tokens),
  then a slow tail (e-folding 30–125 tokens on lags 10–50; flatter still on
  50–120, where s=50 even rises because of the bump cluster at t≈132–148).
- **Regime change?** With n_s = 4 this cannot be decided. The one qualitative
  outlier is s=16 (near-tie injection: large gain, bump-rich, no clean decay
  fit). s=50/100/150 are qualitatively alike: modest gain, fast drop, slow
  bumpy tail. Do not call this a regime change; call it "s=16 is special
  because it was injected at a near-tie."

## 4. Input-embedding vs layer-14 residual injection

(`stats/mid_vs_emb.json`, `figures/fig04_mid_vs_emb.png`; note: different
direction seeds, so this is an unpaired comparison of medians with IQR bands)

| s | dh(s) emb | dh(s) mid | ratio | tail t≥150 emb | tail t≥150 mid | tail ratio | slope (lags 5–30) emb | mid |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.1237 | 0.00341 | ×36 | 6.2e-3 | 2.93e-4 | ×21 | +0.023 (rising, bumps) | +0.019 |
| 50 | 0.0096 | 0.00102 | ×9.4 | 2.1e-3 | 2.83e-4 | ×7.5 | −0.026 | −0.003 |
| 100 | 0.0126 | 0.00107 | ×11.7 | 9.7e-4 | 2.86e-4 | ×3.4 | −0.017 | −0.004 |

- The mid response is **9–36× weaker at injection** (not a single "×10"),
  and the ratio shrinks with lag: by t≥150 it is only ×3.4 for s=100.
- **The earlier note "mid decays faster" is only true for the first ~5 tokens.**
  Beyond that, the mid envelope is *flatter* than emb (slopes −0.003/−0.004 vs
  −0.026/−0.017 dex/token): the layer-14 injection leaves a persistent,
  roughly constant footprint at ~2.9e-4.
- That plateau is a real signal, not a numerical floor: it is structured
  (lag-1 autocorrelation 0.54), identical across s ∈ {16,50,100} (2.83–2.93e-4)
  and across all 8 directions, and dh(t<s) is exactly 0, ruling out fp32 noise.
  It is consistent with later tokens attending the perturbed keys/values of
  position s at layers 15–27, with a gain that is nearly independent of the
  injected direction and of s.
- Direction scatter is smaller for mid (0.02–0.04 dex) than emb (0.07–0.12 dex):
  fewer layers to amplify direction-specific structure → more isotropic response.
- Scale for intuition: max dh/||h|| is 3.8e-4 (emb) and 8.0e-6 (mid) relative to
  the layer-27 hidden norm (median 545). At r=1e-3 the "flux tube" is very thin.

## 5. Larger r: not measured for state injection (limitation + proxy)

There is **no current-state injection data at any r ≠ 1e-3** locally; nothing
about nonlinear onset, threshold crossing, or r-scaling can be *measured* for
this protocol from the existing data. What exists on the *same w100 trajectory*
is a prompt-perturbation teacher-forced r-scan (`local_data/w100_layers/a1.npz`,
12 radii 1e-3→2.0), used here strictly as a proxy
(`stats/r_scaling.json`, `figures/fig06_r_scaling.png`): the layer-27 response
is linear in r (α = 0.98–1.04) over r ∈ [3e-3, 0.2], mildly superlinear above
(α ≈ 1.1–1.26 for r 0.3–0.7), with |α−1|>0.1 onset at r≈0.3; at the bottom pair
(1e-3→3e-3) α = 0.91 with a wide IQR (0.76–0.95), i.e. the smallest responses
are already brushing the numerical floor at r=1e-3. **Inference (not
measurement):** the state response at r=1e-3 is at the low edge of a clean
linear window with roughly two decades of headroom (3e-3…0.2), but a state
r-scan at {3e-3, 1e-2, 3e-2} (a few GPU hours) is needed before making any
r-dependence claim about R(t,s) itself.

## 6. Best visualizations

- **`figures/fig05_response_field.png`** — the figure to look at first: heat map
  of log10 median dh over the (s,t) plane (white = before injection; the
  causal wedge), a zoomed row (s=100) with 3 individual directions + median,
  and the baseline margin with near-ties marked for context. One glance shows:
  causality, ×13 gain variation, fast-drop/slow-tail structure, bumps, and that
  bumps do not line up with margin near-ties.
- `figures/fig05_lag_view.png` — the same field in lag coordinates plus the
  spread-factor-vs-lag curve (mean ×1.58): the cleanest statement of "R is not
  a function of lag".
- `figures/fig01_dir_curves.png`, `fig01_dir_corr_heatmaps.png`,
  `fig01_pc1.png` — direction equivalence (3 individual + aggregate, sorted
  correlation heat maps, PC1 shares).
- `figures/fig02_collapse.png` — all six overlap attempts with their failure
  factors printed on the panels; `fig02_separability.png` — how close the field
  is to a(s)+b(t).
- `figures/fig03_s_dependence.png`, `figures/fig04_mid_vs_emb.png`,
  `figures/fig06_r_scaling.png`.

## 7. Negative results (do not retry these as stated)

1. Lag collapse of R(t,s) under any amplitude or lag normalization — fails at
   ×1.5–1.6 RMS; the curves differ in kind.
2. Direction clustering / special directions at r=1e-3 — no support (no groups,
   no outliers, no cross-s persistence).
3. Bumps as margin near-ties — null as a strict alignment law (enrichment ≈ 1);
   only weak, phase-inconsistent tracking (r ≤ 0.44).
4. "Layer-14 injection decays faster" — only the first ~5 tokens; its tail is
   flatter than the input-embedding response.
5. Regime change in s — not decidable with 4 injection times; s=16's oddity is
   explained by near-tie injection, not by a regime boundary.

## 8. Limitations

One model, one prompt, one trajectory, teacher-forced (the perturbation never
feeds back into generated tokens), 32+8 isotropic Gaussian directions in the
1536-dim space, measurement only at layer 27 (mid: injection at layer 14),
r=1e-3 only, s-grid of 4 points, and — per the audit — the measurement forwards
were effectively full-attention, so these numbers describe the full-attention
teacher-forced kernel, not the windowed one. The mid comparison is unpaired
(different seeds, n=8 vs 32). The r-scaling proxy uses the prompt protocol.

---

## Summary for the main agent

We re-analyzed the existing state-perturbation data (no new runs) and can now
answer the user's questions quantitatively. (1) Directions do NOT form groups
and none is special: PC1 explains 94–97% of log-curve variance, pairwise
correlations have median 0.94–0.98 with worst pairs still ≈0.72–0.91, residual
direction effects are only ×1.15–1.23 modulations, no direction keeps its
idiosyncrasy across different s — so averaging over directions is safe and
"special direction" hunting at r=1e-3 is unsupported. (2) R(t,s) is not a
function of lag: after amplitude normalization by dh(s), early peak, or
dh(s+5), the four s-curves still disagree by a factor ≈1.6 RMS (max ≈2.5), and
lag/s rescaling or first-bump alignment does not fix it; the response field is
only approximately separable (a(s)+b(t) residual ×1.17 on the common late grid).
The failure is real dynamics (non-autonomous, state-dependent gains), not
plotting. (3) The gain at injection varies ×13 across s (10–124, largest at
the near-tie token s=16) and is not explained by the local margin; decay is
two-timescale (half-life 2–8 tokens, then e-folding 30–125 tokens) with bumps
only loosely coupled to margin dips (lagged r ≤ 0.44, phase varies with s).
(4) Layer-14 residual injection is 9–36× weaker at injection than
input-embedding injection but does NOT decay faster after the first ~5 tokens —
it plateaus at a persistent, direction- and s-independent footprint ≈2.9e-4
(real signal, not fp32 noise). (5) Larger r is unmeasured for this protocol;
a prompt-perturbation proxy on the same trajectory says the response is linear
for r∈[3e-3,0.2] with nonlinearity onset ≈0.3, so r=1e-3 is at the low edge of
a clean window and a small state r-scan would settle it. Two audit flags matter
for interpretation: `results/local_data/a1.npz` is the w200 baseline, NOT the
state runs' baseline (use `results/a1_ds_w100/a1.npz`), and the teacher-forced
measurement forwards were effectively full-attention (dh persists past lag 99;
"w100" only describes how the tokens were generated). Best figure:
`figures/fig05_response_field.png`; all numbers in `stats/*.json`.
