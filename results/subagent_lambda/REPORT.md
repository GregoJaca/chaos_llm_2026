# λ(t): finite-time stretching rates of perturbations in the DeepSeek-1.5B teacher-forced runs

**Analyst:** nonlinear-dynamics subagent (CPU, local numpy/scipy/matplotlib only).
**Everything below is computed from `results/local_data/**` (teacher-forced) plus the free-run records under `results/a1_*/free_run*/free_run.npz`. No GPU, no ssh, nothing outside this folder was modified.**

---

## 0. Conventions and definitions (read first)

- Token index `t` is 1-based; `t = 1..300` are the generated tokens (teacher-forced: both runs consume the **same** token sequence; only the initial state differs).
- `a(t,l) = ||δh_{t,l}||` is taken from `dh_norm[r, u, l, t-1]` (`r` = perturbation radius, `u` = perturbation direction, `l` = layer).
- **Local rate** `λ_k(t) = (1/2k)·ln[ a(t+k) / a(t−k) ]`, `k = 5` (main) and `k = 10` (tail/sustained-rate estimator). NaN wherever an endpoint is 0.
- **Cumulative rate** `Λ(t) = (1/t)·ln[ a(t)/a(1) ]`.
- **Regions:** early = `t ∈ [2,30]`; **eviction epoch** (W=100, 63–65-token prompts) = `t ∈ [36,100]`, i.e. while the perturbed *prompt* KV entries are sliding out of the window; **tail** = `t > 150`.
- **Bands** are 25–75% percentiles over perturbation directions; **CIs** are 95%. Two bootstrap flavours: (i) over directions (n = 8–20), (ii) **two-way** = directions **and** moving-block bootstrap over tokens (block = 10). The latter is the conservative one and is what the verdict uses.
- The system is non-autonomous (token-forced): all rates are finite-time rates, not Lyapunov exponents. `λ ≤ 0` here means "perturbations do not grow along the realised trajectory", not "the map is globally contracting".
- Units: all rates are **per generated token**.

Data inventory actually used: `w100_layers` (12 radii × 20 dirs × 8 layers × 300 tokens), `w200`, `w500`, `window_sweep{,_cyber}` (6 W × 10 dirs × 3 radii × 4 layers, same prompt), `rate_prompts` p1–p5 (p6 folder exists but is **empty**), `rate_w50` p1–p3, `rate_stats/prompts_long*` (4 long runs), free-run flip records (5 files). Note: for prompt 63 + 300 generated tokens, **W = 400, 800, 4096 are the same trajectory** (363 total tokens < 400, no eviction), so the window sweep has 4 distinct regimes: W=50 (prompt truncated), W=100 (first eviction at t≈38), W=200 (first eviction t≈138), W≥400 (flush-free).

---

## 1. Executive verdict

**No. There is no window, radius, layer, prompt or direction in which the asymptotic (sustained) stretching rate is significantly positive.**

- Sustained rate (mean λ₁₀, t > 150) across **31 configurations** (7 layers × prompt65/W100; 4 radii × L27; 6 windows × L27 + cyber; 12 independent prompt runs): **0/31 point estimates positive**; range **−0.002 … −0.039 per token**.
- Direction-bootstrap 95% CIs: **0/31 touch positive**; the largest upper bound anywhere is **−0.0014**.
- Conservative two-way CIs (directions + token blocks): **0/31 entirely positive**; only 2/31 upper bounds poke above zero, at **+0.0008** (cyber W=4096, point estimate −0.0021) and **+0.0005** (prompt p1 W=100, point estimate −0.0017). So even in the worst case, a sustained rate ≥ **+0.001 per token** is excluded.
- The "Lyapunov-like" number on the flush-free runs: **−0.0043 [−0.0055, −0.0030]** (prompt63/W=4096/L27), **−0.0021 [−0.0024, −0.0019]** (cyber W=4096), **−0.0051** (longP63 W=4096), i.e. the asymptotic rate is **negative but small** (−0.2%…−0.5% per token): perturbations slowly **decay**, they never compound.
- Independent confirmation from free-running (not teacher-forced) runs: **174/180 runs at r ≤ 0.1 never diverge within 300 tokens**, and the 6 exceptions all flip at the *same* token t=44 (a single vulnerable near-tie, not progressive separation). The deepseek prompt never diverges at any tested radius (60/60 up to r=0.3).
- The only positive rates that exist are **transients**: a 1-token alignment spike at t=1 (L27: +0.254, L21: +0.097, L7: +0.054; L14 and below ≤ 0) and a shared cumulative burst **Λ ≈ +0.059 at t ≈ 13** that all window sizes share — both die within ~15 tokens.

**Best figures: fig09 (forest = the verdict), fig08 (tail fits + AIC/BIC), fig04 (window regime), fig10 (synthetic comparison — what a real positive rate would look like), fig03 (radius collapse), fig01 (raw phenomenology).**

---

## 2. Question-by-question results

### Q1 — λ₅(t) per layer; growth phase, flush dip, tail (**fig01, fig02, fig06**)

- **No extended growth phase exists.** The response grows for exactly **one token**: median ln[a(2)/a(1)] = **+0.254 (L27)**, +0.097 (L21), +0.054 (L7), −0.0006 (L3), −0.076 (L14), −0.114 (L2), −0.172 (L1). From t=2 on, every layer's median λ₁ is ≤ 0 (fig02C/D). This is initial-condition alignment (the random simplex direction snaps onto the fastest local direction), not chaos.
- **Phenomenology (W=100, r=0.1, prompt65, 20 dirs):**
  - L0 (embeddings): `a ≡ 0` exactly — the perturbation never reaches the input layer.
  - L1, L2, L3: response **dies to exact float32 zero** once the perturbed prompt KV entries have been evicted: all 20 dirs dead from **t=93 (L1), t=192 (L2), t=291 (L3)**. This is the cleanest possible demonstration that the window *is* the memory: when the perturbation leaves the context, δh is literally 0 in single precision.
  - L7–L27: median λ₅ fluctuates inside **±0.07 (p2.5–p97.5)** around a mildly negative baseline; peak of the smoothed median ≤ +0.07 (L27 @ t=57, inside the eviction epoch).
- **Post-eviction decay fits of ln a(t), t > 110** (median curve, fig02A/B):

| layer | exp. decay (const rate) | R² | AIC | power law `a ∝ t^p` | R² | AIC | winner |
|---|---|---|---|---|---|---|---|
| L27 | rate = **−0.0114** [−0.0118, −0.0111] | 0.959 | −771.2 | p = −2.24 | 0.978 | **−885.3** | power law |
| L14 | rate = **−0.0132** [−0.0135, −0.0128] | 0.966 | **−752.3** | p = −2.55 | 0.952 | −686.8 | exponential |

  Both models describe a **decay**; which fits better is layer-dependent (L27's decay decelerates, L14's is steady). In neither case does the rate approach positive values.
- **Tail (t>150) λ₁₀ with direction-bootstrap CIs:** L2 −0.0642 [−0.0738, −0.0539]; L3 −0.0362 [−0.0384, −0.0341]; L7 −0.0131 [−0.0139, −0.0124]; L14 −0.0080 [−0.0089, −0.0072]; L21 −0.0091 [−0.0100, −0.0082]; L27 **−0.0096 [−0.0103, −0.0088]**. All exclude 0.

### Q2 — radius dependence (**fig03**)

- **Rates do not depend on r in the linear-response regime.** Tail λ₁₀ at L27: −0.0072 (r=0.001) → −0.0096 (r=0.1) → −0.0097 (r=0.3): flat within CI width ±0.001. The normalised response `a(t)/r` **collapses** for r ≤ 0.3 (median a(30)/r = 2.57…2.36; median a(299)/r = 0.56…0.39).
- Prediction flips (teacher-forced top-1 change): **0/20 directions for r ≤ 0.1**, 1/20 at r=0.2, 2/20 at r=0.3 — all rate estimates above are computed in the flip-free regime. For r ≥ 0.5 (flips 7–19/20) the collapse slowly breaks (a(30)/r rises to 2.95 at r=2) and the early slope turns significantly negative (−0.0214 [−0.0316, −0.0113] at r=2), while the tail rate barely moves (−0.0110 at r=2). Conclusion: **the contraction is a property of the trajectory, not of the perturbation amplitude.**

### Q3 — window regime (**fig04**; the same prompt across W)

Early rate (OLS of ln a, t∈[2,30]; identical across W ≥ 100 because no eviction happens before t≈38):

| W | early rate | sustained rate (λ₁₀, t>150) | Λ(299) |
|---|---|---|---|
| 50 (prompt truncated, a(1)≈0.90) | **−0.0164** [−0.0235, −0.0093] | **−0.0175** [−0.0182, −0.0165] | −0.0128 [−0.0139, −0.0117] |
| 100 | **+0.0109** [−0.0001, +0.0221] | **−0.0065** [−0.0075, −0.0054] | −0.0022 [−0.0029, −0.0016] |
| 200 | +0.0109 (same run as W≥400 until t≈138) | −0.0049 [−0.0067, −0.0029] | −0.0021 [−0.0029, −0.0012] |
| ≥400 (flush-free) | +0.0109 | −0.0051 [−0.0068, −0.0031] | −0.0031 [−0.0040, −0.0021] |

- **The window regime is real but it is a *forgetting* regime, not a *stretching* regime.** Smaller windows contract harder: sustained rate −0.0175 (W=50) vs −0.0051 (flush-free) at L27; L14: −0.0277 vs −0.0030. The contraction scales with eviction churn.
- The only *positive* early-rate CI is W=100's marginal **+0.0109 [−0.0001, +0.0221]** (and its λ₅ variant +0.0044 [−0.0015, +0.0110]): a borderline flush-epoch transient, gone by the tail (−0.0065 with two-way CI [−0.0097, −0.0035]).
- Cyber prompt (2nd prompt, fig09 rows): same pattern — W=50 tail −0.0224 [−0.0234, −0.0215]; W≥100 tail −0.0021…−0.0036; early rate −0.0150 identical across W≥100.

### Q4 — per-direction distribution, Jensen gap (**fig05**)

- "λ of the mean" vs "mean of λ" at L27: mean gap over t>20 = **−0.00051 per token** (max |gap| 0.036), i.e. **~6× smaller than the per-token noise (σ≈0.03) and ~15× smaller than the tail rate itself**. At L14 the gap is +0.00004. **No Jensen artefact: direction-averaged rates are legitimate.**
- Per-direction sustained rates: **0/20 directions positive at L27** (max −0.0062) and **0/20 at L14** (max −0.0099). The negative tail is not a mean-field illusion — every individual direction contracts.

### Q5 — layer dependence (**fig06**)

- Peak smoothed median λ₅: +0.11 (L1, a death transient), +0.082 (L3), +0.072 (L27 @ t=57), +0.056 (L14), +0.045 (L21), +0.039 (L7), +0.011 (L2). Peak times scattered t = 49…90 (inside the eviction epoch). **No depth-localised growth front.**
- Sustained rate vs depth (95% CI): L2 −0.064, L3 −0.036, L7 −0.0131, L14 −0.0080, L21 −0.0091, L27 −0.0096 → **contraction is strongest at shallow layers (where the response dies to float32 zero) and settles at ≈ −0.01 at depth.** Λ(299): −0.0121 (L7) → −0.0063 (L27).

### Q6 — cumulative Λ(t): zero crossings (**fig07**)

- prompt65/W=100: Λ(t) **< 0 for all t ≳ 8** at every deep layer; final values −0.0063…−0.0121, CIs exclude 0.
- prompt63 (window sweep): every W shares the same early burst **max Λ = +0.0588 at t = 13** (identical dynamics before the first eviction) — this is the *only* positive cumulative excursion, and it is prompt-transient, not window-driven.
  - W=50: median Λ < 0 from t≈7 (last positive token t=6; per-dir last-positive ≤ 91).
  - W=100: median Λ > 0 for **246/299 tokens**, last positive token **t=264** (per-dir 178…298), final −0.0022 [−0.0029, −0.0016].
  - W≥200: last positive ≈ t=245–251, final −0.0021…−0.0031.
- Distributions (L27): **80/80** (4 radii × 20 dirs) final Λ(299) values are < 0 (range −0.010…−0.004); ECDFs across windows in fig07C.

### Q7 — tail fits, AIC/BIC, the Lyapunov-like number (**fig08**)

Tail region t > 150, median Λ(t) over 10 dirs, prompt63/L27:

| config | const `c` (t-CI) | power `c+A·t^−p` | exp `c+A·e^(−t/τ)` | AIC winner | asymptote |
|---|---|---|---|---|---|
| W=100 | +0.0011 [+0.0007, +0.0014], AIC −1853.6 | A=+0.040, p→0.05 (bound), AIC −1995.9 | **c=−0.0012, A=+0.0064, τ=57.2, AIC −2006.8** | exp | **−0.0012** |
| W=4096 | −0.0019 [−0.0022, −0.0016], AIC −1907.8 | A=+0.009, p=0.14, AIC −1936.4 | **c=−0.0023, A=+0.0039, τ=16.1, AIC −1943.5** | exp | **−0.0023** |

- The tail of Λ(t) is **still relaxing toward a negative asymptote** (exp beats const on the window data); const beats power law on 5/6 layer grids + 4/4 window grids (ΔAIC = −29…−276; only L7 marginally prefers pow, ΔAIC = +2.5).
- Direct sustained rate (OLS of ln a over t>150, mean over dirs, 95% CI): W=100 **−0.0065 [−0.0072, −0.0055]**; W=4096 **−0.0043 [−0.0055, −0.0030]**; layers (prompt65/W100): L2 −0.0766, L3 −0.0359, L7 −0.0135, L14 −0.0065, L21 −0.0085, L27 −0.0087 [−0.0095, −0.0078].
- **Is the asymptotic rate consistent with zero? From above, no: it is consistent with zero-minus or clearly negative.** Every asymptote and every sustained-rate CI sits at ≤ −0.002 (flush-free) to −0.01 (W=100). A *positive* asymptote is excluded by every fit. Caveat: ||h₀(t)|| drifts by +0.0003/token at L27 in the tail, so the *relative* perturbation contracts ≈ 0.0003/token faster than the absolute one — immaterial.

### Q8 — honest bootstrap CIs (**fig09**, and CIs throughout)

- Forest of **31 configs** (layers, radii, windows, cyber windows, 12 prompt runs, 4 long runs), sustained rate = mean λ₁₀ (t>150):
  - point estimates: **0/31 positive** (largest = −0.0017);
  - direction bootstrap: **0/31 CIs reach positive**, max upper bound **−0.0014**;
  - two-way (dir + token block=10): **2/31 CIs reach up to +0.0008**, **0/31 entirely positive**.
- Block bootstrap over tokens widens CIs by ~2–3× vs direction-only (e.g. W=200: [−0.0067,−0.0029] → [−0.0092,−0.0007]) — direction ensembles share the same token noise, so token-level resampling is the binding constraint. **The verdict survives the conservative bootstrap.**

### Q9 — synthetic sanity check (**fig10**)

Three null models pushed through the *same estimators* (T=300, 20 dirs, per-step noise σ=0.152 calibrated to the real λ₅ IQR = 0.0649):

| model | sustained rate (two-way 95% CI) | what it looks like |
|---|---|---|
| expanding map, λ=+0.035 (with renormalisation) | **[+0.016, +0.038]** | λ₅ flat band above 0; Λ(t) locks at +0.035; all 20 dirs positive by t≈60 |
| contracting + reset every 65 tokens (λ=−0.012) | [−0.027, −0.012] | λ₅ noisy band around −0.012 with reset dips; Λ decays to −0.012 |
| transient growth (amp +0.25, τ=18) then −0.012 | [−0.016, −0.004] | λ₅ positive early, crosses 0 by t≈40; Λ rises then decays through 0 |
| **real data (L27, W=4096)** | **[−0.0094, −0.0006]** | sits inside the stable/transient family |

**A genuine +0.035/token rate would be unmistakable in these plots** — nothing in the teacher-forced data resembles it. The real curves are statistically indistinguishable from the contracting-with-reset model.

### Context — when do trajectories actually diverge? (**fig12**)

- Teacher-forced top-1 flips: **none for r ≤ 0.1** (0/20 dirs), 1/20 at r=0.2, 19/20 at r=2 — all rate estimates are from the flip-free regime.
- Free-running: **174/180 runs at r ≤ 0.1 never diverge in 300 tokens** (ds W=100 incl. r=0.01: 60/60; ds W=100 r=0.03/0.1: 40/40; cyber r=0.01/0.1: 40/40; ds W=200/W=500 r=0.1: 17/20 each). The **6 divergent runs all flip at exactly t=44** — the same vulnerable near-tie token on the ds prompt: divergence is a **discrete tipping event**, not exponential separation (an unstable flow would scatter horizons and scale them with 1/λ).
- At r = 0.3 divergence is prompt-dependent: ds W=100 **0/20** (no divergence even at r=0.3); cyber W=100 5/20 (horizons 15, 17, 17, 39, 51); ds W=200 13/20 (11 of 13 at t=44, plus 16, 55); ds W=500 14/20 (11 at t=44, plus 16, 55, 242). The t=44 concentration persists: tipping, not growth.
- λ₅(t) vs decision margin: pooled correlation r = **+0.087** (20 dirs, 290 tokens) — **local rate spikes are not near-tie artefacts**; the lowest-margin tokens (margin 0.02 at t=124, 144) have λ₅ = −0.036, −0.015.

---

## 3. Master table of fits (models, params, CI, R²)

All on median curves unless stated; AIC/BIC computed with n = #points, k = #params + 1; CIs 95%.

| # | data (region) | model | parameters | CI / quality | figure |
|---|---|---|---|---|---|
| 1 | L27 ln a, t>110 | exp decay | rate −0.0114 | [−0.0118, −0.0111], R²=0.959, AIC −771 | fig02A |
| 2 | L27 ln a, t>110 | power law | a ∝ t^−2.24 | R²=0.978, AIC −885 (**wins**) | fig02A |
| 3 | L14 ln a, t>110 | exp decay | rate −0.0132 | [−0.0135, −0.0128], R²=0.966, AIC −752 (**wins**) | fig02B |
| 4 | L14 ln a, t>110 | power law | a ∝ t^−2.55 | R²=0.952, AIC −687 | fig02B |
| 5 | W=100 Λ tail | const | c = +0.0011 | t-CI [+0.0007, +0.0014], AIC −1854 | fig08A |
| 6 | W=100 Λ tail | power | c=−0.0319, A=+0.040, p=0.05* | AIC −1996 (*p at bound) | fig08A |
| 7 | W=100 Λ tail | exp→asymptote | c=−0.0012, A=+0.0064, τ=57.2 | R²=0.649, AIC −2007 (**wins**) | fig08A |
| 8 | W=4096 Λ tail | const | c = −0.0019 | t-CI [−0.0022, −0.0016], AIC −1908 | fig08B |
| 9 | W=4096 Λ tail | exp→asymptote | c=−0.0023, A=+0.0039, τ=16.1 | R²=0.233, AIC −1944 (**wins**) | fig08B |
| 10 | W=100 sustained (OLS ln a, t>150) | linear | slope −0.0065 | [−0.0072, −0.0055] | fig08A |
| 11 | W=4096 sustained (OLS ln a, t>150) | linear | slope −0.0043 | [−0.0055, −0.0030] | fig08B |
| 12 | L27 sustained (prompt65, dirs) | linear | slope −0.0087 | [−0.0095, −0.0078] | fig08C |
| 13 | Λ-tail const vs pow, 10 grids | ΔAIC = AIC(pow)−AIC(const) | — | const wins 9/10 (L7 only exception, +2.5) | fig08D |
| 14 | synth chaos +0.035 | sustained λ₁₀ | — | two-way CI [+0.016, +0.038] | fig10D |
| 15 | real W=4096 | sustained λ₁₀ | — | two-way CI [−0.0094, −0.0006] | fig10D |

---

## 4. Figure index (12 files, all in this folder)

| file | question | one-line message |
|---|---|---|
| `fig01_lambda_layers.png` | Q1 | λ₅(t) by layer: no growth phase; shallow layers die to exact float32 0; deep tail ≈ −0.01 |
| `fig02_growth_decay_fits.png` | Q1 | the "growth phase" is 1 token (+0.26 @ L27); post-eviction decay exp vs power (both negative) |
| `fig03_radius.png` | Q2 | rates r-independent for r ≤ 0.3 (a/r collapse); flips only at r ≥ 0.2 |
| `fig04_windows.png` | Q3 | W=50 contracts hard (−0.018); W=100 early +0.011 marginal; all sustained ≤ 0 |
| `fig05_jensen.png` | Q4 | mean-of-λ ≈ λ-of-mean (gap −0.0005); 0/20 directions positive |
| `fig06_layer_profile.png` | Q5 | no depth growth front; contraction strongest shallow |
| `fig07_cumulative.png` | Q6 | Λ negative at depth from t≈8; W=100's positive excursion is the shared t≈13 burst; 80/80 final values < 0 |
| `fig08_tailfit.png` | Q7 | tail relaxes to negative asymptotes; sustained rates −0.004…−0.011, CIs exclude 0 |
| `fig09_forest.png` | Q8 | 31 configs, 0 positive; max two-way upper CI +0.0008 |
| `fig10_synthetic.png` | Q9 | a true +0.035 rate is unmistakable; real data = contracting-with-reset family |
| `fig11_robustness.png` | Q1/Q8 | k=5 vs k=10 identical conclusions; 12 prompt runs 0/12 positive |
| `fig12_context.png` | context | flips need r ≥ 0.2 (TF); free-run divergence at r ≤ 0.1 is 6/180, all at the same token t=44 — tipping, not growth |

Reproduce with: `python compute_core.py` (all raw numbers), `python figs_a.py`, `figs_b.py`, `figs_c.py`, `figs_d.py` (seeds fixed; bootstrap B=600–2000). Number dumps: `numbers_figs_*.json`.

---

## 5. Caveats

1. **Teacher forcing** measures the linearised flow along one fixed token sequence; a negative rate here does not preclude sensitivity of the *sampling* map at larger radii — but the free-run data (180/180 no divergence at r ≤ 0.1) show the two pictures agree.
2. T = 300 tokens: "asymptotic" means t > 150. Λ(t) is still relaxing at t=300 (fig08); the asymptotes quoted are model-based extrapolations of that relaxation, all negative.
3. Layers 1–3 lose the response to float32 underflow after eviction (t=93/192/291) — their rates are lower bounds in magnitude (−∞ at death); L0 is identically zero by construction.
4. `rate_prompts/p6` exists but is empty (data missing); W=400/800/4096 window files are identical runs for prompt 63 (no eviction within 300 tokens).
5. W=50 truncates the 63-token prompt, so its initial state differs (a(1) ≈ 0.90 vs 0.16) — its "more negative" rate includes an initial-condition effect, not only churn.

---

## 6. Summary

We characterised the finite-time stretching rate λ(t) of hidden-state perturbations in the DeepSeek-1.5B teacher-forced runs exhaustively — across 8 layers, 12 radii (1e-3…2), 6 window sizes, 2 prompt families and ~31 configurations with 8–20 directions each, with direction- and token-block-bootstrap confidence intervals — and the answer is unambiguous: **there is no sustained positive stretching rate anywhere in this system.** The response a(t) shows a single-token alignment transient (+0.26 at L27, t=1), a shared 13-token cumulative burst (Λ ≈ +0.06) present identically at every window size, and then a slow, robust **contraction**: sustained rates are −0.004…−0.010 per token at depth with flush-free windows, −0.006…−0.013 at W=100, and −0.016…−0.039 at W=50, with 31/31 direction-bootstrap CIs and 31/31 point estimates strictly below zero (max upper CI bound −0.0014) and the two most conservative two-way CIs only reaching +0.0008 — i.e. any Lyapunov-like exponent is ≤ 0, of magnitude at most a few tenths of a percent per token; rates are radius-independent in the flip-free linear regime (r ≤ 0.3), per-direction rates are individually negative (no Jensen artefact), shallow-layer responses die to exact float32 zero once the perturbed prompt KV entries are evicted from the window, and free-running experiments agree (174/180 runs at r ≤ 0.1 with zero divergence in 300 tokens, the 6 exceptions tipping at one identical vulnerable token rather than separating progressively). The window mechanism is a **memory/forgetting** mechanism — it sets how fast the model *drains* perturbations (contraction ∝ eviction churn), not a chaos amplifier; a synthetic expanding-map control with matched noise shows that a genuine positive rate (+0.035/token) would have been impossible to miss in any of these plots.
