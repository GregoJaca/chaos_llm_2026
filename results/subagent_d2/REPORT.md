# D2 / deterministic-signature search in DeepSeek-1.5B last-layer trajectories
**Subagent D2 report — 18 Nov 2026 — CPU-only (numpy/scipy), all pairwise work block-chunked, no N×N matrices ever stored.**

Data: `pecs/{cyber,future,great,interstellar}/last.pt` ([T,1536] fp32), `launch/hidden_states_cond_0_layer_-1.pt`, `pecs/hidden_states_first.pt` (first layer, many duplicates; used as a null). All `.pt` files loaded with a torch-free pickle/zip reader (`pt_load.py`). Everything below is reproducible from the scripts in this folder (`exp0..exp4*.py` → JSON + PNG).

---

## 0. Headline

The dataset is **two datasets**:

1. **cyber, future, great are near-exact repetition loops** (period P = 53 / 74 / 69 tokens; fidelity ≈ 1.7–1.8% of cloud scale; 86–91% of states recur within 5% at lag P). This is LLM repetition-degeneration, and it **generates every "low-dimensional/deterministic-looking" number on those three trajectories**: the moderate D2 values, the high predictability, the low local dimensions. A trivial synthetic model — period-P phase means + linear drift + iid residuals — reproduces all of it (Sec. 2.1).
2. **interstellar (T=1026) and launch (T=3039) are non-repeating** and are the only honest test cases. On them: **no D2 plateau, no dimension below finite-sample ceilings, and (except for one modest, unconfirmed predictability gap on launch) no dynamic signal beyond surrogate controls** (Sec. 2.2, Sec. 3).

The main agent's core observation — raw D2 ≈ time-shuffled D2 — is confirmed but is a **methodological tautology**: time-shuffling preserves the point cloud, and D2/TwoNN/MLE/PCA are functions of the point cloud. I verified raw ≡ shuffled to ≤0.018 max |Δlog C| (≤0.0024 for C>1e-4; 0.0 exactly for interstellar/launch/first). That control can never reject *statics*; the right question is whether any statistic separates the *ordered* trajectory from statics-preserving or second-order-preserving surrogates. Mostly, nothing does — except repetition.

---

## 1. Positive results (with effect sizes and controls)

### 1.1 The three loopy trajectories contain a real, stable limit cycle — but it is repeated text, not an attractor of a chaotic system
**This is a genuine positive finding about the data** (not an estimator artifact), and it is large:

| traj | P (tokens) | fidelity `median‖x_t−x_{t+P}‖/scale` | frac recurring <5% scale | step size / scale | cycle-model variance explained |
|---|---|---|---|---|---|
| cyber | 53 | **0.018** | 0.906 | 0.93 | **92.1%** |
| future | 74 | **0.018** | 0.858 | 0.93 | **87.4%** |
| great | 69 | **0.017** | 0.856 | 0.85 | **84.9%** |

- Each step is ~0.9× the whole-cloud scale (states jump wildly per token), yet after P steps the state returns to within ~1.8% of scale. Norm-autocorrelation peaks at multiples of P (cyber lag 53: 0.963). The loop degrades slowly: repeat distance grows ≈ +0.5–0.6% of scale per cycle (0.018 → 0.058 by cycle 8), i.e. a **drifting limit cycle**.
- Residuals around the cycle are **persistent across cycles** (lag-1 across-cycle autocorr ρ = 0.60 / 0.44 / 0.45), i.e. the repeated segment mutates slowly rather than jittering.
- **Controls (exp2)**: synthetic S1 = phase-means + drift + *iid* residuals reproduces kNN prediction NSE 0.83–0.93 vs data 0.998–0.999 (constant across horizons h=1..32); S1b (AR(1) residuals, ρ matched) reproduces the Kaplan–Glass determinism score (data 1.10–1.35 vs S1b 1.20–1.33); S2 = **same statics, phase order randomized** gives NSE ≈ **−0.05** and det-score 7–17. So: *statics alone predict nothing; the period cycle alone reproduces ~90% of the apparent determinism; the rest is residual persistence.* Nothing chaotic remains once the cycle is removed.

### 1.2 Launch: a modest nonlinear predictability gap beyond *all* surrogate classes (suggestive, single realization, unconfirmed)
kNN (simplex-style, library = first half, Theiler W=600 > partial-recurrence lag 483, k=12, inverse-distance weights), NSE:

| horizon h | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| **launch data** | **0.569** | **0.449** | **0.360** | **0.297** | **0.258** | **0.223** |
| phase-rand (n=7) | 0.108±0.01 | 0.038 | −0.003 | −0.036±0.005 | −0.047 | −0.054 |
| MIAAFT (n=3) | 0.102 | 0.042 | −0.003 | −0.034±0.011 | −0.043 | −0.056 |
| time-shuffled | −0.112 | −0.110 | −0.102 | −0.103 | −0.103 | −0.108 |
| cov-Gauss / OU | ≈ −0.08 | | | ≈ −0.085 | | |
| ridge-PC50 (data) | 0.435 | 0.278 | 0.201 | 0.147 | 0.124 | 0.104 |
| ridge-PC50 (phase surr.) | 0.439 | 0.275 | 0.182 | 0.120 | 0.089 | 0.062 |

- **Effect size**: data kNN NSE at h=8 exceeds the strongest surrogate class by **+0.33** (0.297 vs −0.036, surrogate sd 0.005) and exceeds the data's own *linear* ceiling by **+0.15** (0.297 vs ridge 0.147). The linear part is fully explained by second-order structure (ridge on data ≈ ridge on phase surrogate, as it must be).
- **Caveats that keep this from being a clean positive**: the Kaplan–Glass continuation test shows **no sharpness advantage** (det score h=1/8/16: data 1.06/1.21/1.37 vs phase 1.26/1.37/1.38) — nearby states' futures are *not* markedly more similar than a linear surrogate's. So the +0.15–0.33 NSE is a weak *statistical* predictability edge (probably via mild recurrence structures — the lag-483 partial recurrence, p10 = 9.6% scale), not a sharp deterministic map. It is a single trajectory; no cross-prompt replication exists for it. Label: **suggestive; requires more non-loopy prompts to confirm.**

### 1.3 Static (not dynamic) low-dimensionality is real and large — but it is token-state geometry, not an attractor
TwoNN and Levina–Bickel MLE separate the data clouds massively from covariance-matched Gaussians:

| traj | TwoNN raw | TwoNN shuffled | TwoNN cov-Gauss | TwoNN iid-Gauss | MLE k=8 raw / covG | PCA PR | #PCs for 90% var |
|---|---|---|---|---|---|---|---|
| cyber | 6.72 | 6.72 | 26.7 | 100.8 | 2.33 / 23.3 | 12.2 | 27 |
| future | 6.64 | 6.64 | 28.6 | 111.7 | 2.09 / 25.3 | 12.7 | 32 |
| great | 5.92 | 5.92 | 28.1 | 97.6 | 2.20 / 24.6 | 14.0 | 31 |
| interstellar | 6.23 | 6.23 | 43.9 | 90.8 | 4.38 / 36.4 | 20.4 | **161** |
| launch | 3.32 | 3.32 | 52.4 | 99.5 | 3.17 / 44.2 | 22.3 | **223** |
| first | 16.76 | 21.09* | 23.6 | 213.8 | 15.6 / 21.2 | 14.7 | 24 |

\* first layer differs from its shuffle only because exact duplicates make kNN tie-breaking order-dependent.

Reading: the clouds are **locally thin** (MLE ~2–4, TwoNN ~3–7) yet globally high-dim (n90 = 161/223 for the non-loopy pair; median nearest-neighbour distance only 0.25–0.37× scale). That is the signature of an inhomogeneous union of token-conditioned state clusters — consistent with known LLM anisotropy — **not** of a low-dimensional attractor. Note also that TwoNN/MLE *estimators themselves* are unreliable at N≈1–4k in d=1536 (iid-Gauss "dimensions" of ~100 are pure finite-N bias), so only the raw-vs-null contrast is meaningful, and that contrast says "inhomogeneous cloud", nothing dynamical.

---

## 2. Negative results (exact numbers)

### 2.1 No D2 plateau anywhere; every "D2" is a fit-window choice
Local slope d logC/d logε at C = 0.05/0.1/0.3/0.5/0.7/0.9 (full N, Theiler 0):

| traj | .05 | .1 | .3 | .5 | .7 | .9 | fit C∈[.02,.1] | fit C∈[.1,.9] |
|---|---|---|---|---|---|---|---|---|
| cyber | 5.60 | 5.80 | 3.51 | 3.24 | 2.27 | 0.52 | 1.44 | 3.24 |
| future | 3.42 | 5.84 | 4.61 | 4.28 | 4.28 | 1.06 | 1.23 | 4.16 |
| great | 5.21 | 4.37 | 3.95 | 3.22 | 2.14 | 0.70 | 1.82 | 3.00 |
| interstellar | 9.05 | 9.00 | 6.11 | 4.62 | 4.62 | 1.09 | n/a | 5.33 |
| launch | 9.22 | 9.33 | 6.10 | 3.74 | 2.42 | 1.55 | n/a | 5.12 |
| first | 2.04 | 8.54 | 7.01 | 7.22 | 7.22 | 1.42 | 0.02 (duplicates) | 5.77 |

Slopes decline **monotonically with ε** in every trajectory; the same data yields D2 anywhere from 1.2 to 5.8 depending on the ε window. Nothing resembling a scaling plateau exists at any N (fig1a/1b show N=1024/2048/full — curves shift with N as expected for finite samples, no convergence to a plateau).
The small-ε slopes 9.0–9.3 (interstellar, launch) **exceed the Eckmann–Ruelle ceiling** 2·log₁₀N (6.22 for N=1026, 6.97 for N=3039), i.e. they are finite-sample inflation, not dimension. For the loopy three, the small-ε slopes ~5.2–5.8 sit below the ceiling but measure the loop-tube fine structure (see 2.3).

### 2.2 Raw ≡ time-shuffled — verified, and explained as tautology
max |Δlog C| between raw and time-shuffled curves: cyber 0.018 (occurring at C≈8e-6, ~70-pair counts; 0.00064 for C>1e-4), future 0.010 (0.0018), great 0.0024, **interstellar 0.0, launch 1e-6, first 0.0**. Residual differences are float32 BLAS boundary flips at the smallest counts. Likewise TwoNN raw = shuffled to 2 decimals everywhere (table above). Conclusion: the main agent's raw-vs-shuffled D2 equality was never evidence for or against anything dynamical — the two objects being compared are identical by construction.

### 2.3 Theiler windows barely matter — the D2 values are not a temporal-neighbour artifact
Slope-at-C with Theiler exclusion w (smoothed):
- cyber: w=0 → 4.70/3.57/2.68/0.61; w=53 → 4.70/3.57/2.68/0.62; w=159 → 4.70/3.59/2.69/0.62; w=1000 → 4.65/3.70/2.76/0.66 (C=.05/.3/.5/.9).
- interstellar: w=0 → 8.60/5.33/3.58/0.88; w=500 → 9.28/5.66/3.71/0.88.
- launch: w=0 → 8.98/5.12/3.38/1.12; w=500 → 9.48/5.32/3.47/1.13.

Even excluding all pairs within 1000 steps (≈ 19 loop cycles for cyber), the C-curve does not change materially: **the correlation sum is dominated by loop copies at all lags (cyber NNs at lag 106, 212, …) and by static cloud geometry, not by adjacent-time pairs.** Small-ε D2 for the loopy set ≈ 1.2–1.8 (C∈[.02,.1]) is the drifting tube; the [0.1,0.9] fits ≈ 3.0–4.2 are the phase-cluster ring; neither is an attractor dimension.

### 2.4 interstellar dynamics: clean negative
Best-case kNN NSE 0.329 (h=1) equals the linear ceiling (ridge 0.342) and is matched by phase surrogates at longer horizons (h=8: data 0.042 vs phase −0.033±0.014, MIAAFT −0.035±0.006); determinism score is *worse* than surrogates (data 2.32 vs phase 1.34 at h=8). Persistence NSE is negative at all horizons (−0.40…−1.00). Nothing dynamic beyond linear autocorrelation (per-coordinate lag-1 autocorr median 0.31) survives any control.

### 2.5 No continuation sharpness anywhere outside the loops
Kaplan–Glass continuation distance rc vs neighbour distance r (fig4b): data curves sit on the diagonal (rc ≈ r) for interstellar/launch and are statistically indistinguishable from phase/MIAAFT surrogates; only the loopy trajectories (via their exact repeats) show rc ≪ cloud scale. A sharp deterministic flow would show rc/r → 0 at small r; no non-loopy trajectory does.

---

## 3. Figures (best five; all in this folder)

1. **`fig_loops_scan.png`** (+ `fig_loops_drift.png`) — state-distance vs lag, all six trajectories: cyber/future/great collapse to 1–2% of cloud scale at P=53/74/69; interstellar/launch never leave ~90–95%. The drift panel shows the slow degradation of repeats (drifting cycle). *This figure reframes the whole dataset.*
2. **`fig1a_d2_scaling.png`** + **`fig1b_d2_slopes.png`** — the proper scaling plots the user asked for: C(ε) on a shared normalized axis (ε/cloud scale) for N=1024/2048/full with the time-shuffled overlay (visually identical), and local slopes with Theiler variants. Shows *why* there is no plateau and how the "D2" value is window-dependent.
3. **`fig2_synthloop_C.png`** + **`fig2_synthloop_pred.png`** — the deflation control: the data's C(ε) curve is reproduced by a statics-only phase-shuffled null at all but the smallest ε, and the data's NSE≈0.999 predictability is reproduced by the trivial cycle+drift+residual model while statics alone (S2) predict nothing (NSE≈−0.05).
4. **`fig3_intrinsic_dim.png`** — TwoNN vs nulls (raw = shuffled exactly; raw ≪ cov-Gauss), MLE(k) curves, PCA spectra: locally thin, globally spread clouds; estimator finite-N bias exposed by the iid-Gauss bars.
5. **`fig4a_prediction.png`** + **`fig4b_continuation.png`** — the honest dynamics test on the non-loopy pair: launch's modest NSE gap above every surrogate class (and above its linear ceiling), interstellar's clean null; continuation curves on the diagonal = no deterministic sharpness.

---

## 4. Are the exp0/exp1/exp2 (loop) findings real or artifacts?

**Real.** The loops are a property of the data, not of any estimator: norm autocorrelation, direct state distances at lag P (fidelity 1.7–1.8% of scale over ~55–77 cycles), 86–91% of points recurring, and 85–92% of variance explained by a P-phase linear model are mutually consistent, and every "positive" dynamical number on those three trajectories (NSE 0.998, det-score ~1.2, small-ε D2 ~1.2–1.8, TwoNN ~6) is quantitatively reproduced by synthetic trajectories that contain *nothing but* a period-P cycle, drift, and persistent residuals. What they are **not** is evidence of low-dimensional chaotic dynamics — they are repetition-degenerated text. The one *interpretive* risk is calling the cycles "deterministic structure" without the S2 control; with it, the claim is fully deflated. Conversely, the D2≈shuffled result of the main agent is correct but vacuous (Sec. 2.2), and the moderate D2 values (3–5) reported previously are window- and artifact-dependent (Sec. 2.1/2.3) — a conclusive negative for D2-as-attractor-dimension on this dataset.

---

## 5. Scope / what was not finished
Delay-embedding D2 with Theiler (planned exp6), a systematic permutation-entropy-vs-IAAFT suite (exp5), and windowed/nonstationary D2 (exp7) were cut at wrap-up. Two smoke-level observations worth recording: (i) a single-coordinate PE on cyber (m=5: 0.85 vs IAAFT 0.967 vs shuffle 0.997) is exactly what the period-53 cycling predicts — fewer ordinal patterns — and should *not* be read as nonlinear richness; (ii) first-layer PE/TwoNN are dominated by duplicate states. Recommended follow-ups: (a) regenerate the three loopy prompts with anti-repetition (repetition penalty / temperature) and rerun exp1/exp4; (b) collect several non-loopy trajectories to test whether launch's +0.15–0.33 NSE gap replicates; (c) delay-D2 on PC1 of launch vs IAAFT.

---

## 6. One-paragraph summary for the main agent

Three of the five last-layer trajectories (cyber P=53, future P=74, great P=69) are **near-exact repetition loops** — fidelity 1.8% of cloud scale, 86–91% of states recurring, cycle model explaining 85–92% of variance — so every low-D/deterministic-looking number on them (D2≈1.2–5, TwoNN≈6, kNN NSE≈0.998) is reproduced by a trivial period+drift+jitter synthetic and is repeated-text degeneration, not attractor geometry; their raw-vs-shuffled D2 equality is a tautology (verified: max|Δlog C|≤0.018, TwoNN identical to 2 decimals) because time-shuffling preserves the point cloud. On the two non-repeating trajectories the answer is negative: no C(ε) plateau at any N (slopes decline monotonically 9→1 across ε), small-ε slopes 9.0–9.3 exceed the Eckmann–Ruelle ceiling (6.2/6.97) so they are finite-N inflation, Theiler windows change nothing (w=0 vs 1000), interstellar shows no dynamic signal beyond its linear ceiling (NSE 0.042 vs phase −0.033±0.014 at h=8; det-score worse than surrogates), and no Kaplan–Glass continuation sharpness exists anywhere outside the loops. The only positive worth chasing: **launch** shows kNN NSE +0.33 above the best second-order/marginal-preserving surrogate (0.297 vs −0.036±0.005 at h=8) and +0.15 above its own linear ceiling (ridge 0.147), with the linear part fully surrogate-explained (ridge data ≈ ridge phase) — but it is one trajectory, shows no continuation sharpness (det 1.21 vs phase 1.37), and is plausibly mild recurrence structure, so treat it as suggestive until replicated on more non-loopy prompts. Practically: stop interpreting D2/TwoNN on these trajectories as attractor dimensions, exclude or regenerate the three loopy prompts with anti-repetition settings, and use the launch-style predictability-vs-MIAAFT gap (not D2) as the primary statistic if the project continues.
