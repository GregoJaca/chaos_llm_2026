# Are the 20 perturbation directions equivalent? A second, hostile look.

**Data:** prompt-embedding perturbations `r*u` of a fixed 65-token prompt, 20 directions `u`
forming a *regular simplex* in a 20-dimensional subspace of token-embedding space
(`w100_layers/a1.npz`: 12 radii 1e-3..2, 8 layers, 300 teacher-forced tokens;
plus two **replicates with the same prompt and same directions but different sampled
continuations**: `a1.npz` (w200) and `w500/a1.npz`); free-run flip steps
(`fr_w100_large.npz`, r=0.5/1/2); current-state injections (`state.npz`, `state_mid.npz`).
All scripts and figures for this report are in this folder (`t1_gain.py` ... `t8_extra.py`,
`fig1..fig8*.png`, `numbers_t*.json`).

**The previous conclusion was:** "directions are mostly equivalent: PC1 explains 94-97%
of log-response variance; median pairwise correlation 0.94-0.98; no clusters."
**We tried hard to break it. It breaks — in four specific, quantified ways — but not
everywhere.** The honest summary is: *a shared response backbone (timing, token-profile,
near-tie bottlenecks) plus direction-specific gain, shape modulation, exponent, and — most
importantly — the SIGN of the margin response, which is what actually decides flips.*

---

## 1. What "equivalent" would mean, and which version survives

Two different claims must be separated:

- **E1: "equivalent up to a scalar gain"** — each direction's response is a single number
  (its gain) times one universal response curve. This is FALSE, in four measurable ways
  (gain spread, layer-dependence, radius-dependence, exponent spread; sections 2, 3, 6).
- **E2: "equivalent in every respect"** — after removing gain, nothing direction-specific
  remains. Also FALSE: shapes, hidden-state response vectors, and the sign of the margin
  response are all direction-specific and highly reproducible (sections 3, 4, 5, 8).
- What IS equivalent across directions: the *timing* of propagation (an identical
  per-layer causal front), the identity of the vulnerable tokens (90-95% of directions
  flip, if at all, at the same handful of near-tie tokens), and the overall
  token-profile of the response *magnitude* (pairwise corr 0.89-0.92 at every radius).
- And one entire regime is close to E1: **random Gaussian directions injected into the
  current state are nearly equivalent** (gain spread 1.6-2.1x, no memory across injection
  steps; section 7). Non-equivalence is a property of the *structured token-subspace*
  directions, not of perturbations in general.

---

## 2. Gain: directions differ by up to an order of magnitude, and the gain is a stable
property of the direction (Fig. 1, `numbers_t1.json`)

Gain = median over 300 tokens of ||delta h_t|| at layer 27.

| radius | max/min gain over 20 directions |
|---|---|
| 0.001 | **7.3x** |
| 0.01 | **11.0x** |
| 0.1 | **11.8x** (CV = 0.85) |
| 2 | **14.8x** |

Even in the fully linear regime (r=0.001) the strongest direction is 7x the weakest.
This alone refutes E1 in the most practical sense: "equivalent up to a scalar" still means
the scalar varies 7-12x across directions.

**Is a direction's gain one consistent scalar?** Spearman rank correlations of the
20-dimensional gain vectors:

- across **layers** at r=0.1: adjacent deep layers 0.96-0.97, but layer 1 vs 27 only
  **0.34**; median over all pairs 0.79. At r=2 the median drops to 0.57 (min 0.04).
  The gain *develops* layer by layer; a direction that is strong at layer 27 is only
  moderately predictable from its early-layer gain.
- across **radii** at layer 27: within the linear regime rho=0.98, within the saturated
  regime 0.93, but linear vs saturated only **0.48-0.51** — the ranking reshuffles as
  directions saturate at different rates.
- across **continuations** (same prompt + same directions, three different sampled texts):
  rho = **0.80-0.98 at every radius tested**. This is the key positive result: direction
  identity (i.e., *which token-embedding axis it occupies*) persistently determines gain,
  irrespective of what the model happened to generate. The three hottest directions are
  dirs {10, 11, 12} (token axes 843, 73349, 54373) in the w100 and w200 continuations, and
  {10, 11, 14} in w500 — token axis 843 is the top direction in **all three**.

Best figure: `fig1_gain_consistency.png` (panel d: gains reproduce across continuations).

---

## 3. Shape beyond gain: small, but real beyond any doubt (Fig. 2, `numbers_t2.json`)

Decomposition of log-response variance at layer 27, r=0.1 (two-way ANOVA on log ||dh_t||):

| component | share | RMS (log-units) |
|---|---|---|
| common token-profile | **64%** | 0.90 |
| per-direction gain | **30%** | 0.61 |
| direction-specific shape | **6.1%** (2.7% for t>=95) | 0.28 |

The 6% residual is NOT noise:

- noise floor estimated from 4 replicate radii (conservative: it also contains
  float32 rounding and mild nonlinearity) is only **2.2%** of the between-direction
  shape variance; split-half reliability of direction shapes across radius splits:
  **r = 0.999**;
- all 20/20 directions have shapes beyond noise (chi-square with
  autocorrelation-corrected dof, all p < 0.05; most p < 1e-10);
- the same holds at layer 14 (fraction 0.94) and layer 21 (0.98), and in the
  saturated regime (0.98).

So the earlier "PC1 explains 94-97%" is *technically about the wrong object*: with gains
kept, PC1 of the log-curves is 0.82-0.86 at layer 27 (0.88-0.90 at layer 14) — that
number measures the shared token-profile, not equivalence. After removing each direction's
own gain, PC1 is only **0.24**, and the rest is reproducible direction-specific structure.

**How do shapes differ?** Not along one simple axis: correlations of each direction's
shape with token position are significant for 16/20 directions but with mixed signs
(no global "early vs late" split); with baseline margin for 13/20 (mixed signs); the
deviation matrix needs >=3 PCs (29%/24%/13%) to reach two-thirds of its variance.
Each direction has its own token-fingerprint. Burstiness also varies: the
token-wise coefficient of variation of the response ranges 0.86..2.49 across directions
(median 1.39, `fig8_extra.png` panel b).

Best figure: `fig2_shape.png`.

---

## 4. Geometry: the only geometric degree of freedom is which token axis a direction
occupies — and that one DOES predict gain (Fig. 4, `numbers_t4.json`)

The 20 directions form a regular simplex: every row of `simplex` has max coefficient
0.975 on its own axis and -0.0513 on the other 19; mutual cosines are all exactly -1/19;
L1/L2 is 1.95 and max-coefficient is 0.975 **for every direction**. So scalar geometric
predictors (sparsity, dominant coordinate) are constant *by construction* — no
correlation with gain is possible or needed. The only variable is the identity of the
aligned token axis (`subspace[i]`), and section 2 shows it predicts gain robustly
(rho 0.80-0.98 across continuations; token 843's direction is 4-6x stronger than the
median axis in all three continuations).

Hidden-space geometry of the responses (stored vectors for dirs 0, 1, 2 at layer 27):

- different directions do **not** push the hidden state along a common "error mode":
  cos(delta h_i, delta h_j) = 0.19 / 0.46 / 0.13 at r=0.1, and ~0.0 or slightly negative
  at r=2. The response is at least ~3-dimensional and *decorrelates further* in the
  saturated regime.
- the response is nearly transverse to the baseline state: direct computation against
  the stored baseline hidden states (`margins/w100.npz`, h27; token alignment verified by
  gen_ids) gives cos(delta h, h_t) medians -0.064 / -0.000 / +0.022 for dirs 0, 1, 2,
  with 31% of tokens above |cos|=0.1 — about 2.5x the 1/sqrt(1536)=0.026 genericity
  scale, and with per-direction sign differences (dir 0 is mildly anti-radial, dir 2
  mildly radial).
- caveat discovered on the way: the stored `cos` array is a *derived* quantity — it
  equals 1/sqrt(1+(||dh||/||h||)^2) with correlation 0.994 (mean abs. deviation 8e-4) —
  i.e. it carries no information beyond the norm ratio, and it should not be used as an
  independent measurement of state rotation.

Best figure: `fig4_geometry.png` (panel d: gains by token axis across three continuations).

---

## 5. Flips: the practically important non-equivalence (Fig. 5, `numbers_t5.json`)

A flip = the free-running generated text actually changes. Observed flip counts:
**3/20 directions at r=0.5 (exactly the hot trio 10, 11, 12), 8/20 at r=1, 16/20 at r=2.**
So flip risk is strongly direction-stratified, and the earliest flippers are precisely the
high-gain directions. The predicted threshold r*(d) = min_t r_c (note: the stored r_c is a
*MAGNITUDE* threshold, margin*r/|dm| — sign handled separately below) never misses an
observed flip at any radius (0 misses, vs 4/7/3 false alarms — the magnitude criterion
over-calls because a direction can cross |dm| = margin while pushing the margin *away*
from the tie).

**Why do directions flip at different radii — gain or near-tie luck?** Decomposition
log r*(d) = const + tie part - response part, with tie part = log(margin at bottleneck
token / median margin) and response part = log(|dm| at bottleneck / typical |dm|):

| radius | var(tie part) | var(response part) | cov |
|---|---|---|---|
| 1 | 0.27 | **0.69** | 0.11 |
| 2 | 0.53 | **1.21** | 0.65 |

The response part dominates (2.6x the tie part at r=1): early flipping is mostly
*direction-specific response strength at the vulnerable token*, not just tie luck.
Gain alone does not fully determine it: Spearman(log r*, log gain) = -0.61 (r=1),
-0.73 (r=0.5) — strong but far from -1.

**Where** flips happen is shared: 90-95% of directions have their bottleneck token t*
among the 6 nearest-tie tokens (t=123 with margin 0.020 alone is the bottleneck for
11-13 of the 20 directions). The text's near-ties are a common vulnerability — but
*whether* a direction exploits a given near-tie is direction-specific:

- per-direction sign bias of dm at r=1: fraction of tokens with dm<0 ranges
  **0.29..0.69**, with 16/20 directions outside the coin-flip 2-sigma band
  (r=2: 15/20, range 0.27..0.72). Some directions systematically shrink margins, others
  systematically grow them.
- pairwise sign agreement across directions: median **0.53** over all tokens
  (0.67 on the near-tie tokens only) — the sign pattern is mostly direction-specific.
- at the shared bottleneck token t=123, the signed responses at r=1 span
  **-0.073..+0.253**: different directions push the same near-tie in opposite
  directions, and 5/20 would flip at that token alone.

This is the sense in which directions are genuinely NOT equivalent where it matters:
same radius, same vulnerable token, opposite effects on the decision.

Best figure: `fig5_flips.png`.

---

## 6. The scaling exponent is direction-dependent (Fig. 6, `numbers_t6.json`)

- linear regime (r=0.001..0.1): alpha per direction = 0.92..1.01 (std 0.02) — tight, as
  it must be;
- large r (0.3..2): alpha per direction = **0.96..1.63** (std 0.18). A single-scalar-gain
  model predicts identical alphas; they differ by up to 0.7;
- the margin-response exponent |dm| vs r behaves similarly (0.89..1.64, std 0.18);
- alpha correlates with gain only moderately (Spearman -0.40): high-gain directions do
  not simply stay proportional — the saturation schedule is its own direction property;
- alpha is consistent across layers (rho with layer 21: **0.91**; with layer 14: 0.58),
  so it is a stable direction attribute, not noise.

---

## 7. Current-state injections: here the directions really ARE (nearly) equivalent
(Fig. 7, `numbers_t7.json`)

For 32 random Gaussian directions injected into the current state at layer 27 (r=1e-3):

- gain spread across directions: **1.6-2.1x only** (vs 7-12x for the prompt directions);
- rank correlation of gains across injection steps s=16/50/100/150: median **0.04**,
  min -0.25 — a direction that is strong at s=16 has *no* tendency to be strong at
  s=150. (For the |dm| gains: median -0.07.) The user-requested number is in
  `fig7_state.png` panel (b);
- direction-specific shapes do not persist across s either (cross-s correlation of
  gain-removed profiles: median -0.02);
- the response does fade with later injection (median log-drop -0.51, i.e. ~40% weaker
  at s=150 than s=16), and the *fading is somewhat direction-specific* (std 0.20
  log-units; Spearman(gain at s=16, decay) = -0.50);
- layer-14 injections (state_mid, 8 directions): s=50 vs s=100 rho=0.69, but s=16 vs the
  others ~0.07-0.10;
- magnitude: current-state injection at r=1e-3 produces ~6x more ||dh|| at layer 27 than
  a prompt-embedding perturbation of the same radius (0.0083 vs 0.0014 median).

Interpretation: near-equivalence is the correct description for *isotropic residual
directions* — the model's local response machinery at a given step treats them almost
exchangeably (spread <=2x, no memory). The strong non-equivalence of the prompt
directions comes from their *structured, token-aligned geometry* interacting with the
prompt, not from perturbations being magically different in general. (The state-data
vectors themselves are unknown to us — torch-generated; we treat them as exchangeable
Gaussians, which only strengthens this conclusion.)

---

## 8. Rank-1-ness, the propagation front, and the margin response (Fig. 8, `numbers_t8.json`)

- Pairwise correlation of log||dh|| curves across directions: **0.89-0.92 at every
  radius** (this is the honest version of the earlier 0.94-0.98 claim; PC1 with gain
  kept is 0.82-0.86 at layer 27). After removing each direction's gain, PC1 is only
  0.24-0.26 — the residual 75% is the reproducible direction-specific structure of
  section 3.
- **The margin response is a different story**: the token-profiles of |dm| correlate
  0.35-0.38 across directions (shared near-tie structure), but the *signed* dm curves
  correlate at only **0.04-0.09 for every well-resolved radius (r>=0.003) — essentially
  uncorrelated**. (At r=0.001 the apparent 0.20 is float32 quantization noise; dm has up
  to 3% exact zeros there.) The decision-relevant response is direction-specific even
  where the norm response looks common. (Norm gain and |dm| gain do track each other: Spearman 0.86 at r=0.1,
  0.90 at r=1 — it is the sign and fine structure that differ.)
- Propagation timing is fully direction-independent: at r=0.1, layers 1/2/3 respond
  only on token prefixes 0-91 / 0-190 / 0-289, *identical for all 20 directions and all
  radii* (and layer 0 exactly zero at generated positions). The causal front is a model
  property, and gains were computed on the responsive subsets accordingly.

---

## 9. Clustering done right: no discrete clusters — stable outliers and a hot tail
(Fig. 3, `numbers_t3.json`)

- Correlation-distance (1 - Pearson on z-scored log-curves, r=0.3, layer 27), average
  linkage: silhouette 0.31 at k=2, rising to 0.49 at k=8 (isolating extreme points).
  Against an explicit null (common shape + measured replicate noise, 400 simulated
  datasets): the observed k=2 silhouette is **above the entire null distribution**
  (null median 0.12) — shape structure exists — but the "clusters" the k=2 cut finds are
  outlier pairs, e.g. {dirs 2,5} in 6 of 10 conditions (36% of condition pairs give
  literally identical partitions, ARI=1).
- Gaussian mixtures (diagonal EM, implemented in numpy; no sklearn available) with
  parametric bootstrap under the one-Gaussian null:
  - on curve PCs: best k unstable across runs (3 or 4), bootstrap **P(k>1) = 0.18**;
  - on a 6-feature direction fingerprint (log gain, 3 shape PCs, large-r alpha, sign
    bias): BIC prefers k=3 (148.0 vs 155.9 for k=1) but bootstrap **P(k>1) = 0.10** —
    not significant;
  - on log gain alone: **k=1 wins** (BIC 2.1 vs 4.0; P(k>1) = 0.14). With n=20 the hot
    trio is a heavy tail, not a resolvable second mode.
- Verdict (honest): **k=1 is the statistically defensible answer**; what the data
  actually shows is a continuum with reproducible extreme members — the hot trio
  {10,11,12} in gain (present in all three continuations) and the pair {2,5} in shape.
  If the project wants clusters, it needs more directions, not a different algorithm.

---

## 10. What is robust, what is genuinely different, what is not

**Robust (survives every check, favors the old conclusion):**
- the shared response *backbone*: token-profile of ||dh|| (corr 0.89-0.92), propagation
  front (identical timing), the identity of the near-tie bottleneck tokens (90-95% of
  directions bottleneck at the same 6 tokens; t=123 dominates), the |dm| profile
  (corr 0.35-0.38), and near-tie-driven clustering of *where* flips happen;
- near-equivalence of random current-state directions (spread <=2.1x, no memory across
  injection steps).

**Genuinely different across directions (falsifies equivalence):**
- gain: 7.3x spread even at r=0.001; 11.8x at r=0.1; stable per-direction across three
  continuations (rho 0.80-0.98) — a property of the direction's token axis;
- gain *rank* is not one scalar: rho 0.34 (layer 1 vs 27), 0.48-0.51 (linear vs
  saturated radii);
- shape beyond gain: 6% of log-variance (RMS 0.28 log-units, i.e. +-30% modulations),
  split-half reliability 0.999, significant for 20/20 directions;
- large-r exponent: 0.96..1.63 across directions (single-gain model predicts equality);
- sign of the margin response: per-direction dm<0 fractions 0.29..0.69 (16/20 beyond
  2-sigma), pairwise signed-dm correlation ~0.05 (uncorrelated), opposite signs at the
  same near-tie token;
- flip thresholds: 3 vs 16 of 20 directions flip between r=0.5 and r=2; response-part
  variance 2.6x the tie-part variance at r=1;
- hidden-state response vectors: not a single error mode (pairwise cos 0.13-0.46 at
  r=0.1, ~0 at r=2).

**Not supported / fails:**
- discrete direction clusters: silhouettes are outlier-driven, GMM k=1 wins under
  bootstrap (P(k>1) = 0.10-0.18), the 1-D gain distribution is unimodal at n=20;
- "some directions early, others late" as a global axis: position-correlations are
  significant per direction but with mixed signs; the shape differences are
  high-dimensional fingerprints, not one interpretable axis;
- simple geometric predictors of gain (L1/L2, dominant coordinate): impossible by
  construction — all such properties are identical across the simplex directions;
- the stored `cos` array as an independent rotation measurement: it is derived from the
  norm ratio (corr 0.994 with the transverse formula) and should be dropped from future
  analyses (or relabeled).

**Practical caveats:** dm is float32-quantized at r<=0.003 (up to 3% exact zeros) — sign
analyses use r>=0.1; r_c is a magnitude threshold (sign-blind) — flip prediction needs
the sign channel added; dh vectors are stored for only 3 directions (response-geometry
claims are limited to those pairs); the state-data vectors are torch-seeded and were
treated as unknown exchangeable Gaussians (a conservative choice for our conclusions).

**Best figures:** `fig5_flips.png` (the practically important non-equivalence),
`fig1_gain_consistency.png` (gain spread + cross-continuation stability),
`fig4_geometry.png` panel d (gain by token axis, three continuations),
`fig2_shape.png` (shape beyond gain), `fig7_state.png` (the contrasting
near-equivalence of state injections), `fig8_extra.png` panel a (the whole equivalence
hierarchy in one plot: norms vs |dm| vs signed dm).

---

## Summary for the main agent

We re-analyzed the 20-direction prompt-perturbation data trying to overturn the
"directions are equivalent" conclusion, and the overturn succeeds in the senses that
matter, while being clear about what stays equivalent. The old numbers (PC1 94-97%,
pairwise 0.94-0.98) describe the shared *token-profile* of the norm response and are
correct as far as they go — but they answer the wrong question. Even in the linear regime
the directions' gains differ by 7-12x, the gain rank is only moderately stable across
layers (rho 0.34 far pairs) and reshuffles between linear and saturated radii (0.5), the
large-radius exponent varies 0.96..1.63, and after removing each direction's gain the
remaining shape differences (6% of variance) are real with split-half reliability 0.999.
Most importantly, the signed margin response is essentially uncorrelated across directions
(r ~ 0.05) and 16/20 directions carry a significant sign bias, so at the same near-tie
token different directions push the decision in opposite directions; observed free-run
flips are accordingly direction-stratified (3/20 directions flip at r=0.5 — exactly the
persistent high-gain trio aligned with token axes 843/73349/54373 — 8/20 at r=1, 16/20
at r=2, with zero prediction misses), and the variance decomposition shows this is
driven ~2.6x more by direction-specific response strength than by near-tie luck. Gain is
a stable property of the direction's token axis across three different continuations of
the same prompt (rho 0.8-0.98), so "which token axis" is a genuine, geometry-level
predictor. What remains equivalent: propagation timing (identical causal front for all
directions), the location of the vulnerable tokens, and — crucially — random Gaussian
current-state injections, which are near-equivalent (<=2.1x gain spread, no memory across
injection steps); the non-equivalence is specific to structured token-subspace
perturbations. We also found two data quirks worth fixing upstream: the stored `cos`
array is a derived norm-ratio (not an independent measurement), and `r_c` is sign-blind
(|dm| based), which explains its flip false-alarms; adding the dm sign channel would
sharpen flip prediction. Net recommendation: report responses as "shared backbone +
direction-specific gain, exponent, shape, and margin sign", treat the hot trio
{10,11,12} as first-class objects, and if more resolution on direction identity is
wanted, increase the number of directions rather than re-clustering these 20.
