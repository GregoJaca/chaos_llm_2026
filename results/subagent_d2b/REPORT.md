# D2 shape investigation: why C(eps) curves look "good" vs "weird"

**Subagent D2b report — 18 Nov 2026 — CPU-only, block-chunked, float64 reference arithmetic.**
Scripts in this folder reproduce everything: `check_data.py`, `exp_shape1.py` (figs 1-4), `exp_shape2/3/4.py` (figs 5-7), `crossing_numbers.py`, `diag_surrogates.py`; JSONs `check_data.json`, `shape1/2/3/4.json`, `numbers_shape.json`. `pt_load.py`/`common.py` copied from `subagent_d2` and reused. Read `subagent_d2/REPORT.md` first; this report re-examines the **shape** question only.

---

## Verdict up front

**The user's memory is real and now fully explained.** There are (at least) five distinct C(eps) shapes obtainable from this same data by innocent pipeline choices. Four of them look "good" or "textbook" and none of the five is an attractor dimension — and the "weird" sigmoid we published for the non-loop trajectories is the honest shape of that data:

| # | Curve shape | Where it comes from | Looks like | Means |
|---|---|---|---|---|
| A | steep sigmoid, whole rise inside 0.16-0.19 decades, slope monotone 9 -> 1.5 | interstellar, launch raw 1536-D cloud (any metric, any centering, any N) | "weird" | honest: an effectively ~10-20-dim inhomogeneous cloud with distance concentration (IQR/median = 0.20-0.24). **No scaling region exists, or is even geometrically possible** |
| B | knee at the repeat scale, then a quasi-linear mid-range | cyber/future/great (repetition loops, P=53/74/69) | "good"; fit r2 up to 0.999 | the loop tube + phase ring of repeated text, not an attractor |
| C | flat floor at all eps, then rise | first layer (352,035 exact-duplicate pairs = 4.2% of all pairs; 36 states repeated up to 305x) | "good-ish" if fit carelessly | duplicates; small-eps fits give D2 = 0.02 (r2 = 0.12) |
| D | textbook straight line, slope ~1.95 over ~1 decade, r2 = 0.999 | ANY scalar (norm, PC1, step-dist) delay-embedded with **m=2** | "good" | the fill dimension of a 2-D cloud: white noise at m=2 gives the identical curve (0.96 dec @ 1.96); all surrogates including the time-shuffle reproduce it |
| E | partial flattening, slope 2.2-2.5 that stops growing with m | launch PC1 embedded at m >= 4 | "saturating D2 ~ 2.3!" | effective rank of a red, heavy-tailed marginal's embedding covariance: IAAFT nulls land at 1.7-2.4 with the same pattern |

Reference curves generated with the identical detector at the identical N=3031: **Henon** flat over 2.57 decades, D2 = 1.157 (r2 = 0.9998); **Lorenz** (x[::4], m=8, tau=4) flat over 0.96 decades, D2 = 1.984 (r2 = 1.000). So the machinery, grid, and plateau detector do find real scaling regions when they exist.

---

## Ten bullets (all numbers exact, from the JSONs in this folder)

1. **Centering is a red herring — provably.** Euclidean C(eps) is translation-invariant: centered vs uncentered gives max |dlog C| = **0.00** on interstellar and launch (float64), and the previous fp32 uncentered pipeline differs from the float64-centered reference by at most **1.7e-06** (launch). Anisotropy is too small (||mu||/scale = 0.37, 0.46, 0.82 for interstellar/launch/cyber) for fp32 cancellation to matter (fp32 rel. error at the median distance: 1e-08-7e-08). If the user's notebook showed different shapes for "centered vs uncentered", it must have been a different distance (cosine) or a different axis/grid, not the centering itself.

2. **Absolute vs normalized eps axis: identical shape, only x-labels move.** The cloud scales are 127.1 (interstellar), 122.9 (launch), 90.1 (cyber), **1.68 (first layer)** — a notebook with a fixed absolute eps grid (or fixed plot range) gets comparable curves across last-layer prompts but misses the first layer by ~70x. This alone can flip a curve from "sits nicely in the axes" to "stairs at the right edge".

3. **N = 512/1024/2048/full only moves the depletion edge.** All N-curves collapse onto one another above C ~ 10/N; below it they truncate (fig 1, with 1/N and 10/N lines per N). Nothing about the mid-range shape changes with N — so a "good-looking" curve was never an N effect.

4. **The non-loop raw-cloud curves have no room for a scaling region — quantitatively.** The entire rise C = 0.1 -> 0.9 spans **0.156 decades** (interstellar) and **0.195 decades** (launch) of eps; slope at C = 0.1 is **8.98 / 9.33** vs the Eckmann-Ruelle ceiling 2 log10(N) = **6.21 / 6.97**; the region where slope <= ceiling AND C in [10/N, 0.5] is **1 grid point (0.00 decades)** for interstellar and **0.039 decades** for launch. A best-fit search over all contiguous >= 1-decade windows avoiding depletion (C >= 10/N) and saturation (C <= 0.9) returns **None** on both. This is the concentration-of-measure signature: q10/med = 0.80, q90/med = 1.19.

5. **The metric/preprocessing variants do NOT rescue the shape — they steepen it.** On launch: Euclid IQR/med = 0.243; per-coordinate z-scoring contracts it to **0.126** (and boosts the small-eps slope range to 10.8); cosine on raw states 0.246; cosine on centered states **0.122**. Every variant is a steep sigmoid (fig 2); none has a >= 1-decade flat slope anywhere below the ceiling. A notebook that z-scored coordinates or used cosine saw *more* step-like curves than ours, not more textbook ones.

6. **The classic artifacts, made explicit (fig 3).** (a) Saturation: C -> 1 at eps ~ 1.5-1.9 x median. (b) Depletion: below C ~ 10/N the Poisson 2-sigma band explodes (launch). (c) The naive `cdist` convention (include i=j) fabricates a flat floor at 1/N with slope -> 0 — a "plateau" made of the diagonal. (d) A **linear** eps grid over the same data turns the small-eps end into a coarse staircase on log-log axes — grid choice alone changes the apparent shape. (e) First layer: the duplicate floor is **C = 0.042 at every eps** (352,035 exact zero-distance pairs, 4.2% of all pairs — 36 distinct states with multiplicities up to 305, audited exactly in `dup_audit.py`; the naive `eps -> 0` "zero" counts under-report it because fp64 Gram cancellation leaves half the duplicate pairs at ~1e-15 instead of 0); any small-eps fit measures duplicates (D2 = 0.02).

7. **Theiler windows never reshape the raw-cloud C(eps) — and the one exception proves the rule (fig 4).** Launch raw: w = 0/10/50/100 changes C by at most **0.10** in log (deep in the count-noise zone). Cyber raw: w <= 50 changes <= 2.2%; w = 100 (= 2 periods, P=53) removes the freshest loop-copy pairs and changes C by up to **5.8 in log — but only at eps ~ 0.011 x median where C ~ 4e-5**; for C > 1e-2 the effect is **0.65%**. Temporal correlations enter a point-cloud statistic *only through repeated points*; no repeats -> no Theiler effect. (On delay embeddings the picture is the same at these ACFs: max |dlog C| = 0.067 for launch PC1 m=5 tau=1; 1.60 for cyber's looped PC1.)

8. **The delay-embedding (notebook) route produces genuinely "good-looking" curves that are all traps (figs 5, 7).** At **m = 2** every observable gives a 0.86-0.96-decade plateau at slope 1.92-1.96 with r2 up to 0.9999 (e.g. launch norm m=2 tau=5: 0.96 dec @ 1.95; PC1 m=2: 0.86 dec @ 1.92) — and white noise at m=2 gives **the same curve** (0.96 dec @ 1.96); phase-rand (x5), IAAFT (x3) AND the time-shuffle all reproduce it (max |dlog C| <= 0.17); D2(N) even "converges": 1.89 (N=768) -> 1.92 (1536) -> 1.95 (3036) — converging to the fill dimension 2. At m >= 3 the flat regions collapse to <= 0.6 decades and the slope rides the noise diagonal (launch norm 1.84 -> 5.10, step-dist 1.85 -> 5.66 as m: 2 -> 10).

9. **The one curve that looks like saturation (launch PC1, m >= 4, slope 2.2-2.5) is matched by IAAFT nulls.** Data m-sweep (tau=1): 1.79, 2.24, 2.49, 2.50, 2.47, 2.25, 2.26; IAAFT nulls (marginal+spectrum): 1.72-1.74, 2.18-2.21, 2.17-2.46, 2.35-2.41, 1.99-2.16, 1.79-1.86, 2.09-2.37 — same flattening; phase-rand (spectrum only, Gaussianized marginal) rides the diagonal up to ~5.3. At m=8 the mid-range C matches IAAFT to **0.99-1.08**; the small-eps differences are marginal effects (at eps = 0.2 x med: data C = 3.5e-3 vs phase-rand ~1.5e-5 — a 170x gap from Gaussianizing a heavy-tailed marginal — vs IAAFT 1.3-1.5x, vs time-shuffle 0.15x). **No plateau on the data or on any surrogate** (data flat region 0.25 dec; IAAFT 0.30; phase 0.10; shuffle 0.40). This is the only place where our C(eps) is thinner than a second-order null — consistent with the mild recurrence noted in subagent_d2's report, and it is statics, not a scaling region.

10. **The D2(m) saturation test (fig 7) is the definitive negative for the scalar route.** References saturate: Lorenz slope-at-C=0.1 is flat in m (1.37 -> 1.62; plateau fit D2 = 1.984 at m=8), Henon 1.157 at m=2. Our trajectories: ||x_t|| and step-dist ride the y = m white-noise diagonal until the ER ceiling clamps them; PC1 partially flattens but the IAAFT band flattens with it. **Nothing separates from its null the way Lorenz separates from white noise.**

## Why the shapes differ between variants (the user's question, answered)

The dominant factor is **which trajectory**: loops (cyber/future/great) give the knee + beautiful mid-range fits (cyber best 1-decade fit D2 = 1.111, r2 = 0.9991 over eps = [0.057, 0.58] x median — the drifting loop tube; ring fit D2 = 3.24, r2 = 0.984 — the phase ring), while non-loop trajectories give the steep sigmoid. Second: **which layer** (first layer = duplicate floor). Third: **the notebook route itself** — m=2 delay embeddings always look textbook (fill dimension), m >= 8 looks "saturated" for red series (effective rank). Fourth: **preprocessing and grid conventions** (z-score/cosine steepen; linear grids staircase; absolute axes break across layers; cdist self-pairs add a fake floor). Centering and fp32 — the two things we checked first — are provably irrelevant here.

## Best figures

- **`fig3_artifacts.png`** — the four non-dynamical ways a C(eps) curve gets its shape (loop knee with the r2=0.999 tube fit annotated, duplicate floor, depletion zone with 1/N-10/N lines and Poisson band + cdist floor, linear-grid staircase).
- **`fig6_reference_shapes.png`** — side-by-side: Henon/Lorenz (real plateaus at the same N) vs launch PC1 vs its IAAFT/AR(1)/phase/shuffle nulls. This is the figure that shows what "good" actually looks like.
- **`fig7_saturation_test.png`** — D2(m) vs m: references saturate, our observables ride the noise diagonal or their IAAFT bands.
- **`fig1_N_axes.png`** — the identity checks (centering/fp32/axis) and N-dependence with depletion lines.
- **`fig5_delay_embedding.png`** — the m-sweep family + panel (d), the m=2 trap with all surrogates on top of the data.
- **`fig4_theiler.png`**, **`fig2_metric_variants.png`** — the Theiler no-op (with the loop exception located) and the metric variants.

## What would be needed to settle D2 definitively

1. **More, longer, non-loopy trajectories.** At T ~ 3k the ER-legal window on this data is 0.0-0.04 decades; dimensions up to ~7 are in principle measurable and none is found. Measuring D2 > 7 needs N > 10^(D2/2) ~ 10^5+ steps *of a genuinely non-repeating trajectory* — i.e. anti-repetition sampling (repetition penalty, temperature) at 10-100x longer horizons, or pooling many prompts.
2. **Regenerate the three loopy prompts with anti-repetition** and confirm their curves become the steep sigmoid (shape A) — that would close the loop on "the good fits were the loops".
3. **Recover the user's notebook and rerun it verbatim on these trajectories.** The two candidate explanations — loop data and m=2 embeddings — both reproduce "good-looking" shapes here; comparing their pipeline cell-by-cell would pin down which one they remember. The cosine/z-score/cdist/linear-grid variants are all documented in figs 2-3 if their notebook used any of them.
4. If a low-D claim is ever to be defended, it needs the **fig-7-style saturation test plus IAAFT nulls at several N**, not a single-window r2.

---

## One-paragraph summary for the main agent

The user is right that the shapes vary, and wrong only about where the variation comes from: centering/normalization/N are provably no-ops here (max |dlog C| = 0.0 centered-vs-uncentered; 1.7e-06 for the fp32 pipeline; N only moves the C < 10/N depletion edge), and the non-loop trajectories (interstellar, launch) have an intrinsically "weird" C(eps) — a steep sigmoid whose entire rise spans 0.16-0.19 decades with slope 9 -> 1.5, an ER-legal (slope <= 2 log10 N) window of at most 0.04 decades, and no 1-decade fit anywhere — because the raw cloud's pairwise distances are concentrated like a ~10-20-dim homogeneous cloud (IQR/median 0.20-0.24). The "good-looking" curves they remember are real objects, all reproduced here: (a) the repetition loops' tube+ring (cyber: 1-decade fit D2 = 1.11, r2 = 0.999, and ring fit D2 = 3.24, r2 = 0.98), (b) the first layer's duplicate floor (C = 0.042 at all eps), (c) m=2 delay embeddings, which give a textbook slope-1.95 over 0.96 decades with r2 = 0.999, D2(N) "converging" 1.89 -> 1.95 — and are matched exactly by white noise, phase-rand, IAAFT, and the time-shuffle (it is the fill dimension of any 2-D cloud), and (d) launch PC1's apparent "saturation" at 2.2-2.5, which IAAFT (marginal+spectrum) nulls reproduce at 1.7-2.4 — while references run through the identical detector at the same N cleanly (Henon D2 = 1.157 over 2.57 decades; Lorenz D2 = 1.984 over 0.96 decades, r2 = 1.000). Theiler windows change nothing on raw clouds except the loop-copy region at C < 1e-4 (the repeated-points mechanism), and the honest D2 statement for this data is: **no scaling region exists on any non-loop trajectory in any metric, centering, N, or embedding we tested; every previously quoted D2 is a window/tube/fill-dimension/duplicate artifact; settling D2 needs 10^5-step anti-repetition trajectories, and the m=2 + loop explanations for the notebook's "good" shapes can be confirmed by rerunning the user's notebook verbatim.**
