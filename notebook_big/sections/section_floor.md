# Part floor. The FP32 rounding floor, layer by layer

## Summary

Running the same 400-token sequence twice in 32-bit floating point — once clean and once with a tiny perturbation added to the prompt embeddings — does not return identical numbers, because the arithmetic rounds differently; the difference is the *floor*. Measured at five depths it is exactly zero at the embedding output (the perturbation never touches the embeddings of generated tokens), then 2.17e-4 at state 12, 3.44e-4 at 24, 9.52e-4 at 36 and 2.73e-3 at 47 (the last state) — a factor of 12.6 across the measured depth. The floor is not a property of the perturbation: it is a fixed fraction of the state's own norm, about 2.6e-6 (21–25 units of fp32 precision), and the state norm itself grows from 83 to 1062 between states 12 and 47, which is the whole of the depth dependence. At the campaign radius $r=0.1$ the real response is 31, 22, 19 and 17.5 times the floor at states 12, 24, 36, 47, and at least 7.0 times at the worst single step of the deepest state, so nothing in the campaign is floor-dominated; the smallest radius at which that remains true is 3.2e-3 at state 12 and 5.7e-3 at state 47.

---

## 1. What was measured

### 1.1 Setup

| item | value |
|---|---|
| model | `DeepSeek-R1-Distill-Qwen-14B`, 48 blocks, $d=5120$ |
| precision | FP32 (TF32 off), no autocast |
| prompt | `interstellar_travel`, $P=65$ tokens |
| continuation | 400 greedy tokens of the baseline, **teacher-forced** in every perturbed pass |
| perturbation | one unit direction $u$ (8 directions) times radius $r$ (9 radii, $10^{-4}\dots1$), added to the prompt embeddings only |
| saved states | $l\in\{0,12,24,36,47\}$; 0 = embedding output, $l$ = output of block $l-1$, 47 = input to the final norm |
| samples per cell | 8 directions × 300 steps = 2400 values, all medians below over $u$ and $t\in[100,400)$ |
| file | `results_big/ds14/calib/calib.npz` |
| baseline state norms | `results_big/ds14/a1_inter_full/a1.npz`, field `norm_h0` — same model, **identical `prompt_ids`** (checked with `array_equal`), 300 not 400 tokens |

The calibration npz stores `dh_norm`, `dm`, `r_star`, `margins` and the token ids; it stores **no baseline state norms**. Every statement below that involves a norm uses the `norm_h0` array of the A1 run, which is the same model on the same prompt with the same saved state indices. The two runs differ in generated length (400 vs 300), which matters only if the norm drifts with $t$; it does not: the median norm of state 47 lies between 1.02e3 and 1.10e3 in each of the six 50-step blocks.

### 1.2 Quantities

| symbol | explicit expression | meaning | why this quantity |
|---|---|---|---|
| $\delta h_t^l(u)$ | $h_t^l(u)-h_t^l(0)$, stored as `dh_norm` $=\lVert\delta h_t^l(u)\rVert$ | displacement of the state at depth $l$, step $t$, under direction $u$ | the object the perturbation experiment actually reads out |
| $M(r,l)$ | $\mathrm{median}_{u,\,t\ge100}\lVert\delta h^l_t(u)\rVert$ | response at radius $r$ and depth $l$ | one number per radius per layer; medians because the per-step distribution has a long right tail |
| $F_l$ | $M(10^{-4},l)$ | **the floor**: the response left when the perturbation is so small that the linear term is negligible | the measured quantity of this Part; the smallest radius is by construction floor-dominated |
| plateau value | $\mathrm{median}\{M(10^{-4},l),M(3\times10^{-4},l)\}$ | floor estimated from two radii instead of one | a robustness check on $F_l$, not a different quantity |
| $G_l$ | $M(0.1,l)/0.1$ | gain: state units of response per unit radius | the linear-response coefficient; $G_l r$ is the signal the perturbation produces |
| $n_l$ | $\mathrm{median}_{t\ge100}\lVert h_t^l(0)\rVert$ | norm of the unperturbed state at depth $l$ | the natural scale that rounding errors live on |
| $r_{\min}(l)$ | $F_l/G_l$ | radius at which $G_l r=F_l$, i.e. the arithmetic term equals the linear term | the smallest radius at which the response is still half signal, half arithmetic |
| $r_{3F}(l)$ | $3F_l/G_l$ | radius at which the linear term is 3× the floor | the radius below which the floor inflates the measured response by more than 5.4% |

### 1.3 The plateau check, done before the floor was used

$F_l$ is only a floor if $M$ has actually stopped moving with $r$. It has:

- $M(3\times10^{-4})/M(10^{-4})=1.008,\ 1.003,\ 1.001,\ 1.008$ at states 12, 24, 36, 47 — agreement to **0.8%**, far inside the 20% that would make the plateau claim doubtful, so both radii sit on the plateau.
- $M(10^{-3})/M(10^{-4})=1.069,\ 1.042,\ 1.029,\ 1.035$ — still within 7%.
- The floor is the same for every direction: at $r=10^{-4}$ the largest per-direction median is at most 1.023× the smallest, at every state. Rounding noise does not know which direction was perturbed.
- The floor does drift a little with $t$ at the deepest state: the median over $t\in[250,400)$ is 1.20× the median over $t\in[100,250)$ at state 47 (1.01, 1.03, 1.08 at states 12, 24, 36). This is the one place where a single median hides structure; §4 checks the ratio step by step because of it.

The exact zero at state 0 is a useful anchor rather than a nuisance: the perturbation changes only the 65 prompt embeddings, the generated tokens are the same tokens, so at the embedding output the two passes are bit-identical. The floor is produced by the 48 blocks of arithmetic, not by the input.

## 2. `ds14floor_perlayer.png` — the floor per layer

![floor per layer](figures/ds14floor_perlayer.png)

**(a) Response against radius.** $x$: $r$ on a log scale, $10^{-4}\dots1$. $y$: $M(r,l)$ on a log scale, one curve per non-zero saved state (four curves, light blue for the shallow state, dark blue for the deep one), with markers at the nine measured radii. The shaded band marks $r\le3\times10^{-4}$, the plateau; the red dotted vertical line is the campaign radius $r=0.1$; the grey dashed segment in the empty lower-right corner is a slope-1 reference (on log-log, a linear response is a straight line of slope 1).

Numbers: $M$ is tabulated below. At the two smallest radii all four curves are flat to within 1%; the local log-log slope over $10^{-4}\dots10^{-3}$ is 0.01–0.03, i.e. the curves do not move. Over $r=10^{-2}\dots3\times10^{-1}$ the same slopes are 0.99, 0.97, 0.96, 0.95 — the response is proportional to $r$ to within 5%. At the top, $r=0.3\dots1$, the slopes are 1.00, 1.05, 1.13, 1.14: the deepest states grow faster than linearly, which is the model's own nonlinearity already noted in Part 0, not floor.

**(b) Floor against depth.** $x$: saved state index $l$; $y$: $F_l$ on a log scale. Red discs are $F_l$ from $r=10^{-4}$; open squares are the two-radius plateau value (they coincide); the blue line is $2.61\times10^{-6}\,n_l$ using the nine `norm_h0` layers of the A1 run, drawn at all nine saved indices — the four measured points lie on it, the other five are predictions. $F_l$ runs 2.17e-4, 3.44e-4, 9.52e-4, 2.73e-3 at states 12, 24, 36, 47; state 0 is annotated as exactly 0 because a log axis cannot show it.

The full table, every number in the figure:

| $r$ \ $l$ | 0 | 12 | 24 | 36 | 47 |
|---|---:|---:|---:|---:|---:|
| 1e-4 | 0 | 2.169e-4 | 3.436e-4 | 9.523e-4 | 2.729e-3 |
| 3e-4 | 0 | 2.187e-4 | 3.446e-4 | 9.535e-4 | 2.749e-3 |
| 1e-3 | 0 | 2.320e-4 | 3.582e-4 | 9.795e-4 | 2.824e-3 |
| 3e-3 | 0 | 3.061e-4 | 4.267e-4 | 1.134e-3 | 3.261e-3 |
| 1e-2 | 0 | 7.049e-4 | 8.425e-4 | 2.068e-3 | 5.627e-3 |
| 3e-2 | 0 | 2.023e-3 | 2.314e-3 | 5.522e-3 | 1.460e-2 |
| 1e-1 | 0 | 6.720e-3 | 7.587e-3 | 1.809e-2 | 4.774e-2 |
| 3e-1 | 0 | 2.024e-2 | 2.253e-2 | 5.394e-2 | 1.434e-1 |
| 1 | 0 | 6.708e-2 | 7.976e-2 | 2.091e-1 | 5.636e-1 |

(Medians; the means are 1–5% higher, e.g. 2.856e-3 vs 2.729e-3 at state 47, because of the right tail.)

**What the figure does not show.** Only five states were saved, so between them the curves are straight lines in the drawing, not measurements; the blue norm law in (b) at states 6, 18, 30, 42 is a prediction. The medians erase a per-step variation of up to 8× at fixed $r$ (state 47, $r=0.1$: the per-step median over directions runs from 2.04e-2 to 1.63e-1). And a log axis hides the exact zero at state 0, which is why it is written out.

## 3. `ds14floor_law.png` — what the depth dependence is

![floor law](figures/ds14floor_law.png)

**(c) Floor against state norm.** $x$: $n_l$, $y$: $F_l$, both log. Four points, labelled by state. Solid line: the free power law $F\propto n^{0.977}$; dashed: slope 1. The exponent is 0.977 with an rms log residual of 0.074. The ratio $F_l/n_l$ is 2.614e-6, 3.019e-6, 3.027e-6, 2.568e-6 at states 12, 24, 36, 47 — a factor 1.18 total spread while $n_l$ changes by a factor 12.8. In units of fp32 machine epsilon (1.19e-7) that is 22, 25, 25 and 22 units.

**(d) Which depth law fits.** Horizontal bars: rms of $\ln(\text{predicted}/\text{measured})$ over the four layers, for six candidate laws; the text gives the rms and the largest single-layer deviation. The 10% dotted line is a visual reference, not a test. Results:

| law | rms | worst point (log dev.) | measured/predicted at $l=12,24,36,47$ | reading |
|---|---:|---:|---|---|
| $F\propto l^{1/2}$ | 93% | 128% | 0.28, 0.31, 0.70, 1.76 | rejected: predicts 1.98× growth from $l=12$ to 47, observed 12.58× |
| $F\propto l^{p}$ | 36% | 49% | 1.32, 0.61, 0.83, 1.49 | rejected: best $p=1.77$, residuals zigzag by up to 1.6× |
| $F\propto e^{cl}$ | 17% | 24% | 1.20, 0.79, 0.91, 1.16 | poor: one $e$-fold per 13.6 blocks, systematic S-shape |
| $F\propto n^{p}$ | 7.4% | 9.2% | 0.91, 1.06, 1.09, 0.95 | accepted, $p=0.977$ |
| $F\propto n$ | 10.4% | 14.8% | 1.00, 1.16, 1.16, 0.98 | accepted, $\kappa=2.610\times10^{-6}$ (21.9 ε) |
| $F\propto\sqrt{\sum_b n_b^2}$ | 7.2% | 11.0% | 1.02, 0.90, 1.09, 0.99 | accepted, $k_q=2.176\times10^{-6}$ (18.3 ε) |

**(e) The ratio step by step.** $x$: generated step $t\in[100,400)$; $y$: $\mathrm{median}_u\lVert\delta h^{l}_t\rVert\big/\mathrm{median}_u\lVert\delta h^{l}_t\text{ at }r{=}10^{-4}\rVert$ — the same ratio as in §4 but computed per step before taking any median over $t$, one curve per non-zero state. Solid grey line: 1 (signal equals floor); dotted grey: 3. Medians over $t$ are 30.4, 20.0, 18.0, 16.5 at states 12, 24, 36, 47; the 10th percentiles are 21.2, 14.9, 13.3, 10.6; the minima are 16.5 (t=313), 11.4 (t=293), 8.7 (t=294), 7.0 (t=293). No step out of 300 at any state falls below 7× the floor.

**What the figure does not show.** Four points cannot separate laws that differ by less than ~10%; the difference between $n^{0.98}$ and a quadrature sum of per-block errors is below that resolution (their rms values, 7.4% and 7.2%, are not significantly different). The freedom in (d) is 1–2 parameters per law fitted on the same four numbers that are then scored, so the residuals are optimistic. And the norm is borrowed from a second run.

## 4. Floor against signal — and the better comparison

At $r=0.1$ the median response and the floor give

| $l$ | signal $M(0.1)$ | $F_l$ | signal/floor | relative signal $M/n_l$ | relative floor $F_l/n_l$ | ratio of the two |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 6.720e-3 | 2.169e-4 | 30.98 | 8.10e-5 | 2.61e-6 | 30.98 |
| 24 | 7.587e-3 | 3.436e-4 | 22.08 | 6.66e-5 | 3.02e-6 | 22.08 |
| 36 | 1.809e-2 | 9.523e-4 | 19.00 | 5.75e-5 | 3.03e-6 | 18.99 |
| 47 | 4.774e-2 | 2.729e-3 | 17.49 | 4.49e-5 | 2.57e-6 | 17.49 |

**Where measurement is weakest: the deepest state, and the reason is the signal, not the floor.** The floor relative to the state's own norm is the same at every layer (2.6–3.0e-6), so normalizing by the norm changes the ratio by at most 2% (last column). What falls with depth is the *relative* response: 8.1e-5 of the state norm per unit radius at state 12, 4.5e-5 at state 47. The deepest read-out is therefore the one closest to its floor, by a factor 1.8 — a real effect, but a small one.

The obvious comparison, signal at $r=0.1$ divided by the floor, is the right one for the campaign because $r=0.1$ is the radius actually used, but as a general metric it is misleading: it is proportional to $r$, so it says nothing about the model and everything about the chosen radius. The dimensionless version is $r/r_{\min}(l)$, which is the same number here (17.5 at the deepest state) and remains meaningful if the radius changes. Two more things the naive comparison hides, both checked:

1. **The floor grows with $t$ by up to 20%.** At state 47 the floor is 1.20× larger over $t\in[250,400)$ than over $[100,250)$. The per-step ratio in figure (e) absorbs this and still has a minimum of 7.0, so the median is not hiding a failure, but it does mean a single median-based floor is accurate only to ~20% at the deepest state.
2. **The floor and the signal add in quadrature, not linearly.** With no free parameter — $F_l$ from $r=10^{-4}$, $G_l$ from $r=0.1$ — the model $\lVert\delta h\rVert(r)=\sqrt{F_l^2+(G_lr)^2}$ reproduces all 32 measured points with $r\le0.3$ to an rms of **1.71%** (worst 5.83%, at $r=3\times10^{-3}$, state 47). Adding them linearly, $\lVert\delta h\rVert=F_l+G_lr$, is off by an rms of 14.1% and up to 26.9% in the same range. This is the strongest single check that $F_l$ is an independent noise term rather than a rescaling of the response: it predicts the whole transition region from the two endpoints. At $r=1$ the measured response is up to 18% above the model, which is the model's own nonlinearity, not floor.

## 5. How the floor accumulates: what the data support

Four accumulation models and their predictions:

- **Independent rounding per block, equal size, adding in quadrature:** $F_l=\delta\sqrt{l_{\rm block}}$ → $F\propto l^{1/2}$. Predicts a 1.98× rise from state 12 to 47; measured 12.58×. Rejected by a factor 6 in growth.
- **Multiplicative growth:** each block multiplies the error by a constant → $F\propto e^{cl}$. Fitted $c=0.0734$ per block (one $e$-fold per 13.6 blocks) with 17% rms residuals and a systematic S-shape: the prediction is too low at 12 and 47, too high at 24 and 36. Not supported.
- **Free power of depth:** $F\propto l^{1.77}$, 36% rms. Not supported.
- **Rounding error proportional to the local state norm, accumulating in quadrature:** $F_l=k_q\sqrt{\sum_{b\le l}n_b^2}$. Because $n_b$ grows by a factor 12.8 over the measured range, this sum is dominated by the last few blocks and is numerically almost proportional to $n_l$ itself. Fitted with one constant, $k_q=2.18\times10^{-6}$ (18.3 ε), it gives 7.2% rms and 11% worst-case residuals.

**The data support the last one, and its practical content is "$F_l$ is a fixed fraction of the state norm".** The single-constant version $F_l=2.61\times10^{-6}n_l$ (21.9 ε) matches to 10% rms. Physically: every block's arithmetic leaves a relative error of order 20 fp32 units on the state it writes, and the visible floor is the quadrature sum of those errors weighted by the (growing) state norm. With four measured layers the two versions — proportional to the current norm, or quadrature over per-block norms — cannot be separated beyond the ~10% residual level; both are consistent, and both say the floor is set by the *size of the numbers being rounded*, not by the number of blocks alone.

That distinction matters for extrapolation. "Floor grows like $\sqrt{\text{depth}}$" would be reassuring; it is wrong here because the residual stream grows with depth. The measured statement is: **absolute floor grows roughly as fast as the state norm grows**, which for this model is a factor 12.6 between states 12 and 47, i.e. 7.5% per block on average.

## 6. The smallest usable radius per layer

Solving $G_lr=F_l$:

| $l$ | $G_l$ | $F_l$ | $r_{\min}=F_l/G_l$ | $r_{3F}=3F_l/G_l$ |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | undefined (no floor, no gain) | undefined |
| 12 | 6.720e-2 | 2.169e-4 | 3.23e-3 | 9.68e-3 |
| 24 | 7.587e-2 | 3.436e-4 | 4.53e-3 | 1.36e-2 |
| 36 | 1.809e-1 | 9.523e-4 | 5.26e-3 | 1.58e-2 |
| 47 | 4.774e-1 | 2.729e-3 | 5.72e-3 | 1.71e-2 |

Reading: at $r=r_{\min}$ the measured displacement is $\sqrt2$ times the linear term (the two contributions are equal and add in quadrature), so half of the squared response is arithmetic. At $r_{3F}$ the measured displacement is only 5.4% above what a floor-free linear response would give, and the floor's share of the squared response is 10%. The campaign radius 0.1 sits 17.5–31 times above $r_{\min}$, and 5.8–10.3 times above $r_{3F}$ — inside the safe region with room to spare, but not by orders of magnitude at the deepest state. A 10× margin over $r_{\min}$ would require $r\ge0.057$ at state 47, which is close to the campaign radius; the choice $r=0.1$ is therefore near the small end of what this 48-layer fp32 network can resolve at its last state, and a deeper model would need a larger radius for the same margin.

This is also consistent with the "clean window" of Part 0 ($r\in[0.03,0.3]$), which was defined by a different criterion — the radius range where $M(r)/r$ is constant to a few percent. $r_{\min}$ is the absolute validity edge (floor equals signal); 0.03 is where the floor inflates $M(r)/r$ by only ~2% (at state 47: $M(0.03)/0.03=0.487$ against the asymptotic $G=0.477$). Both numbers describe the same transition from opposite sides.

## 7. What this does not show

- Only four non-zero states were saved (12, 24, 36, 47 of 48 blocks). The shape of $F_l$ between them is interpolation; in particular the shallowest measured state is already 12 blocks deep, so nothing here says how fast the floor rises over the first quarter of the network.
- One model, one prompt, one precision, one random seed for the directions. The constant 2.6e-6 is a property of this network in fp32; a different architecture, norm placement, or dtype would change it. bf16 keeps 8 mantissa bits against fp32's 24, so the per-operation error — and with it the floor at a given state norm — would be roughly $2^{16}\approx6\times10^4$ times larger; fp64 would lower it by about $2^{29}\approx5\times10^8$.
- The floor was measured at generated positions only, because that is where `dh_norm` is stored. Nothing is claimed about prompt positions, where the perturbation itself lives.
- The 20% drift of the floor with $t$ at the deepest state is real and the medians hide it; §4's per-step check bounds its effect on the campaign but not on other analyses that use a single floor number.
- $r_{\min}$ uses $G_l$ measured at $r=0.1$; if the response were nonlinear already at $r=0.03$, $r_{\min}$ would move. The linearity check in §2 (slope 0.95–0.99 over $10^{-2}\dots3\times10^{-1}$) says it does not, to 5%.

## 8. What it implies for perturbation experiments on deep models

The floor is not an inconvenience to be subtracted; it is a *scale* of the system "network + arithmetic", and it is set by the norm of the state being written at each depth. Three practical consequences, each backed by a number above:

1. **Radii must be chosen per depth, not per model.** The deepest read-out has the largest floor (2.73e-3) and the smallest relative response, so it sets the floor of the usable radius: 5.7e-3 here. The minimum that works at state 12 (3.2e-3) is below the one at state 47 (5.7e-3), so a radius chosen from the shallow state alone would be floor-contaminated at the deep one.
2. **The margin is measurable in advance.** Both $F_l$ and $G_l$ come from the same cheap calibration run at two radii ($10^{-4}$ and 0.1), so $r_{\min}$ is known before the campaign, per layer, without any assumption about the model's dynamics.
3. **The wall exists but is far.** For this 48-block fp32 network the floor is 5.7% of the measured $r=0.1$ response at the last state (3.2% at state 12), and the response is at least 7× the floor at every one of the 300 steps, 17.5–31× in the per-state medians. A model whose state norm grows another order of magnitude at fixed gain would put $r_{\min}$ near $5\times10^{-2}$ and force the campaign radius up. The measurement here is the calibration that would detect that before it silently corrupts a result.
