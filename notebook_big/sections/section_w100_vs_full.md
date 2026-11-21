# Part window regime. True window 100 against full attention on the 14B: the same two razor ties, a x2.2 flush, and a late response that does not decay

## Summary

The same prompt, seed and 1000 directions were run twice on `DeepSeek-R1-Distill-Qwen-14B` in FP32, once with full attention and once with a *true* sliding window of 100 tokens (the `layer_types` patch was verified by a bitwise departure test). The two runs are bit-identical up to generated step 34 and first differ at exactly $t=35=W-P$, as the window law requires; the greedy text itself changes at $t=45$. The same two razor ties survive: $m_{21}=9.727\times10^{-5}$ and $m_{19}=1.144\times10^{-3}$ nats in both runs, and 418 and 41 of the windowed flips are still controlled at $t=21$ and $t=19$. The window adds one new late tie at $t=297$ ($m=4.79\times10^{-3}$ nats), which raises the predicted flip fraction from 46.4% to 49.3% without removing a single full-attention flip. The last-layer response jumps by a factor 2.20 across $t=35$ (the 1.5B showed 1.73), the typical response gain rises from 0.58 to 1.49, and the late-time response decays to 0.58 of its early value under full attention but stays at 0.93 under the window. In other words the window does not cut the late response: it makes it larger and more bursty, because only one prompt position leaves the window per step and the perturbation survives in the generated-token states that stay inside it.

---

## 1. The runs, the quantities, and the verification

### 1.1 Setup

| item | full attention | true window 100 |
|---|---|---|
| model | `DeepSeek-R1-Distill-Qwen-14B` (48 blocks, hidden size 5120) | same |
| precision / decoding | FP32, TF32 off; greedy, fixed length | same |
| prompt | `interstellar_travel`, $P=65$ tokens | same |
| continuation | $T=300$ baseline greedy tokens | same, but **a different text from $t=45$** |
| directions | $K=1000$, 50 disjoint simplex blocks of 20, unit norm on the $65\times5120$ prompt embeddings | same construction, same seed |
| radius | $r=0.1$ only | same |
| attention | full: the request `window=100` was silently inactive (transformers gates it on `layer_types`, which the stock config does not set) | true: patched, window 100, verified |
| saved states | indices $[0,6,12,18,24,30,36,42,47]$ | same |
| file | `results_big/ds14/a1_inter_full/a1.npz` | `results_big/ds14/a1_inter_w100/a1.npz` |
| wall time | 1590.6 s | 1606.8 s |

Two auxiliary files are used for checks, not for the main comparison:

- `results_big/ds14/a1_inter_w4096/a1.npz` — 32 directions, 500 tokens, window 4096. Its first 300 baseline tokens are **identical** to the full-attention run's and its margins differ from them by at most $4.8\times10^{-5}$ nats (batch-shape round-off). It confirms that the full-attention run is a valid "no clipping within the horizon" reference.
- `results_big/ds14/calib/calib.npz` — 8 directions, 400 tokens, 9 radii, `meta.json` says `window: 100`. **It is not a true-window run.** Its first 300 baseline tokens are identical to the *full-attention* run's (not to the windowed run's) and its margins differ from the full run's by at most $7.2\times10^{-5}$ nats. The `window` field in these metadata files records the requested value, not the active mask (the full-attention meta also says `window: 100`). The calibration therefore establishes radius linearity for full attention only; it is used below with that caveat.

### 1.2 Data facts that must precede any number

1. **The window is real and starts at exactly $W-P$.** For $P=65$, $W=100$ the 100-key window of the response position $P+t$ is $P+t-99\dots P+t$, so the first prompt key (position 0) leaves at $t=100-65=35$. Measured: every saved state is bit-identical for $t\le34$ and differs from $t=35$ onward. This is the same departure test used for the 1.5B and for the other 14B window runs; the first departure is at exactly $t=W-P$.
2. **Margins lag the states by one step.** The engine stores the response at absolute position $P+t$ (`dh_norm[:, P:P+T]`) and the margin of the token predicted from position $P+t-1$. The first margin difference is therefore at $t=36$, one step after the first state difference. The first *token* difference is at $t=45$; the two baseline continuations diverge there (`" has"` versus `"'s"`), so every margin, threshold and response after $t=45$ refers to a different text in the two runs.
3. **The baseline scale is unchanged.** The median baseline residual norm at the last saved state over $t\in[150,300)$ is 1046.6 (full) and 1077.3 (window). The absolute response comparison is therefore not a normalisation artefact.

### 1.3 Quantities

Let $u$ index a direction and $t\in\{0,\dots,299\}$ a generated step. All thresholds use the signed rule, because only a *negative* change of the top-1/top-2 gap can close it.

| symbol | explicit expression | meaning | why it is used |
|---|---|---|---|
| $m_t$ | $\ell_t^{(1)}-\ell_t^{(2)}$ (nats) | gap between the two best tokens of that run's own baseline | a flip requires this gap to be closed |
| $\delta m_t(u)$ | $m_t(u)-m_t$ (nats, stored `dm`) | change of that gap under direction $u$ at $r=0.1$ | response of the decision variable without letting the text diverge |
| $r_c^-(t,u)$ | $m_t\,r/\max(-\delta m_t(u),0)$; $+\infty$ if $\delta m_t\ge0$ | radius at which step $t$ would flip | solve $-\delta m_t\ge m_t$ for the radius |
| $r_*(u)$ | $\min_t r_c^-(t,u)$ | threshold radius of the direction | flips at the run radius iff $r_*(u)\le0.1$ |
| $t_*(u)$ | $\arg\min_t r_c^-(t,u)$ | controlling step | which near-tie is the weak point |
| $\rho_t^{47}(u)$ | $\lVert\delta h_t^{47}(u)\rVert/r$ | absolute response of the last saved state (layer 47) per unit radius | the quantity that the window acts on |
| $G(u)$ | $\mathrm{median}_t\,\rho_t^{47}(u)$ | typical last-state gain of the direction | one number per direction, robust to single-step bursts |
| $\Phi$ | $\mathrm{median}_{t\in[36,45]}\rho_t^{47}\big/\mathrm{median}_{t\in[25,34]}\rho_t^{47}$ | flush ratio across the first clipped position | tests the 1.5B flush signature |
| $\Lambda$ | $\mathrm{median}_{t\in[150,300)}\rho_t^{47}\big/\mathrm{median}_{t\in[0,50)}\rho_t^{47}$ | late-time fraction | tests whether the window suppresses the late response |

The two windows in $\Phi$ and $\Lambda$ are pooled over directions and steps; medians over directions at each step, then over steps, give the same answers to within 5% and are reported in the JSON.

### 1.4 The comparison in one table

| quantity | full attention | true window 100 |
|---|---:|---:|
| smallest margin, step | $9.727\times10^{-5}$ (t=21) | $9.727\times10^{-5}$ (t=21) |
| second margin, step | $1.144\times10^{-3}$ (t=19) | $1.144\times10^{-3}$ (t=19) |
| median / mean margin over 300 steps (nats) | 3.096 / 5.00 | 1.826 / 2.92 |
| margins below 0.05 / 0.1 / 0.5 | 4 / 8 / 34 | 6 / 9 / 52 |
| predicted flips at $r=0.1$ | 464 (46.4%) | 493 (49.3%) |
| flips controlled at t=19 / 21 / 297 | 43 / 421 / 0 | 41 / 418 / 34 |
| flips that are new under the window | — | 29, all at t=297 |
| flush ratio $\Phi$ | 0.99 | 2.20 |
| late fraction $\Lambda$ | 0.58 | 0.93 |
| median gain $G$ | 0.583 | 1.493 |
| depth growth, state 1 -> state 8 | x8.4 | x38.6 |
| first differing baseline token | — | t=45 (`" has"` vs `"'s"`) |

---

## 2. Figure `ds14reg_margins.png` — the early razor ties are untouched; every late near-tie is new

![margins](figures/ds14reg_margins.png)

**Setup.** Both runs, same prompt, same directions, $r=0.1$. Each panel plots the full margin trace $m_t$ of that run's own greedy continuation, $t=0\dots299$. The grey band on the left is the region $t\le35$ where the two runs are identical by construction (it contains both razor ties); the dashed line is $t=35=W-P$. Dotted horizontals are 0.05, 0.1 and 0.5 nats. The inset lists the five smallest gaps and the number of steps below each threshold; the eight smallest are given in the table below.

**Axes.** $x$: generated step $t$ (0 = first generated token), 0–300. $y$: top-1 minus top-2 logit gap in nats, log scale $6\times10^{-5}$–40, chosen freely because the gaps span six decades.

**Numbers.**

| rank | full attention (step: nats) | true window 100 (step: nats) |
|---:|---|---|
| 1 | 21: $9.727\times10^{-5}$ | 21: $9.727\times10^{-5}$ |
| 2 | 19: $1.144\times10^{-3}$ | 19: $1.144\times10^{-3}$ |
| 3 | 189: $2.224\times10^{-3}$ | 297: $4.791\times10^{-3}$ |
| 4 | 273: $1.916\times10^{-2}$ | 145: $4.010\times10^{-2}$ |
| 5 | 255: $5.151\times10^{-2}$ | 88: $4.050\times10^{-2}$ |
| 6 | 171: $5.394\times10^{-2}$ | 239: $4.109\times10^{-2}$ |
| 7 | 168: $8.033\times10^{-2}$ | 271: $5.221\times10^{-2}$ |
| 8 | 205: $8.067\times10^{-2}$ | 152: $5.965\times10^{-2}$ |

| threshold (nats) | full | window |
|---|---:|---:|
| $m_t<0.05$ | 4 | 6 |
| $m_t<0.1$ | 8 | 9 |
| $m_t<0.5$ | 34 | 52 |
| median $m_t$ | 3.096 | 1.826 |

**What is in the figure.**
- The first two gaps are identical to the bit, because both steps precede the first clipped margin at $t=36$. The two ties are not merely "similar"; they are the same numbers.
- The margin traces diverge from $t=36$ onward. The maximum difference is 20.0 nats at $t=53$, where the full run has a very confident decision (21.2 nats) and the windowed run has a nearly tied one (1.20 nats; different tokens: ids 51998/24306 vs 11/13).
- The whole spectrum shifts down under the window: the median gap falls from 3.096 to 1.826 nats and the mean from 5.00 to 2.92, so the count below 0.5 nats rises from 34 to 52. This is not caused by the truncation of a few states; it is a property of the different continuation.
- The late near-tie list is completely replaced. Full attention has 189, 273, 255, 171, 168, 205; the window has 297, 145, 88, 239, 271, 152. Only $t=21$ and $t=19$ survive.
- The baseline text itself changes at $t=45$: full continues `"The user has specified several systems to compare: chemical rockets, nuclear propulsion, ..."`; the window continues `"The user's query is detailed, well-organized, suitable for a graduate-level course or professional reference. The document is detailed, well..."`. The windowed text is visibly more repetitive (see §3 and §8).

**What it shows.** The window changes the decision landscape in two distinct ways: it raises the number of gaps below 0.05 nats from 4 to 6 and below 0.5 nats from 34 to 52, and it replaces every late weak point with a new one, while leaving the two earliest ties exactly where they were.

**What it does not show.** It cannot say whether a given late margin differs because the attention renormalisation changed the state or because the text took a different branch at $t=45$; after $t=45$ the two traces are margins of different sentences. It also says nothing about which of these late ties would actually flip a *free-running* windowed continuation, because no such run exists yet and the thresholds are teacher-forced predictions.

---

## 3. Figure `ds14reg_response_shape.png` — a x2.2 flush at the first clipped token, then bursts

![response shape](figures/ds14reg_response_shape.png)

**Setup.** For each run, $\rho_t^{47}(u)=\lVert\delta h_t^{47}(u)\rVert/r$ at the last saved state (layer 47), median over the 1000 directions at each step; the shaded band is the 25–75% range over directions. Both curves use their own run's baseline (the perturbed and unperturbed forward passes of the same run), so the difference is the response to the same direction under the two attention masks — after $t=45$ on two different baseline sentences. Dashed line: $t=35=W-P$. Log $y$.

**Axes.** $x$: generated step $t$, 0–300. $y$: $\lVert\delta h_t^{47}\rVert/r$, median and interquartile band, log scale 0.15–25.

**Numbers.**

| quantity | full | true window 100 |
|---|---:|---:|
| median $\rho^{47}$ over $t\in[25,34]$ (pre-clip) | 0.836 | 0.836 |
| median $\rho^{47}$ over $t\in[36,45]$ | 0.832 | 1.837 |
| flush ratio $\Phi$ | 0.99 | **2.20** |
| $\Phi$ from per-step medians (median of ratios) | 0.93 | 2.11 |
| steps with median $\rho^{47}>3$ | 1 / 300 | 57 / 300 |
| largest per-step median $\rho^{47}$ | 3.15 at $t=258$ | 15.25 at $t=291$ |

Step-by-step ratio of the two median curves just after the first clip:

| step | $\rho^{47}$ full | $\rho^{47}$ window | window / full |
|---:|---:|---:|---:|
| 35 | 0.0877 | 0.0910 | 1.04 |
| 36 | 0.0864 | 0.2059 | 2.38 |
| 37 | 0.0592 | 0.1414 | 2.39 |
| 38 | 0.0751 | 0.1585 | 2.11 |

The per-state flush ratio for the window run is 1.65 (state 1, block 0), 2.83 (state 2), 2.72, 2.32, 2.11, 2.12, 2.02, 2.20 (state 8, layer 47); for the full run it lies between 0.92 and 1.12 at every state.

**What is in the figure.**
- The two curves are identical up to $t=34$. At $t=35$ the windowed response is 4% larger; at $t=36$ it is 2.4x larger, and it stays around 2x through $t=45$. This is the same flush the 1.5B showed, with a larger amplitude (2.20 against 1.73).
- The flush is strongest at shallow depth: 2.83 at state 2, against 1.65 at state 1 and 2.20 at the last state. The effect of the truncation is largest just after the first block and is partly damped on the way to layer 47.
- Beyond $t=50$ the windowed curve does not decay smoothly. It has two burst stretches, $t\approx114$–$132$ and $t\approx260$–$294$, where the median response reaches 6–15 per unit radius; the full run never exceeds 3.15. These stretches coincide with the most repetitive passages of the windowed text (`"...sections like an introduction, a structure, then sections on..."` around $t=114$; `"introduction, structure, content organization, conclusion..."` around $t=270$). The coincidence is suggestive but not a controlled test.
- In the troughs between bursts the windowed curve falls to 0.3–1.0, so the median over $[150,300)$ is 1.03 while the burst steps reach 15. The median is the robust statistic used below; the band shows the spread over directions is wide at every step.

**What it shows.** The true window produces the advertised flush at the first clipped prompt position, with a depth-dependent amplitude. It does not suppress the deep response later; if anything it amplifies it, intermittently.

**What it does not show.** The median curve is not a typical trajectory: the 25–75% band is a factor 2–3 wide and the bursts are a population effect, not a single direction blowing up. It does not establish that repetition causes the bursts (the text is itself an output of the window), and it cannot separate the direct effect of dropping a prompt key from the indirect effect of the different continuation. The response is measured at one radius only; a larger radius may leave the linear regime.

---

## 4. Figure `ds14reg_rstar.png` — 46.4% -> 49.3% predicted flips, all of them still at the same two ties

![rstar](figures/ds14reg_rstar.png)

**Setup.** Left: empirical distribution (ECDF) of the signed threshold radius $r_*(u)=\min_t r_c^-(t,u)$ over all 1000 directions, for both runs, truncated at $r_*=60$ (the largest finite value is 55.5). The vertical dashed line is the measurement radius $r=0.1$; the dotted horizontals are the predicted flip fractions. Right: the number of directions whose controlling step is each of eight steps, for both runs.

**Axes.** Left: $x$: threshold radius $r_*$, log, $2\times10^{-3}$–60; $y$: fraction of directions, 0–1. Right: $x$: directions with that controlling step, 0–590; $y$: step $t_*$, categorical.

**Numbers.**

| quantity | full | true window 100 |
|---|---:|---:|
| predicted flip fraction at $r=0.1$ | 46.4% (464) | 49.3% (493) |
| $r_*$ p10 / p50 / p90 / p95 | 0.0171 / 0.116 / 2.59 / 7.21 | 0.0171 / 0.104 / 1.85 / 3.47 |
| $r_*>1$ | 166 | 148 |
| controlling step 21 / 19 | 503 / 236 | 488 / 201 |
| controlling step 297 / 276 | 0 / 0 | 174 / 80 |
| flips controlled at 19 / 21 / 297 | 43 / 421 / 0 | 41 / 418 / 34 |
| flips shared with the other run | 464 | 464 |
| flips present only in this run | 0 | 29 (all at t=297) |

**What is in the figure.**
- The two ECDFs are indistinguishable below $r_*\approx0.02$ and separate only in the upper tail. The median threshold falls from 0.116 to 0.104 and the 90th percentile from 2.59 to 1.85; the window makes the average direction somewhat easier to flip, but not dramatically.
- The flip fraction is 46.4% under full attention, exactly the number in the earlier A1 section, and 49.3% under the window: a rise of 2.9 percentage points.
- The flip sets are nested: every one of the 464 full-attention flips is also a predicted flip at $r=0.1$ under the window. The 29 new flips are all controlled at $t=297$, the window's new late near-tie ($m=4.79\times10^{-3}$ nats). Five further directions have their controlling step moved to $t=297$ while already flipping at 19 or 21, which is why 34 flips are attributed to $t=297$.
- The controlling-step counts stay concentrated: 68.9% of all directions have $t_*=21$ or $19$ under the window (73.9% full), and 93.1% of the windowed flips are decided at those two steps (100% full). The window adds a third late gate; it does not replace the first two.

**What it shows.** The window changes the *count* of predicted flips by +2.9 points, but changes the *mechanism* of the extra flips completely: they come from a new late tie, not from the two razor ties moving. This is the opposite of the 1.5B window comparison quoted in the introduction: there the minimum margin moved from 0.0201 to 0.0031 and the controlling step changed. On the 14B the minimum margin is at $t=21$, before the first clipped position, so it cannot move; the window's effect appears only in the late tail.

**What it does not show.** These are *predicted* flips from a first-order teacher-forced response, not observed free-run flips. The prediction for the window run uses the window's own margins and `dm` on its own text; it is a statement about that continuation, not about all windowed continuations. The ECDF is truncated at 60; the tail above that is not shown. And the +29 is an addition to the flip *set*, not evidence that a free-running window run would flip 29 more directions: the linear rule is calibrated at $r=0.1$ but not re-validated on a windowed free run.

---

## 5. Figure `ds14reg_direction_gain.png` — the typical gain is 2.6x larger and the late decay nearly disappears

![direction gain](figures/ds14reg_direction_gain.png)

**Setup.** Left: ECDF of $G(u)=\mathrm{median}_t\lVert\delta h_t^{47}(u)\rVert/r$ over the 1000 directions, both runs. Right: ECDF of the per-direction late/early ratio $\mathrm{median}_{t\in[150,300)}\rho_t^{47}(u)\big/\mathrm{median}_{t\in[0,50)}\rho_t^{47}(u)$, both runs; the vertical black line is 1 (no change), the dotted lines are the medians.

**Axes.** Left: $x$: $G(u)$, log, 0.3–5; $y$: fraction of directions. Right: $x$: late/early ratio, log, 0.2–4; $y$: fraction of directions.

**Numbers.**

| quantity | full | true window 100 |
|---|---:|---:|
| $G$ p10 / p50 / p90 | 0.409 / 0.583 / 1.297 | 0.936 / 1.493 / 2.659 |
| $G$ mean | 0.738 | 1.670 |
| median per-direction late/early | 0.570 | 0.917 |
| median per-direction late/pre-clip early ($t\in[25,34]$) | 0.594 | 1.147 |
| pooled late/early ($\Lambda$) | 0.577 | 0.930 |
| directions with late/early $>1$ | 6.1% | 42.3% |

**What is in the figure.**
- The entire gain distribution shifts right by a factor 2.6 at the median. The shift is not a few outliers: p10 rises from 0.41 to 0.94 and p90 from 1.30 to 2.66, so the two ECDFs barely overlap.
- Under full attention the late response is systematically smaller than the early one: the median direction retains 57% of its $[0,50)$ response, and 94% of directions are below 1. Under the window the median retains 92%, and 42% of directions are above 1.
- Measured against the pre-clip level $[25,34]$, which is bit-identical in the two runs, the contrast is sharper: full falls to 0.59, the window *rises* to 1.15. The window does not cut the late response; it roughly preserves it, and at depth it increases it.
- The premise "the window cuts the prompt's direct influence after $t=35$, so the late response should be smaller" is not what the data show. The direct prompt visibility does fall — one prompt position leaves per step, and none are left after $t=99$ — but by then the perturbation has been written into the key/value states of generated tokens that remain inside the window, and those states feed every later step. The late response is therefore carried by the generated-token chain rather than by direct prompt attention, and it is larger, not smaller. The data cannot apportion the effect between attention reweighting, the different continuation, and deep-layer amplification.

**What it shows.** The window raises the response gain of every direction and nearly removes the late-time decay that full attention shows. The effect is a population shift, not a tail.

**What it does not show.** $G$ is an absolute norm; the response relative to the residual-stream norm stays small (about $10^{-4}$ typically and $10^{-3}$ at the largest burst), so "larger gain" does not mean a destabilised state. The comparison is between two different texts after $t=45$, so the late/early ratio mixes the mask effect with the continuation effect. The $[0,50)$ window contains the flush in the windowed run, which inflates its denominator; the $[25,34]$ variant is given for that reason.

---

## 6. Figure `ds14reg_depth_gain.png` — the window halves block 0 and doubles layer 47

![depth gain](figures/ds14reg_depth_gain.png)

**Setup.** For each saved state $l$, the median of $\lVert\delta h^l(u)\rVert/r$ over all directions and all 300 steps, both runs. The dashed grey line is the same quantity computed on $t<35$ only; there the two runs are bit-identical, so one line serves both and it is a direct check that the pre-window depth profile is the common starting point. State 0 is omitted because its response is exactly zero (the perturbation is added to the prompt embeddings, not to the generated-token embeddings).

**Axes.** $x$: saved state index 1–8, labelled with the layer the engine stored (state 1 = block 0 output, layer 6; state 8 = layer 47). $y$: $\lVert\delta h^l\rVert/r$, median over $t$ and directions, log 0.028–2.6.

**Numbers.**

| state (layer) | full, all $t$ | window, all $t$ | both, $t<35$ |
|---|---:|---:|---:|
| 1 (6) | 0.0775 | 0.0376 | 0.2035 |
| 2 (12) | 0.1006 | 0.1587 | 0.2484 |
| 3 (18) | 0.1001 | 0.1785 | 0.2192 |
| 4 (24) | 0.1102 | 0.2051 | 0.2095 |
| 5 (30) | 0.1510 | 0.3145 | 0.2317 |
| 6 (36) | 0.2504 | 0.5237 | 0.3241 |
| 7 (42) | 0.4285 | 0.9234 | 0.5626 |
| 8 (47) | 0.6532 | 1.4509 | 0.8739 |
| growth 1→8 | x8.4 | x38.6 | x4.3 |

**What is in the figure.**
- Before the window acts, the depth profile is identical in the two runs: it rises from 0.2035 at block 0 to 0.8739 at layer 47, a factor 4.3.
- Averaged over all 300 steps, full attention *lowers* the whole profile because the late response decays: 0.0775 at block 0 to 0.6532 at layer 47 (factor 8.4). The window lowers block 0 further (0.0376, half the full-run value) but raises every state from 2 upward, ending at 1.4509 at layer 47 (2.2x the full-run value; factor 38.6 from block 0).
- The two profiles cross between states 1 and 2 relative to each other (window below full at state 1, above it from state 2 on); the window profile crosses its own pre-window level between states 4 and 5. Shallow layers see a *smaller* time-averaged response under the window, deep layers a *larger* one. This is the opposite of a uniform attenuation, and it is consistent with the shallow flush being damped on the way down while the late bursts accumulate at depth.

**What it shows.** The window reverses the direction of the depth change: the response becomes more top-heavy, not uniformly weaker. The growth from the first block to layer 47 rises from 8.4x to 38.6x.

**What it does not show.** The profile is descriptive: it mixes two effects (the time average includes the late bursts and the different text) and the eight states are samples, not all 48 blocks. A monotone statement about every layer is not justified. The pre-window line is a check on the setup, not a separate condition.

---

## 7. Facts that qualify the whole comparison

1. **The window changes the text itself.** The greedy continuations are identical only up to $t=44$ and differ at $t=45$ (`" has"` vs `"'s"`). Every margin, threshold and response after that step describes a different sentence. The design is paired in prompt and directions, not in continuation.
2. **The available calibration is not a true-window calibration.** `results_big/ds14/calib/` has `window: 100` in its metadata and was intended as a true-window calibration, but its baseline tokens equal the full-attention run's and its margins match them to $7\times10^{-5}$ nats. The window was inactive in that run too. Radius linearity is therefore established for full attention (at layer 47, $\mathrm{d}h/r$ is flat to 1.1% between $r=0.03$ and $0.3$, and rises at $r\ge1$), not for the windowed run. The windowed run provides a single radius.
3. **Index offset.** The engine stores the response at position $P+t$ and the margin of the token predicted from $P+t-1$. All statements about "the first clipped step" use $t=35$ for states and $t=36$ for margins; comparing a margin at 35 to a response at 35 would be off by one.
4. **Flips are predictions.** 464 and 493 are teacher-forced first-order predictions at $r=0.1$, not observed free-run flips. The 46.4% full-attention value reproduces the earlier A1 section exactly, which is a consistency check, not a new validation.
5. **A "burst" is a deep-layer response, not a divergence.** It reaches $\lVert\delta h^{47}\rVert\approx1.5$ at unit radius, about $1.4\times10^{-3}$ of the baseline state norm (1077). The baseline scale is nearly the same in both runs, so the burst is a real absolute difference, but it is small relative to the stream.
6. **The 29 extra flips are one gate, not a broad increase.** All 29 are controlled at $t=297$, a step whose margin is $4.8\times10^{-3}$ nats. Five more directions switch their *controlling* step to 297 without changing their flip status. Any statement about "the window makes the model more flippable" rests on this single late tie.
7. **One prompt, one model, one radius, teacher forcing.** The windowed run has no free-run counterpart; the free-run validation was performed on the full-attention run only. The text after $t=45$ is specific to this prompt and seed.

---

## 8. What follows directly

- The window law holds on the 14B exactly: states are bit-identical for $t<35=W-P$ and depart at $t=35$, the same test that held to the token on the 1.5B for five windows.
- The 1.5B's margin result does **not** transfer as a change of the minimum: on the 14B the minimum is $9.727\times10^{-5}$ nats at $t=21$ in both runs, because $t=21$ lies before the first clipped position. The window's margin effect appears as new late ties and a lower median gap (3.10 -> 1.83 nats), not as a moved razor tie.
- The predicted flip fraction rises from 46.4% to 49.3%, entirely by adding 29 directions controlled at the new $t=297$ tie. No full-attention flip is lost, and 93% of the windowed flips are still decided at $t=19/21$.
- The flush signature is present and larger than on the 1.5B: the last-layer response jumps by 2.20 across $t=35$ (1.5B: 1.73), and the jump is strongest at shallow depth (2.83 at state 2).
- The late response is *not* suppressed. The median direction keeps 0.92 of its early response under the window against 0.57 under full attention, and the median gain is 2.6x larger. The window therefore does not behave like a truncation of the prompt's influence; it behaves like a persistent, more repetitive context.
- The windowed continuation is visibly repetitive from about $t=60$ (`"...professional reference. The document is detailed, well-organized, suitable for a graduate-level course or professional reference..."`). The two response bursts fall inside the most repetitive stretches. Whether the repetition causes the bursts is an open question and the natural target of the windowed free run.

## 9. Reproduce

```
python scripts_spark/figs_w100_vs_full.py
```

Writes `notebook_big/figures/ds14reg_{margins,response_shape,rstar,direction_gain,depth_gain}.png` (dpi 170) and `notebook_big/figures/ds14reg_numbers.json`. The JSON contains the eight smallest margins and threshold counts per run, the full median response profiles, the per-state flush ratios, the burst ranges, the $r_*$ distributions, the gain distributions, the controlling-step tables, the flip-set comparison, the radius-linearity table, and both auxiliary checks. The script reads the npz files in place and does not modify them.
