# Part A1-deep. One thousand directions on the 14B model: where the flips come from

## Summary

A single prompt of 65 tokens is sent through DeepSeek-R1-Distill-Qwen-14B in FP32, the model generates 300 tokens greedily, and then 1000 different unit perturbations of size $r=0.1$ are added to the prompt embeddings while the generated tokens are held fixed. For every direction and every generated step we record how much the top-1/top-2 logit gap would move; from that we compute the radius at which each step would flip. Almost half the directions (464 of 1000) would flip some token at $r=0.1$, but **all 464 flips are decided at only two generated steps, $t=21$ and $t=19$**, the two smallest logit gaps in the run. The directions that destroy the margin at $t=21$ are essentially unrelated to those that destroy it at $t=189$ (correlation $-0.14$), and the 24 stored response vectors are nearly orthogonal (mean cosine $+0.0025$). Consequently the direction's overall response strength $G$ is a weak predictor of its threshold radius (correlation $-0.18$): which razor-thin tie a direction happens to hit matters far more than how strong its hidden-state response is.

---

## 1. Exact setup, data, and every quantity defined

### 1.1 The run

| item | value |
|---|---|
| model | `DeepSeek-R1-Distill-Qwen-14B` (48 blocks, hidden size 5120) |
| precision | FP32, one precision for the whole run; TF32 disabled |
| decoding | greedy, fixed length, no sampling, no EOS stop |
| prompt | "Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. Compare chemical rockets, nuclear propulsion, laser sails, antimatter drives, and other theoretical concepts in terms of energy requirements, achievable speeds, technological feasibility, and projected timelines for development. Include discussion of major projects in the history of the field." |
| prompt length | $P=65$ tokens |
| attention window | 100 tokens (`set_window(model, 100)`) |
| continuation | $T=300$ baseline greedy tokens, stored in `gen_ids` |
| directions | $K=1000$, built as 50 independent simplex blocks of 20 |
| direction geometry | each direction is a unit vector with 20 non-zero coordinates on the $65\times5120$ prompt-embedding matrix; blocks use disjoint coordinates, so cross-block directions are exactly orthogonal; within a block every pair has inner product $-1/19=-0.0526$ |
| radius | one radius $r=0.1$ (`radii=[0.1]`) |
| perturbation support | all 65 prompt positions (not the generated positions) |
| teacher forcing | every perturbed forward pass is run on the **baseline** `prompt_ids` + `gen_ids`; the continuation tokens are identical to the unperturbed run, so the measured $\delta m$ is the continuous response at each baseline step, not the trajectory after a divergence |
| baseline check | two identical rows in one batch produced bit-identical tokens (`baseline_duplicate_identical: true`) |
| saved states | hidden-state indices `[0, 6, 12, 18, 24, 30, 36, 42, 47]`; state 0 is the embedding output, state $l\ge1$ is the output of block $l-1$ (transformers convention), so state 47 is the output of block 46, not the final block. The final-norm/lm_head input was stored only as 24 difference vectors (`dh_last`, directions 0–23) |
| engine / environment | `run_a1_big`, torch 2.14.1+cu130, transformers 4.57.6, 1590.6 s wall time on `spark-d5eb` |
| file | `results_big/ds14/a1_inter_full/a1.npz` (+ `meta.json`) |

“Window 100” matters for interpretation: once $65+t>100$, the earliest prompt positions fall outside the attention window. The perturbation is applied only to prompt positions, so at late steps its influence reaches the current token indirectly, through the states of generated tokens that were computed while the prompt was still visible. The late-time response is therefore not a direct prompt-embedding effect.

### 1.2 Quantities used in this section

For a direction $u$ (row of `simplex`) and a generated step $t$:

| symbol | explicit expression | what it is | why this quantity |
|---|---|---|---|
| $m_t$ | $\ell_t^{(1)}-\ell_t^{(2)}$ | top-1 minus top-2 logit gap of the unperturbed greedy run (nats) | the only scalar that says how close the baseline decision is; a flip requires this gap to be closed |
| $\delta m_t(u)$ | $\ell_t'^{(1)}-\ell_t'^{(2)}-m_t$ at the perturbed forward pass (stored `dm`) | change of that gap under perturbation (nats) | the continuous response of the decision variable; it is measured without letting the trajectory diverge |
| $g_t(u)$ | $\lvert\delta m_t(u)\rvert/r$ | local sensitivity of the gap to radius (nats per unit radius) | converts a response into the radius needed to close the gap |
| $r_c^-(t,u)$ | $m_t\,r/\max(-\delta m_t(u),0)$, $+\infty$ if $\delta m_t\ge0$ | the radius at which step $t$ would flip, for direction $u$ | a flip needs $\delta m_t\le -m_t$; solving for the radius gives this expression |
| $r_*(u)$ | $\min_t r_c^-(t,u)$ | the direction's threshold radius (smallest radius that flips **some** token) | one number per direction; direction $u$ flips at the run radius iff $r_*(u)\le r$ |
| $t_*(u)$ | $\arg\min_t r_c^-(t,u)$ | the controlling step | identifies which near-tie is the weak point for this direction |
| $G(u)$ | $\mathrm{median}_t \lVert\delta h_t^{47}(u)\rVert/r$ | typical last-state response gain (per unit radius) | a robust, scalar measure of how strongly the direction moves the final stored state |
| $R_t^{l}(u)$ | $\lVert\delta h_t^{l}(u)\rVert/\lVert h_t^{l}\rVert$ | relative response at state $l$ | says what fraction of the residual stream is being displaced; the absolute norm alone grows with depth simply because the stream grows |
| $\cos_{ij}$ | $\langle \delta h_i,\delta h_j\rangle/(\lVert\delta h_i\rVert\lVert\delta h_j\rVert)$ | angle between two response vectors | tests whether all directions produce one common response mode (1-D geometry) or many |
| $r_*(u)G(u)$ | product of the two scalars above | collapse test for the naive scaling $r_*\sim m/G$ | if $\delta m_t$ at the controlling step were proportional to $G$, this product would be constant |

The signed threshold is essential. The engine's stored `r_c` array was computed with $\lvert\delta m\rvert$ (`run_a1_big.py`, line 212), not with $\max(-\delta m,0)$. Recomputing the signed version is not cosmetic: the unsigned definition calls **810** directions flips at $r=0.1$, including directions whose margin is *widened* and which cannot flip. The signed definition gives **464**. Every number below uses the signed definition.

### 1.3 The ten smallest gaps and who is controlled by them

| step $t$ | $m_t$ (nats) | directions with $t_*=t$ (all 1000) | of which flip at $r=0.1$ |
|---:|---:|---:|---:|
| 21 | $9.727\times10^{-5}$ | 503 | 421 |
| 19 | $1.144\times10^{-3}$ | 236 | 43 |
| 189 | $2.224\times10^{-3}$ | 146 | 0 |
| 273 | $1.916\times10^{-2}$ | 56 | 0 |
| 255 | $5.151\times10^{-2}$ | 8 | 0 |
| 171 | $5.393\times10^{-2}$ | 5 | 0 |
| 168 | $8.033\times10^{-2}$ | 1 | 0 |
| 205 | $8.067\times10^{-2}$ | 1 | 0 |
| 0 | $1.009\times10^{-1}$ | 5 | 0 |
| 247 | $1.121\times10^{-1}$ | 15 | 0 |

For reference the margin distribution is very skewed: median $m=3.096$ nats, mean $4.996$, minimum $9.727\times10^{-5}$ at $t=21$; 8 of 300 steps have $m<0.1$, 3 have $m<0.01$. The threshold radii are equally skewed: over all 1000 directions, $r_*$ has p10 $=0.0171$, median $=0.116$, p90 $=2.59$, maximum $55.5$. Among the 464 directions that flip, $r_*$ has median $0.0314$ (p10 $0.0112$, p90 $0.0770$, max $0.0995$); 536 directions have $r_*>0.1$ and 166 have $r_*>1$.

---

## 2. Figure `ds14_controlling_steps.png` — two steps decide every flip

![controlling steps](figures/ds14_controlling_steps.png)

**Setup.** Same run as §1.1 (DS-R1-14B, FP32, greedy, $P=65$, window 100, $T=300$, 1000 unit directions, $r=0.1$, teacher-forced on the baseline tokens). Panel (a) is a histogram of $t_*(u)$ over all 1000 directions; the light bars are all directions, the dark bars are the subset with $r_*(u)\le0.1$ (the directions that actually flip). Panel (b) restricts the $x$-axis to the ten smallest-margin steps and counts how many directions have that step as their controller.

**Axes.**
- (a) $x$: generated step $t$ (0 = first generated token); $y$: number of directions with $t_*(u)=t$.
- (b) $x$: the ten near-tie steps, with the value of $m_t$ printed on the second line; $y$: number of directions controlled by that step, log scale.

**What is in the figure.**
- There are 22 distinct controlling steps among 1000 directions, but only **2** among the 464 flipping directions.
- Top-3 overall: $t=21$ controls 503 directions (421 flips), $t=19$ controls 236 (43 flips), $t=189$ controls 146 (0 flips). These three account for $885/1000=88.5\%$ of all directions. The two flipping steps account for $464/464=100\%$ of the flips.
- The remaining 115 directions are spread over 19 steps, the largest being $t=273$ with 56 directions, $t=247$ with 15, $t=98$ and $t=255$ with 8 each.
- In panel (b), the bars are not monotone in $m_t$: $t=247$ ($m=0.112$) controls 15 directions while $t=255$ ($m=0.0515$) controls 8 and $t=171$ ($m=0.0539$) controls 5. A near-tie is a necessary condition for a small $r_*$, but the response at that step also has to point the right way.

**What the figure shows.** The weak points are shared: nearly every direction is limited by the same two tokens, $t=21$ and $t=19$. It is not the case that a thousand directions explore a thousand different decisions.

**What it does not show.** It does not show that the other eight near-ties are irrelevant for other radii or other prompts. At $r=0.1$ their $r_*$ is above the run radius, so they do not flip here; at a larger radius they would. The figure also says nothing about the free-run consequences of a flip (that requires the separate free-run validation).

---

## 3. Figure `ds14_sign_structure.png` — half destroy, half widen, and the two ties are independent

![sign structure](figures/ds14_sign_structure.png)

**Setup.** Same run. For each of the three most important controlling steps ($t=21,19,189$) the figure histograms the signed response $\delta m_t(u)/r$ over all 1000 directions. Red is $\delta m_t<0$ (the margin is being destroyed), blue is $\delta m_t\ge0$ (the margin is being widened; such a direction can never flip at that step). The dashed vertical line is $-m_t/r$: a direction to its left would flip **at** this step at radius $r=0.1$. The shaded region is that flip region. Panel (b) is a scatter of $\delta m_{21}$ against $\delta m_{189}$ in millinats (raw $\delta m$, no division by $r$).

**Axes.**
- (a1)–(a3) $x$: $\delta m_t(u)/r$ (nats per unit radius); $y$: directions per bin.
- (b) $x$: $\delta m_{21}(u)$ (millinats, $10^{-3}$ nats); $y$: $\delta m_{189}(u)$ (millinats).

**Numbers for the three histograms.**

| step | $m_t$ | $\delta m<0$ | $\delta m\ge0$ | pass $r_c^-\le0.1$ at this step | median $\delta m/r$ | p25 / p75 | p5 / p95 | median $g_t$ | median $g_t/G$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | $9.727\times10^{-5}$ | 52.9% | 47.1% | 42.7% | $-3.13\times10^{-4}$ | $-2.67\times10^{-3}$ / $+2.56\times10^{-3}$ | $-8.21\times10^{-3}$ / $+8.99\times10^{-3}$ | $2.61\times10^{-3}$ | $4.14\times10^{-3}$ |
| 19 | $1.144\times10^{-3}$ | 49.1% | 50.9% | 7.5% | $+2.33\times10^{-4}$ | $-4.65\times10^{-3}$ / $+5.11\times10^{-3}$ | $-1.40\times10^{-2}$ / $+1.40\times10^{-2}$ | $4.87\times10^{-3}$ | $7.47\times10^{-3}$ |
| 189 | $2.224\times10^{-3}$ | 50.7% | 49.3% | 0.1% | $-8.63\times10^{-5}$ | $-2.64\times10^{-3}$ / $+2.30\times10^{-3}$ | $-8.09\times10^{-3}$ / $+6.77\times10^{-3}$ | $2.48\times10^{-3}$ | $5.23\times10^{-3}$ |

The sign split is close to 50/50 at all three steps. The distributions are centered near zero and are symmetric to within a few percent. At $t=21$ the median absolute response is only $3.1\times10^{-4}$ nats per unit radius, i.e. **about 16 times smaller than the median response at $t=19$** ($4.9\times10^{-3}$) and **about 1900 times smaller than the median gain $G$** ($0.583$). The gap at $t=21$ is so thin that even this tiny projection is enough to close it.

The “pass” column is not the same as the “$\delta m<0$” column. At $t=21$, 52.9% of directions decrease the gap but only 42.7% decrease it by at least $m_t$; the remaining 10.2% move it by less than $9.7\times10^{-5}$ nats and do not flip at $r=0.1$. At $t=19$ the corresponding fraction passing is 7.5%; at $t=189$ it is 0.1%.

**Among the flipping subset.** At $t=21$, 92.45% of the 464 flipping directions have $\delta m_{21}<0$ (the other 7.55% are controlled at $t=19$ and are margin-widening at 21). At $t=19$ only 50.4% are negative, and at $t=189$ 47.2%; those two steps are not the controllers for most flips.

**Panel (b): are the two near-ties the same direction geometry?** No.

| pair | Pearson (all) | Spearman (all) | Pearson (flips) | Spearman (flips) |
|---|---:|---:|---:|---:|
| 21 vs 189 | $-0.142$ ($p=6.8\times10^{-6}$) | $-0.123$ | $-0.053$ | $-0.092$ |
| 21 vs 19 | $+0.072$ | $-0.032$ | $-0.176$ | $-0.207$ |
| 19 vs 189 | $-0.047$ | $-0.030$ | $-0.027$ | $-0.021$ |

Quadrant counts for 21 vs 189 (all directions): $(-,-)$ 245, $(-,+)$ 284, $(+,-)$ 262, $(+,+)$ 209 — essentially uniform. The 21-vs-189 correlation is statistically detectable because $n=1000$, but it explains only $0.02$ of the variance ($r^2=0.020$). The 21-vs-19 correlation even changes sign between all directions and the flipping subset. The two near-ties are not controlled by one shared direction geometry: a direction that suppresses the margin at 21 has essentially a 50/50 chance of widening it at 189.

**Why this quantity.** The signed response is the right object because a flip is one-sided: only $\delta m_t\le -m_t$ can flip, so the sign is part of the definition, not a nuisance. A histogram of $\lvert\delta m_t\rvert$ would hide the almost perfect 50/50 split and would make the direction set look uniformly dangerous when half of it is protective.

**What the figure does not show.** It does not show which directions flip in a free run after the first divergence, and it does not resolve the $t=21$ threshold visibly on the plotted scale: the dashed line sits at $-9.7\times10^{-4}$ per unit radius, only about one bin from zero. The printed “pass 42.7%” is the reliable number for that threshold.

---

## 4. Figure `ds14_gain_structure.png` — strong, broadly spread, and nearly orthogonal responses

![gain structure](figures/ds14_gain_structure.png)

**Setup.** Same run. Panel (a) uses all 1000 directions and the state-47 response norm at every step; panel (b) uses the 24 directions (0–23) for which the full final-norm difference vectors were stored.

**Definitions for this figure.**
- $G(u)=\mathrm{median}_t\lVert\delta h_t^{47}(u)\rVert/r$. State 47 is the deepest stored hidden state (output of block 46, per the transformers index convention). The median is over the 300 generated steps.
- $\mathrm{median\ step}$ $t_{\rm med}(u)=\arg\min_t\bigl\lvert\lVert\delta h_t(u)\rVert-\mathrm{median}_{t'}\lVert\delta h_{t'}(u)\rVert\bigr\rvert$: the step whose response norm is closest to the direction's median norm. The 24 vectors are compared at their own median step.
- $\cos_{ij}$: cosine between the final-norm difference vectors of directions $i$ and $j$ at their median steps.

**Gain numbers (all 1000 directions).**

| statistic | $G(u)$ |
|---|---:|
| min | 0.224 |
| p5 | 0.370 |
| p25 | 0.475 |
| median | 0.583 |
| p75 | 0.877 |
| p95 | 1.648 |
| max | 3.064 |
| max/min | 13.7× |

The distribution is approximately log-normal (log10 $G$ has mean $-0.181$, std $0.195$). For comparison, the **arithmetic mean over $t$** has median 0.815 (1.40× the median of $G$), and the **maximum over $t$** has median 8.44 (14.5× the median of $G$). The mean and max are dominated by token-locked response bursts (§5), which is why $G$ uses the median. This is a deliberate choice: the mean would rank directions by whether they happen to sit on a burst token, not by their typical strength.

**The 24 stored vectors.** Their norms at the median step have median $1.528\times10^{-2}$ (p5 $1.05\times10^{-2}$, p95 $4.14\times10^{-2}$). Pairwise cosine over all $\binom{24}{2}=276$ pairs:

| statistic | pairwise $\cos_{ij}$ |
|---|---:|
| mean | $+0.0025$ |
| median | $+0.0026$ |
| p5 | $-0.062$ |
| p25 | $-0.028$ |
| p75 | $+0.025$ |
| p95 | $+0.081$ |
| min | $-0.147$ |
| max | $+0.222$ |
| fraction with $\lvert\cos\rvert>0.1$ | 4.3% |

The responses are nearly orthogonal. The first principal component explains 6.5% of the variance, versus 4.17% expected for 24 isotropic vectors; the first eight PCs are 6.5, 6.3, 5.5, 5.2, 5.1, 4.6, 4.6, 4.5%. The cosine of each response vector to the pooled mean response has median 0.196, essentially the isotropic value $1/\sqrt{24}=0.204$. There is at most a faint common component, not a shared 1-D mode.

Within-block and cross-block responses are statistically indistinguishable: directions 0–19 share the same input subspace (their input-space cosines are $-1/19=-0.053$) but their response cosines average $+0.0025$; directions 20–23 are input-orthogonal to block 0 and their response cosines to it average $+0.0033$. The network does not preserve the input block geometry, and it does not collapse all responses onto one axis.

The response vectors are also only partly coherent in time: the median cosine between $\delta h_t$ and $\delta h_{t+1}$ is 0.317 (mean 0.329), at lag 10 it is 0.090 (mean 0.109), and at lag 50 it is 0.061 (mean 0.080). Adjacent steps share about a third of their direction; by ten steps the memory of the direction is mostly gone.

**The requested response-vs-baseline-state angle is not recoverable from this dump.** The run stored baseline state **norms** (`norm_h0`), not baseline state **vectors**, and it stored the cosine between the perturbed state and the baseline state, $\cos(h',h_0)$. That state cosine is $0.99998\ldots1.0000004$ (median exactly 1.0; 49.9% of entries are exactly 1.0 in float32). With $\lVert\delta h\rVert/\lVert h_0\rVert\sim10^{-4}$, the true angle would change $\cos(h',h_0)$ by only $\sim10^{-8}$, below the float32 resolution of the stored array. Recovering $\theta(\delta h,h_0)$ from these numbers is therefore impossible, not merely noisy. What can be said from the stored state cosine is that the perturbed state differs from the baseline state in direction by at most $\sqrt{2(1-\cos)}\approx0.007$ rad, which is an upper bound on the total rotation, not the angle of the response. A future run that stores the 5120-dimensional baseline vectors would make the requested quantity measurable. The substitute used here is the pairwise response cosine, which answers the more important question for this project: whether the geometry is effectively 1-D.

**What the figure shows.** Strength varies by a factor 13.7, but direction is almost uniformly distributed on the sphere: the responses are not copies of one another and there is no single “perturbation mode.” A scalar gain is therefore an incomplete description of a direction, and one should expect directions with similar $G$ to have very different effects on any particular logit gap.

**What it does not show.** It does not measure the response angle relative to the baseline state (not stored), and 24 directions are a small sample: 20 come from block 0 and 4 from block 1, so the cross-block comparison rests on 80 pairs. The PCA spectrum is estimated from 24 vectors and cannot resolve a common component much below 4%.

---

## 5. Figure `ds14_shape_by_layer.png` — depth grows the response; the relative response shrinks

![shape by layer](figures/ds14_shape_by_layer.png)

**Setup.** Same run, all 1000 directions. Each of the nine small panels is one stored hidden state $l\in\{0,6,12,18,24,30,36,42,47\}$. The solid line is the median over directions of $\lVert\delta h_t^l(u)\rVert/r$ at each step; the band is the interquartile range over directions. The right panel is a heatmap of the median **relative** response $\lVert\delta h_t^l(u)\rVert/\lVert h_t^l\rVert$ for the eight non-zero states, with a log color scale.

**Axes.**
- Small panels: $x$ = generated step $t$ (0–299); $y$ = $\lVert\delta h_t^l\rVert/r$ (state-response norm per unit radius). Free $y$ range per panel.
- Right panel: $x$ = step $t$; $y$ = state index $l$; color = $\lVert\delta h_t^l\rVert/\lVert h_t^l\rVert$ (dimensionless, log scale 5e-5 to 5e-4).

**State 0 is exactly zero.** At $l=0$ the state of a *generated* token is its token embedding; the perturbation is added only to the 65 *prompt* embeddings, so the generated-token embedding is untouched at every $t$. The array `dh_norm[:,0,:]` is exactly 0.0 for all $1000\times300$ entries (checked; max absolute value 0). This is a structural zero, not a small number. The response first appears at state 6, after attention has mixed prompt information into the generated positions.

**Per-layer numbers (median over directions).**

| state $l$ | median $G$-like gain | abs at $t=0$ | $t=10$ | $t=50$ | $t=100$ | $t=258$ | $t=299$ | peak of median curve | end/peak | median relative response |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | 0 |
| 6 | 0.0775 | 0.286 | 0.194 | 0.137 | 0.0815 | 0.0473 | 0.0521 | $t=1$, 0.301 | 0.17 | $1.53\times10^{-4}$ |
| 12 | 0.101 | 0.359 | 0.266 | 0.184 | 0.133 | 0.0596 | 0.0663 | $t=1$, 0.380 | 0.17 | $1.23\times10^{-4}$ |
| 18 | 0.100 | 0.321 | 0.281 | 0.176 | 0.292 | 0.125 | 0.0564 | $t=1$, 0.321 | 0.18 | $1.08\times10^{-4}$ |
| 24 | 0.110 | 0.297 | 0.253 | 0.183 | 0.502 | 0.230 | 0.0584 | $t=100$, 0.502 | 0.12 | $9.72\times10^{-5}$ |
| 30 | 0.151 | 0.322 | 0.253 | 0.242 | 0.824 | 0.429 | 0.100 | $t=100$, 0.824 | 0.12 | $8.75\times10^{-5}$ |
| 36 | 0.250 | 0.486 | 0.334 | 0.318 | 0.864 | 0.692 | 0.227 | $t=97$, 0.946 | 0.24 | $8.11\times10^{-5}$ |
| 42 | 0.428 | 0.767 | 0.561 | 0.451 | 1.242 | 1.707 | 0.425 | $t=258$, 1.707 | 0.25 | $7.85\times10^{-5}$ |
| 47 | 0.653 | 1.143 | 0.865 | 0.631 | 1.682 | 3.150 | 0.688 | $t=258$, 3.150 | 0.22 | $6.18\times10^{-5}$ |

**Depth scaling.** The typical gain rises from 0.0775 at state 6 to 0.653 at state 47, a factor **8.43**. The peak of the median curve rises from 0.301 to 3.150, a factor 10.5. The baseline state norm, however, rises much faster:

| state | 0 | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 47 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| median $\lVert h_t^l\rVert$ | 1.52 | 51.5 | 84.0 | 93.6 | 114.3 | 175.2 | 311.5 | 549.8 | 1067.3 |

That is why the relative response *falls*: median relative response goes from $1.53\times10^{-4}$ at state 6 to $6.18\times10^{-5}$ at state 47, a factor **2.47**. The residual stream grows roughly $1067/51.5=20.7\times$ over the same range while the response grows only $8.4\times$.

Both normalizations are necessary. The absolute norm is what the logits see: the margin change is $\delta m_t=\delta h_t^{\rm final}\cdot(W_{[{\rm top1}]}-W_{[{\rm top2}]})$, so a larger absolute response can move a gap more. The relative norm says what fraction of the state is being displaced, i.e. whether the response is a small perturbation of a growing stream or a large one. At state 47 the displacement is $\sim6\times10^{-5}$ of the state norm: tiny in relative terms, yet $0.65$ units of absolute norm per unit radius, enough to matter because the logit-gap direction is even more sensitive.

**There is no simple rise–peak–decay in time.** The response is largest at $t=0$–$1$ for states 6–18 (the first generated position sees the most direct prompt influence), then decays. At deeper states the largest values move to token-locked bursts: $t=258$ at states 42 and 47, $t=97$–$100$ at states 24–36, and $t=58,39,26$ in individual directions. At state 47 the median curve is 1.143 at $t=0$, 0.631 at $t=50$, 1.682 at $t=100$, **3.150 at $t=258$**, and 0.688 at $t=299$. The peak step of the median curve therefore moves from $t=1$ (states 6–18) to $t=97$–$100$ (states 24–36) to $t=258$ (states 42–47). Per direction, the last-state maximum occurs at $t=258$ for 109/1000 directions, $t=58$ for 62, $t=39$ for 60, $t=97$ for 48, $t=26$ for 45; the rest are spread over many tokens.

This is why the “rise time” is not a fixed transient: the response amplitude is a property of the *baseline token* as much as of the layer. The same perturbation produces a 14.5× larger response at its best token than at its median token. Any single “response time” or “decay time” would be an artifact of averaging over token content.

**Why the heatmap.** Plotting eight relative curves on one axis produced an unreadable band because the curves differ by an order of magnitude and all contain sharp token spikes. The heatmap keeps the same information (state × step) and makes the two trends visible at once: an early bright region that fades with depth, and a bright vertical stripe at $t=258$ that appears only at the deepest states.

**What the figure does not show.** It does not separate “the perturbation is decaying” from “the prompt has left the 100-token attention window”; both happen together after $t\approx35$. It also does not show the final-norm response for all 1000 directions (only state 47 is stored for all; the final norm only for 24). And the small panels are medians over directions: a single direction can have a peak anywhere, as the per-direction peak counts show.

---

## 6. Figure `ds14_rstar_vs_gain.png` — the tie, not the gain, sets the threshold

![rstar vs gain](figures/ds14_rstar_vs_gain.png)

**Setup.** Same run. Panel (a) is a log–log scatter of $r_*(u)$ against $G(u)$ for the 834 directions with finite $r_*<1$ (166 directions have $r_*>1$; the threshold is above 1 and is not shown). Points are colored by the controlling step. The dashed lines are $r_*=c_t/G$ fitted separately inside each controlling-step group, with $c_t=\mathrm{median}(r_*G)$ for that group. The dotted line is the naive analytic guess $r_*=m_{\min}/G$ with the global minimum margin $m_{\min}=9.727\times10^{-5}$. Panel (b) is a box plot of $r_*G$ grouped by controlling step, with the number of directions in each group.

**Axes.**
- (a) $x$: $r_*(u)=\min_t r_c^-(t,u)$ (threshold radius, log); $y$: $G(u)$ (median state-response gain, log).
- (b) $x$: controlling step $t_*$; $y$: $r_*(u)G(u)$ (nats, log). A horizontal dashed line marks the median over all 834 directions.

**Deriving the scaling.** A step flips when
$$m_t+\delta m_t\le0 \quad\Longleftrightarrow\quad \lvert\delta m_t\rvert\ge m_t\ (\delta m_t<0).$$
With $\delta m_t(u)=r\,g_t(u)$ and $g_t(u)=\lvert\delta m_t(u)\rvert/r$,
$$r_c^-(t,u)=\frac{m_t}{g_t(u)}.$$
If the local sensitivity at the controlling step were proportional to the global gain, $g_{t_*}(u)=\beta_{t_*}\,G(u)$, then
$$r_*(u)=\frac{m_{t_*(u)}}{\beta_{t_*(u)}\,G(u)},$$
so $r_*G=m_{t_*}/\beta_{t_*}$ should be constant **within** a group of directions sharing $t_*$, and should change across groups whenever $m_{t_*}$ or $\beta_{t_*}$ changes. This is the correct statement of the “$r_*\sim m/G$” intuition; the naive version with one global $m$ assumes that every direction is controlled by the same tie, which is false.

**Global correlation.** Over 834 directions,
$$\mathrm{Pearson}\bigl(\log r_*,\log G\bigr)=-0.176\quad(p=3.1\times10^{-7}),\qquad
\mathrm{Spearman}=-0.132.$$
So the raw correlation is weak, negative, and statistically detectable only because $n$ is large; $G$ explains about 3% of the variance in $r_*$. By contrast, $\mathrm{Pearson}(\log r_*,\log m_{t_*})=+0.742$ ($r^2=0.551$). A multiple regression
$$\log r_*=2.863+0.695\log m_{t_*}-0.687\log G$$
has $R^2=0.610$: the margin of the controlling tie carries most of the explanatory power, and $G$ enters with a negative coefficient of the expected order ($-0.69$).

**Within-group behavior.** Inside a fixed controlling step, the slope of $\log r_*$ versus $\log G$ should be near $-1$ if $g_{t_*}\propto G$. It is:

| $t_*$ | $n$ | Pearson$(\log r_*,\log G)$ | slope | median $r_*G$ | p25–p75 of $r_*G$ | median $g_{t_*}/G$ |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 503 | $-0.365$ ($p=2.9\times10^{-17}$) | $-0.806$ | 0.0222 | 0.0128–0.0446 | $4.37\times10^{-3}$ |
| 19 | 236 | $-0.302$ ($p=2.3\times10^{-6}$) | $-0.684$ | 0.1247 | 0.0718–0.2217 | $9.18\times10^{-3}$ |
| 189 | 146 | $-0.545$ ($p=1.2\times10^{-12}$) | $-0.978$ | 0.4250 | 0.3000–0.6926 | $5.23\times10^{-3}$ |
| 273 | 56 | $-0.591$ ($p=1.7\times10^{-6}$) | $-1.213$ | 1.7771 | 1.039–3.222 | $1.08\times10^{-2}$ |
| 247 | 15 | $-0.474$ ($p=0.074$) | $-2.447$ | 3.5006 | 2.442–4.416 | $3.20\times10^{-2}$ |
| 255 | 8 | $-0.515$ ($p=0.191$) | $-0.470$ | 4.0585 | 3.349–5.380 | $1.29\times10^{-2}$ |
| 98 | 8 | $-0.317$ ($p=0.444$) | $-0.627$ | 9.2162 | 5.638–10.508 | $5.22\times10^{-2}$ |
| 0 | 5 | $-0.759$ ($p=0.137$) | $-1.370$ | 7.9259 | 6.215–8.974 | $1.27\times10^{-2}$ |
| 171 | 5 | $-0.823$ ($p=0.087$) | $-1.670$ | 12.5128 | 8.392–14.142 | $4.31\times10^{-3}$ |

Within the four large groups the slope is between $-0.68$ and $-1.21$, i.e. close to the predicted $-1$; the correlation is weak but real because the local sensitivity $g_{t_*}$ has its own spread at fixed $G$. Across groups the median $r_*G$ goes from 0.0222 to 12.51, a factor **563**. That is the collapse failure: the product is not a universal constant. The naive $m_{\min}/G$ line is below the data by a median factor of **513**.

**The product distribution.** Over 834 directions, $r_*G$ has p5 $=0.00826$, p25 $=0.0184$, median $=0.0499$, p75 $=0.152$, p95 $=0.480$; log-standard-deviation 1.295, coefficient of variation 1.43, p95/p5 $=58$. This is not a constant.

**Where the extra factor comes from.** The local sensitivity at the controlling step is much smaller than the median state gain:
$$g_{t_*}(u)=\frac{\lvert\delta m_{t_*}(u)\rvert}{r},\qquad
\frac{g_{t_*}}{G}: \text{median }5.996\times10^{-3}\ (p5\ 9.13\times10^{-4},\ p95\ 2.50\times10^{-2}).$$
The median local sensitivity is $3.80\times10^{-3}$ nats per unit radius, about 0.6% of the median $G=0.583$. The correlation between $\log g_{t_*}$ and $\log G$ is only $+0.317$ (slope 0.77), so a strong direction is somewhat more effective at the tie, but only weakly. The reason is geometric: $G$ measures the norm of the full 5120-dimensional response, while $\delta m_t$ measures its projection onto one particular direction $W_{[\text{top1}_t]}-W_{[\text{top2}_t]}$. A direction can be strong overall and nearly orthogonal to that logit-difference vector, or weak overall and well aligned with it. The near-tie step $t=21$ is precisely such a case: its local gain is $2.61\times10^{-3}$, one of the smallest of the ten near-ties.

**Why this quantity.** The product $r_*G$ is the natural test of the hypothesis that “one scalar gain controls the threshold.” It is not a normalization chosen to make the data collapse; it is what the algebra predicts if $\delta m_t\propto G$. Showing that it fails by 563× across ties, while working approximately inside a tie, is the cleanest way to separate the two effects.

**What the figure shows.** (i) Within a tie, stronger directions do have smaller thresholds, with a slope near $-1$. (ii) Across ties, the margin $m_{t_*}$ dominates: a direction controlled by the $9.7\times10^{-5}$ tie has a threshold $\sim500$ times smaller than a direction of the same gain controlled by the $5\times10^{-2}$ tie. (iii) A single global $r_*\sim m/G$ law with one $m$ is wrong by two to three orders of magnitude.

**What it does not show.** It does not prove that the local sensitivity is unrelated to $G$ at every step; it shows the relation is weak at the controlling steps of this run. It does not include the 166 directions with $r_*>1$, where the linear extrapolation is farthest from the measured radius. And the within-group slopes are themselves uncertain for the small groups ($n\le15$), where the $p$-values are not significant.

---

## 7. What is established, and what is a limitation

**Direct findings.**
1. At $r=0.1$, 464/1000 directions flip some token; all 464 are controlled by $t=21$ (421) or $t=19$ (43). No other step produces a flip at this radius.
2. The three smallest-gap steps account for 88.5% of all controlling assignments (503 + 236 + 146 of 1000).
3. The signed response at a near-tie is almost symmetrically split: 52.9% destroy / 47.1% widen at $t=21$, 49.1%/50.9% at $t=19$, 50.7%/49.3% at $t=189$.
4. The responses at $t=21$ and $t=189$ are nearly independent across directions (Pearson $-0.142$; quadrants 245/284/262/209). There is no single “danger direction” shared by the two ties.
5. The state-response gain spans 13.7× (0.224–3.064, median 0.583), and the 24 stored response vectors are nearly orthogonal (mean cosine $+0.0025$; top PC 6.5%). The geometry is not 1-D.
6. The response grows 8.43× from state 6 to state 47 in absolute terms, but the relative response falls 2.47× because the baseline state norm grows 20.7×. State 0 is exactly zero.
7. Inside a controlling-step group, $\log r_*$ vs $\log G$ has slope $-0.68$ to $-1.21$; across groups the product $r_*G$ spans 563×. The controlling margin, not $G$, is the leading determinant of $r_*$ ($r^2=0.551$ alone; 0.610 with $G$).
8. The local logit-gap sensitivity at the controlling step is only $\sim0.6\%$ of the median state gain (median $g_{t_*}/G=6.0\times10^{-3}$), i.e. the decision direction is close to orthogonal to the typical response.

**Limitations of this dump, stated explicitly.**
- The engine's stored `r_c`/`pred_T` use $\lvert\delta m\rvert$ and overcount flips (810 vs 464). Use the signed recomputation.
- The response angle relative to the baseline state cannot be computed: only baseline norms and the state cosine were stored. The state cosine is 1 at float32 precision.
- $G$ uses state 47 (output of block 46), not the final-norm input that actually feeds the logits. For the 24 directions where both exist, $G_{47}/G_{\rm final}=4.06$ (p5 3.74, p95 4.20) with log-correlation 0.996; the proxy is a nearly constant rescaling, which is adequate for rankings and for the $r_*$ regression, but its absolute value should not be compared to final-norm numbers.
- Only one radius ($r=0.1$) was run. The threshold formula extrapolates linearly; the calibration run showed proportionality in $r\in[0.03,0.3]$, so the extrapolation to $r\sim0.002$–$0.05$ is an assumption, not a measurement.
- Teacher forcing holds the continuation fixed. The measured $\delta m_t$ is the response at each baseline step, before any divergence. It does not include the feedback of an already-flipped token.
- The 24 response vectors cover directions 0–23 only (20 from block 0, 4 from block 1). The pairwise-cosine result is a 24-dimensional sample of a 1000-dimensional ensemble.
- The window is 100 tokens, so after $t\approx35$ the prompt is progressively outside the attention window; late-time responses are propagated through generated-token states. The layer curves mix “depth growth” with “prompt leaving the window.”

All numbers in this section are in `notebook_big/figures/ds14_a1_deep_numbers.json`; the figures are produced by `scripts_spark/figs_a1_deep.py` from `results_big/ds14/a1_inter_full/a1.npz` and `meta.json`.
