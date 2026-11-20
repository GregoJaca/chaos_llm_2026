# Part cross-scale. DeepSeek-R1-Distill-Qwen-1.5B vs -14B under one protocol

## Summary

Two models of the same family — 28 layers with $d=1536$, and 48 layers with $d=5120$, roughly eight times the parameters — generate 300 greedy tokens from the identical 65-token prompt with an attention window of 100, and then the same kind of prompt-embedding perturbation (a unit direction times $r=0.1$) is applied and the continuation is teacher-forced. The larger model's decisions are far more fragile: its median top-1/top-2 logit gap is 3.10 nats against 4.78 for the small one, and its thinnest gap (9.7e-5 nats at step 21) is 207 times thinner than the small model's thinnest (2.0e-2 nats at step 123). As a consequence the radius at which a random direction first flips a token — a "flux-tube radius", the largest perturbation size along that direction that leaves the greedy continuation intact — is 13 times smaller in the 14B (median 0.116 vs 1.55 in absolute embedding units, 17 times smaller after dividing by the token-embedding scale), and 464 of 1000 directions flip something at the run radius against 0 of 20. The response itself is *not* stronger in the bigger model: per unit radius its last-state displacement is 2.3 times smaller in absolute units and 4.6 times smaller relative to the state it lives in, so the fragility is a property of how close the decisions are, not of how hard the perturbation pushes. Two shape features do differ: only the 1.5B shows a window-flush surge at $t=35$ (its last state jumps 1.70x across it, the 14B's 1.11x), and the 1.5B's response decays to 19% of its early level by $t=200$ while the 14B's is still at 68% — though that second difference is partly textual, because the small model falls into a repeating four-token loop at $t=143$ and the large one does not.

---

## 1. The two runs and every quantity

### 1.1 Setup

| item | 1.5B | 14B |
|---|---|---|
| model | `DeepSeek-R1-Distill-Qwen-1.5B` | `DeepSeek-R1-Distill-Qwen-14B` |
| blocks / hidden size | 28 / 1536 | 48 / 5120 |
| precision | FP32 | FP32 |
| prompt | `interstellar_travel`, "Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. …" | same, **token-identical** (`prompt_ids` compare equal) |
| prompt length | $P=65$ | $P=65$ |
| attention window | $W=100$ | $W=100$ |
| continuation | $T=300$ greedy tokens, model's own | $T=300$ greedy tokens, model's own |
| decoding | greedy, no sampling, no EOS stop | same |
| perturbation | prompt embeddings only, unit direction $u$ times $r$, 20 non-zero coordinates | same construction |
| directions | $K=20$, **one** simplex block of 20 | $K=1000$, 50 independent blocks of 20 |
| radius used | $r=0.1$ (index 4 of `radii=[1e-3 … 2]`) | $r=0.1$ (the only radius) |
| teacher forcing | all $K$ perturbed passes reuse baseline `prompt_ids`+`gen_ids` | same |
| saved states | layer indices `[0,1,2,3,7,14,21,27]` (state 0 = embedding, 27 = input to final norm) | `[0,6,12,18,24,30,36,42,47]` (47 = input to final norm) |
| `dh_last` vectors | 3 directions (0–2), 1536-d, last state | 24 directions (0–23), 5120-d, last state |
| file | `chaos_llm_2026/results/local_data/w100_layers/a1.npz` (read-only) | `results_big/ds14/a1_inter_w100/a1.npz` + `meta.json` |

Two things the table makes explicit. (i) The **prompt is identical but the continuation is not**: the two models generate different tokens, so "$t=21$" is the 22nd token of a different text in each run. Comparable across models are the protocol, the *distribution* of margins over steps, and the flush step $t^*=W-P=35$ (which follows from the mask, not from the tokens). (ii) The **direction sets are not equally rich**: the 1.5B samples one 20-dimensional input subspace, the 14B samples fifty, and cross-block directions in the 14B are exactly orthogonal while within-block pairs have inner product $-1/19$. An ECDF over 20 directions is a 20-step staircase; an ECDF over 1000 is smooth. That is a sampling-precision difference, not a bias, but it is why the 1.5B curves look coarse.

*Note on a quoted number.* The analysis brief quotes a 14B median margin of 1.68 nats. The `margins` array in this npz has median **3.096** (mean 4.996, p10 0.435, min $9.727\times10^{-5}$). I could not reproduce 1.68 from the array (median of the 100 smallest margins is 0.686, of the 20 smallest 0.116, of steps $t<100$ is 4.558). Everything below uses 3.096, which is also the value used in `section_a1_deep.md`. The quoted minima (9.7e-5 at $t=21$, and 2.0e-2 at $t=123$ for the 1.5B) reproduce exactly.

### 1.2 Quantities

Notation: $u$ indexes a direction, $t\in\{0,\dots,299\}$ a generated step, $\ell^{(1)}_t,\ell^{(2)}_t$ the top two logits of the unperturbed run, $h_t^{\ell}$ the unperturbed state at saved index $\ell$, $\delta h_t^{\ell}$ the same state under the perturbation.

| symbol | explicit expression | meaning | why this quantity |
|---|---|---|---|
| $m_t$ | $\ell^{(1)}_t-\ell^{(2)}_t$ | top-1 minus top-2 logit gap of the unperturbed continuation (nats) | the only scalar that says how close the baseline decision is; a flip needs this gap closed |
| $\delta m_t(u)$ | $m_t(u)-m_t$ at the perturbed pass, stored `dm` (nats) | change of that gap under direction $u$ at radius $r$ | response of the decision variable, measured without letting the text diverge |
| $r^-_c(t,u)$ | $m_t\,r/\max(-\delta m_t(u),0)$, $+\infty$ if $\delta m_t\ge0$ | radius at which step $t$ would flip under direction $u$ | solving $-\delta m_t\ge m_t$ for the radius, and only $\delta m_t<0$ can close the gap |
| $r_*(u)$ | $\min_t r^-_c(t,u)$ | **flux-tube radius**: largest $r$ along $u$ that keeps every baseline argmax | one number per direction, comparable across models and radii (linearity) |
| $t_*(u)$ | $\arg\min_t r^-_c(t,u)$ | controlling step | says which near-tie is the weak point |
| $s_{\rm emb}$ | $\mathrm{median}_t\lVert h_t^{0}\rVert$ | median token-embedding norm (layer-0 state norm) | converts $r$ into "fraction of a token vector"; the perturbation is added at this layer |
| $s_{\rm last}$ | $\mathrm{median}_t\lVert h_t^{\ell_{\max}}\rVert$ | median norm of the last saved state | what the network actually carries at the read-out depth |
| $G(u)$ | $\mathrm{median}_t\lVert\delta h_t^{\ell_{\max}}\rVert/r$ | absolute per-direction gain (state units per unit radius) | robust scalar response strength |
| $G_{\rm rel}(u)$ | $\mathrm{median}_t\lVert\delta h_t^{\ell_{\max}}\rVert/\lVert h_t^{\ell_{\max}}\rVert$ | relative gain (dimensionless) | the only gain comparable across models with different stream norms |
| $\rho(u)$ | $G(u)/s_{\rm emb}$ | gain per unit radius in embedding-scale units | second normalization, used as a robustness check |

Both $r_*$ and $G$ are computed from the **signed** threshold ($\delta m<0$ only). The npz's own `r_c` array was built with $|\delta m|$; using it would call 810/1000 directions flips in the 14B instead of 464. The recomputation is three lines and is the difference between a physically meaningful threshold and a symmetric one.

### 1.3 What is, and is not, comparable across scales

This is the part where a cross-scale comparison can quietly go wrong, so it is worth being explicit.

1. **$r$ itself is comparable by construction.** A direction is a unit vector in $\mathbb{R}^{65\times d}$ with 20 non-zero entries; the perturbation is $r\,u$ scattered into the prompt embedding matrix. Its Euclidean norm is exactly $r$ in both runs, and it is spread over the same number of coordinates (20). So $r=0.1$ is the same *absolute* perturbation in both models. Raw $r_*$ is therefore already a meaningful comparison: "the perturbation size at which the greedy continuation first breaks".
2. **But the embedding scale is not identical.** $s_{\rm emb}=1.206$ (1.5B) vs $1.516$ (14B): the 14B's token vectors are 26% longer, so $r=0.1$ is $0.083\,s_{\rm emb}$ in the 1.5B and $0.066\,s_{\rm emb}$ in the 14B. If anything this makes the raw comparison *conservative* — the 14B is being hit relatively more softly. Dividing by $s_{\rm emb}$ widens the gap from 13x to 17x. Both numbers are reported; neither changes the conclusion, which is the useful outcome of a normalization check.
3. **Last-layer norms are not comparable and must not be used to normalize $r$.** $s_{\rm last}$ is 545 (1.5B) and 1067 (14B) — the 14B residual stream carries 2.0x more norm — and this ratio is a property of depth, width, and RMSNorm gains, not of the perturbation's entry point. Normalizing the *threshold radius* by $s_{\rm last}$ would therefore mix a decision property with a representation-scale property. Normalizing the *response* by it is exactly right, because the question there is "how much of what the layer carries is displaced".
4. **Raw response norms are not comparable for the same reason.** $\lVert\delta h\rVert$ at the read-out depth is measured in a space whose typical vector is twice as long in the 14B. So $G$ is reported next to $G_{\rm rel}$ and never alone.

Concretely: $r_*/s_{\rm emb}$ is the normalized *input* comparison (used in Fig. `xscale_rstar`), $G_{\rm rel}$ is the normalized *output* comparison (used in Fig. `xscale_gain`). Cross-normalizing (radius by the last-layer norm, gain by the embedding norm) is possible but answers a different and less interpretable question; $\rho=G/s_{\rm emb}$ is included in the JSON as the check that this choice does not drive the result (median $\rho$: 1.105 vs 0.385, i.e. 2.9x, between the 2.3x of raw $G$ and the 4.6x of $G_{\rm rel}$).

### 1.4 The comparison, in one table

| quantity | 1.5B | 14B | ratio (1.5B / 14B) |
|---|---:|---:|---:|
| $K$ (directions), radius | 20, $r=0.1$ | 1000, $r=0.1$ | — |
| median margin $m_t$ | 4.778 nats | 3.096 nats | 1.54 |
| smallest margin | $2.016\times10^{-2}$ (t=123) | $9.727\times10^{-5}$ (t=21) | 207 |
| steps with $m_t<0.1$ nats | 6 / 300 | 8 / 300 | — |
| controlling step (mode) | $t=116$ (6/20) | $t=21$ (503/1000) | — |
| directions controlled by the top-3 steps | 75% | 88.5% | — |
| $r_*$ median, raw (embedding units) | 1.555 | 0.1164 | 13.4 |
| $r_*$ p10 / p90, raw | 0.486 / 4.12 | 0.0171 / 2.59 | — |
| $s_{\rm emb}$ | 1.206 | 1.516 | 0.80 |
| $r_*$ median, $/s_{\rm emb}$ | 1.289 | 0.0768 | 16.8 |
| $r_*$ p10 / p90, $/s_{\rm emb}$ | 0.403 / 3.42 | 0.0113 / 1.71 | — |
| flips at $r=0.1$ | 0 / 20 | 464 / 1000 | — |
| flips at $r=0.3$ / $r=1$ | 1 / 4 | 671 / 834 | — |
| smallest $r_*$ | 0.133 (dir 10) | $2.44\times10^{-3}$ (dir 63) | 54 |
| $G$ median, raw | 1.333 | 0.583 | 2.29 |
| $G$ p10 / p90 | 0.922 / 2.24 | 0.409 / 1.30 | — |
| $s_{\rm last}$ | 545.3 | 1067.3 | 0.51 |
| $G_{\rm rel}$ median | $2.535\times10^{-4}$ | $5.567\times10^{-5}$ | 4.55 |
| $\rho=G/s_{\rm emb}$ median | 1.105 | 0.385 | 2.87 |
| fp32 response floor, last state | $1.0\times10^{-3}$ | $2.7\times10^{-3}$ | — |
| gain floor $\text{floor}/r$ | 0.010 | 0.027 | — |
| measured $G$ / gain floor | 133x | 22x | — |
| clean $r$ window | $[10^{-3},0.1]$ | $[0.03,0.3]$ | — |
| flush ratio, last state ($\overline{\lVert\delta h\rVert/r}_{36..40}$ over $\overline{\lVert\delta h\rVert/r}_{30..34}$) | 1.70 | 1.11 | — |
| late/early response ratio ($t\in[200,300)$ vs $[0,50)$) | 0.186 | 0.680 | 0.27 |

The fp32 floor is a property of the response measurement, not of this analysis: at the last state a perturbed forward pass at the cleanest radius reproduces the baseline only to $\approx10^{-3}$ (1.5B) / $2.7\times10^{-3}$ (14B) in $\lVert\delta h\rVert$. Both medians sit 22–133x above that floor, so no number here is round-off. The floor does, however, set the smallest meaningful gain, which is why the noisy end of the $r$ ladders is excluded from the table above (clean windows quoted).

---

## 2. Figure `xscale_margins.png` — the 14B's weakest tie is 207x thinner

![margins](figures/xscale_margins.png)

**Setup.** Both runs of §1.1, unperturbed pass only ($r=0$): 300 top-1/top-2 gaps per model. Top row: $m_t$ against $t$. Bottom row: the ECDF of the same 300 values, log $x$. In each bottom panel the own model is bold and the other model is the pale dashed reference. The dotted vertical line at $t=35$ is the flush step $W-P$; the dotted line at 0.1 nats in the ECDF is only a reading aid.

**Axes.**
- (a),(b) $x$: generated step $t$, 0 = first generated token; $y$: $m_t$ in nats, log scale, range $0.6\min$ to $8\max$ (free per panel). Open circles: the five smallest gaps of that model; the number next to each circle is its rank in the table below/inside the panel.
- (c),(d) $x$: $m_t$ in nats, log scale; $y$: ECDF $P(m_t\le x)$, linear 0–1. The inset box repeats the five smallest gaps with their steps.

**The five smallest gaps** (the shared table for this figure):

| rank | 1.5B $t$ | 1.5B $m_t$ (nats) | 14B $t$ | 14B $m_t$ (nats) |
|---:|---:|---:|---:|---:|
| 1 | 123 | $2.016\times10^{-2}$ | 21 | $9.727\times10^{-5}$ |
| 2 | 143 | $2.041\times10^{-2}$ | 19 | $1.144\times10^{-3}$ |
| 3 | 116 | $6.902\times10^{-2}$ | 189 | $2.224\times10^{-3}$ |
| 4 | 2 | $6.959\times10^{-2}$ | 273 | $1.916\times10^{-2}$ |
| 5 | 147 | $7.310\times10^{-2}$ | 255 | $5.151\times10^{-2}$ |

**What is in the figure.**
- The two margin *distributions* are similar in centre but not identical in shape: median 4.778 vs 3.096 nats, p10 0.628 vs 0.435, and both have 6–8 steps below 0.1 nats. Their ECDFs differ by at most 0.217 (at $m\approx3.1$ nats) and cross at 6.7 nats: below that value the 14B accumulates slightly more mass, above it the 1.5B does.
- The difference is entirely in the extreme lower tail. The 1.5B's five smallest values are all in $[0.020,0.073]$ — a floor of two hundredths of a nat. The 14B reaches $9.7\times10^{-5}$, then $1.1\times10^{-3}$, then $2.2\times10^{-3}$: three steps more than an order of magnitude below the 1.5B's floor.
- The 1.5B's weakest steps cluster at $t=116$–$147$ (four of the five within 31 steps); the fifth is at $t=2$. The 14B's are spread over $t=19$–$273$.
- The zig-zag visible in both curves after $t\approx150$ is the text, not noise. The 1.5B's continuation is the exact 4-token cycle `1576, 807, 525, 11050` from $t=143$ to the end, so its $m_t$ repeats with period 4 to within 2.2 nats; the 14B's 300 tokens contain no such cycle (the best lag, $p=9$, matches 2.4% of positions — chance). So the 1.5B's late-time statistics describe a degenerate loop, the 14B's describe fresh text.

**What it shows.** The 14B does not merely have slightly tighter decisions; it has a genuinely thinner *floor*, by two orders of magnitude. That single fact, not the shape of the distribution, is what drives every other cross-scale number in this section.

**What it does not show.** It does not show that thin margins are *more common* in the 14B — they are not, at the 0.1-nat level (8 vs 6 of 300 steps). It says nothing about whether these margins are *stable*: a token that is nearly tied at one prefix can be comfortable at another, and only one prefix is measured per model. And it cannot separate "bigger model" from "different text": the 1.5B's continuation is not the 14B's, and margins are a property of the prefix as much as of the model.

---

## 3. Figure `xscale_rstar.png` — threshold radii, raw and normalized

![rstar](figures/xscale_rstar.png)

**Setup.** Both runs, $r=0.1$. $r_*(u)=\min_t m_t r/\max(-\delta m_t(u),0)$ per direction, ECDF over directions. Panel (a) plots $r_*$ in absolute embedding units; panel (b) plots $r_*/s_{\rm emb}$ with $s_{\rm emb}=\mathrm{median}_t\lVert h_t^{0}\rVert$ = 1.206 (1.5B) and 1.516 (14B). Bottom row: the same data as a log-log survival function $\hat P(r_*>x)=1-\hat F(x)$ with the $+1/K$ correction, which is the only way to see the tail with 20 samples. Short vertical ticks on the ECDFs mark p10/p50/p90; dashed vertical lines mark the run radius $r=0.1$ in the same units as the panel (0.083 and 0.066 in (b)).

**Axes.**
- (a),(b) $x$: $r_*$ (embedding units) or $r_*/s_{\rm emb}$ (dimensionless), log scale; $y$: ECDF, 0–1.
- (c),(d) $x$: same as above, log; $y$: $P(r_*>x)$, log.

**Numbers.**

| | 1.5B | 14B |
|---|---:|---:|
| $r_*$ p10 / p50 / p90 (raw) | 0.486 / 1.555 / 4.124 | $1.713\times10^{-2}$ / 0.1164 / 2.592 |
| $r_*$ min / max (raw) | 0.133 / 5.53 | $2.44\times10^{-3}$ / 55.5 |
| $r_*/s_{\rm emb}$ p10 / p50 / p90 | 0.403 / 1.289 / 3.418 | $1.13\times10^{-2}$ / $7.68\times10^{-2}$ / 1.710 |
| flips at $r=0.1$ | 0 / 20 | 464 / 1000 |
| flips at $r=0.3$ / $r=1$ / $r=2$ | 1 / 4 / 14 | 671 / 834 / 886 |

**What is in the figure.**
- The raw medians differ by 13.4x; after dividing by $s_{\rm emb}$ by 16.8x. The normalization *increases* the gap, because the 14B's embedding vectors are 26% longer and the same absolute $r$ is therefore a relatively smaller kick.
- The bulk of the 1.5B's distribution spans less than a decade (p10–p90: 0.486–4.12) while the 14B's spans 2.2 decades ($1.7\times10^{-2}$–2.59), and the 14B has an even heavier upper tail: its survival curve stays above $10^{-2}$ out to $r\approx10$, i.e. about ten directions need more than $10r$, with a maximum of 55.5. Those are directions whose perturbation *widens* every near-tie and therefore never breaks the continuation at any reasonable radius. In this npz all 1000 $r_*$ are finite, so none is literally infinite; the 536 directions above the run radius are simply not flipped at $r=0.1$.
- The 1.5B's minimum, 0.133, belongs to direction 10 — the same direction that is the most destructive at $t=123$ (§6). It is 1.33x above the run radius, which is why the run has zero flips despite a 0.02-nat tie.
- The 14B's smallest, $2.4\times10^{-3}$, is 41x *below* the run radius; 464 directions are below it.

**What it shows.** The threshold radius — the width of the tube of perturbations that leaves the continuation intact — is an order of magnitude narrower in the 14B, and the gap survives the normalization check (it widens slightly rather than closing).

**What it does not show.** With $K=20$ the 1.5B ECDF is a 20-step staircase: its p10 and p90 are each one sample away from the next stair, and the "min" is the minimum of 20 draws, not a population minimum. The 1.5B's max (5.53) is also not a real upper bound — 20 samples cannot see a 55-unit tail even if one exists. And $r_*$ is computed from teacher-forced responses: it is the radius at which the *first* token would flip while the prefix is still the baseline one, not the radius at which the free-running text changes.

---

## 4. Figure `xscale_gain.png` — the 14B responds *less*, absolutely and relatively

![gain](figures/xscale_gain.png)

**Setup.** Both runs, $r=0.1$, last saved state (index 27 for the 1.5B, 47 for the 14B — both the input to the final norm, i.e. the penultimate exit, matched relative depth 27/28 vs 47/48). $G(u)=\mathrm{median}_t\lVert\delta h_t^{\rm last}\rVert/r$; $G_{\rm rel}(u)=\mathrm{median}_t\lVert\delta h_t^{\rm last}\rVert/\lVert h_t^{\rm last}\rVert$ at the run radius (dimensionless, no division by $r$). ECDFs over directions, log $x$. Dotted vertical lines: the fp32 floor expressed in the same units, $\text{floor}/r$ = 0.010 (1.5B) and 0.027 (14B) in (a), and $\text{floor}/s_{\rm last}$ = $1.8\times10^{-6}$ / $2.5\times10^{-6}$ in (b).

**Axes.**
- (a) $x$: $G$ in state units per unit $r$, log; $y$: ECDF.
- (b) $x$: $G_{\rm rel}$, dimensionless, log; $y$: ECDF.

**Numbers.**

| | 1.5B | 14B |
|---|---:|---:|
| $G$ min / p10 / p50 / p90 / max | 0.635 / 0.922 / 1.333 / 2.235 / 7.47 | 0.224 / 0.409 / 0.583 / 1.297 / 3.06 |
| $G_{\rm rel}$ min / p10 / p50 / p90 / max | $1.17\times10^{-4}$ / $1.68\times10^{-4}$ / $2.54\times10^{-4}$ / $4.22\times10^{-4}$ / $1.45\times10^{-3}$ | $2.12\times10^{-5}$ / $3.89\times10^{-5}$ / $5.57\times10^{-5}$ / $1.24\times10^{-4}$ / $2.95\times10^{-4}$ |
| $s_{\rm last}$ | 545.3 | 1067.3 |
| median above fp32 gain floor | 133x | 22x |

**What is in the figure.**
- The two raw ECDFs are separated by a factor 2.3 at the median and do not overlap except in the upper tail (the 1.5B's p10, 0.922, is above the 14B's p75, 0.877). The 1.5B also has a single direction at $G=7.47$, 5.6x its own median — a heavy right tail that the 14B does not have (max 3.06, 5.2x its median).
- In relative units the median separation grows to 4.6x. The bulk is separated — the 1.5B's p10 ($1.68\times10^{-4}$) is 3.0x the 14B's median and 1.36x its p90 — but the tails are not: 12% of the 14B's directions have a larger relative gain than the 1.5B's weakest direction ($1.17\times10^{-4}$), and the 14B maximum ($2.95\times10^{-4}$) is 2.5x that minimum. So the ordering is a shift of the distribution, not a complete separation.
- Both are far above the fp32 floor, so the ordering is not a precision artifact: 133x and 22x above the respective gain floors.

**What it shows.** The bigger model is the *less* responsive one per unit perturbation, both in absolute state units and as a fraction of what its residual stream carries. Combined with §3 this localizes the fragility entirely in the decision margins: $r_*\sim m/|\delta m|$ falls 13x because $m$ falls 207x at the weakest step while the response falls only 2.3x.

**What it does not show.** It does not show that the 14B is "less sensitive" in any behavioral sense — a smaller relative displacement can still flip a decision if the decision is nearly tied, which is exactly what happens. $G$ is a median over 300 steps and hides the time structure (that is Fig. `xscale_shape`). And the normalizations are not interchangeable: dividing by $s_{\rm last}$ answers "what fraction of the carried state", dividing by $s_{\rm emb}$ answers "how many token vectors' worth"; they give 4.6x and 2.9x respectively, and both are reported in the JSON.

---

## 5. Figure `xscale_shape.png` — same depth ordering, different persistence, and a flush only in the small model

![shape](figures/xscale_shape.png)

**Setup.** Both runs, $r=0.1$, three matched relative depths: early (1.5B L3 / 14B L6, 11% / 13% deep), middle (L14 / L24, 52% / 51%), last saved state (L27 / L47, the penultimate exit in both). Curves are $\mathrm{median}_u\lVert\delta h_t^{l}\rVert/r$ at each $t$, each divided by its own median over $t$ (300 values), so they are dimensionless and their absolute levels are not comparable — only their shapes are. The dotted vertical line is $t^*=W-P=100-65=35$; the annotated ratio is $\bar m_{36..40}/\bar m_{30..34}$, i.e. the mean level just after the flush over the mean level just before it. The horizontal grey line is 1 (the curve's own median).

**Axes.** $x$: generated step $t$; $y$: $\mathrm{median}_u\lVert\delta h_t^{l}\rVert/r$ divided by its median over $t$ (dimensionless), log scale, free per panel.

**Numbers.**

| depth | late/early (t 200–299 vs 0–49) | peak / late | flush ratio 36–40 / 30–34 | step t=34→35 |
|---|---:|---:|---:|---:|
| 1.5B L3 (11%) | 0.0024 | 643 | 0.96 | 1.15 |
| 1.5B L14 (52%) | 0.169 | 15.0 | 0.78 | 1.00 |
| 1.5B L27 (last) | 0.186 | 18.4 | **1.70** | **2.06** |
| 14B L6 (13%) | 0.286 | 2.66 | 1.00 | 1.16 |
| 14B L24 (51%) | 0.420 | 5.95 | 1.14 | 1.22 |
| 14B L47 (last) | 0.680 | 3.17 | **1.11** | 0.80 |

**What is in the figure.**
- **Early depth.** Both curves decline monotonically through the window, but by very different amounts. The 1.5B's normalized response falls from 38 at $t=0$ to 1 at $t=148$, to 0.04 at $t=250$, and is exactly 0 from $t=290$ on (its flush cascade reaches layer 3). The 14B's falls from 4.0 at $t=0$ to 1 at $t=123$ and then flattens at 0.7–0.9 — it never decays away.
- **Middle depth.** Both show the same qualitative object: a broad hump (1.5B peak at $t=84$, 14B at $t=100$), then decay. The 1.5B falls 14x below its hump by $t=250$; the 14B falls 2.9x and then flattens.
- **Last state.** The 1.5B has the clearest structure: a rise to a peak at $t=85$ (8.5 in units of $r$, 18x its late level) and a monotone decay afterwards. The 14B's last-state curve is a noisy plateau with excursions (a $3.15\,r$ spike at $t=258$) and no hump: peak/late is 3.2 versus 18.4.
- **The flush at $t=35$.** The 1.5B's last state jumps by 2.06x in the single step $t=34\to35$ and 1.70x in the 5-step means, then keeps rising to its $t=85$ peak. The 14B's last state does not: the single-step ratio is 0.80 (a *decrease*) and the 5-step ratio is 1.11. Across all saved layers the 1.5B shows the surge only at L21 (1.38) and L27 (1.70) and a small *dip* at L1–L14 (0.73–0.96); the 14B never exceeds 1.14 at any saved layer.
- The zero-response layer is L0 in both (100% of entries exactly 0): the perturbation is on prompt positions and the read positions are generated tokens, so at the embedding layer there is nothing to see.

**What it shows.** The response has the same *architecture* in both models (zero at the embedding, growing with depth, decaying with $t$), but two scale-dependent differences are visible: the large model's response persists over the whole 300-step window instead of collapsing, and it does not show the window-flush surge that the small model shows at $t=35$. The flush is a mask event and its step is identical in both protocols, so the *timing* agreement is structural; the *amplitude* disagreement (1.70 vs 1.11) is the empirical content.

**What it does not show.** It does not establish why the surge is absent in the 14B. Two candidates are (i) the response is dominated by directions that are insensitive to the dropped prompt token, and (ii) the 14B's sliding-window mask was not applied in as many layers — Qwen2 applies the window per layer via `max_window_layers`, both runs only set `config.sliding_window=100` and `use_sliding_window=True`, and neither npz records the resulting per-layer mask or the model config. For the 1.5B the mask is independently known to be live (the controlled same-token $W=100$ vs $W=1000$ comparison departs exactly at $t=35$); for the 14B the npz alone cannot distinguish "no mask" from "mask but no surge". This is a real ambiguity, not a rhetorical one, and it should be resolved by dumping `config.sliding_window`, `use_sliding_window`, `max_window_layers` and the `layer_types` of the loaded model on the next 14B run. It also does not compare like-for-like text: the two models generate different tokens — the 1.5B's continuation degenerates into a 4-token loop from $t=143$ while the 14B's never repeats — and a response can decay differently simply because the token stream changes how much prompt content is re-injected. The late-time decay difference (0.19 vs 0.68) is therefore *not* a clean model-size effect.

---

## 6. Figure `xscale_dm.png` — how far the responses sit from the margin they must cross

![dm](figures/xscale_dm.png)

**Setup.** Both runs, $r=0.1$, at each model's **own** smallest-margin step: $t=123$ for the 1.5B ($m=2.016\times10^{-2}$ nats), $t=21$ for the 14B ($m=9.727\times10^{-5}$ nats). Histogram of $\delta m_t(u)/r$ over all directions of that model. Two vertical markers: the red dashed line at $-m_t/r$ (nats per unit radius) is the slope a direction must beat to close the gap *at $r=0.1$* — a direction left of it flips; the grey dash-dot line at $-m_t$ (nats) is the same margin without dividing by $r$, i.e. the threshold only if the run radius were 1.

**Axes.** $x$: $\delta m_t(u)/r$ in nats per unit radius; $y$: number of directions (free per panel). The $x$ ranges differ by a factor 2.8 between panels (1.5B: $-0.22$ to $0.06$; 14B: $-0.045$ to $0.055$).

**Numbers.**

| | 1.5B, $t=123$ | 14B, $t=21$ |
|---|---:|---:|
| $m_t$ | $2.016\times10^{-2}$ nats | $9.727\times10^{-5}$ nats |
| $-m_t/r$ (flip threshold) | $-0.2016$ | $-9.727\times10^{-4}$ |
| $\delta m_t/r$ min / p5 / p50 / p95 / max | $-0.136$ / $-0.0688$ / $+0.0091$ / $+0.0376$ / $+0.0525$ | $-0.0399$ / $-0.0082$ / $-3.13\times10^{-4}$ / $+0.0090$ / $+0.0514$ |
| directions with $\delta m_t<0$ | 7 / 20 | 529 / 1000 |
| directions left of $-m_t/r$ | 0 / 20 | 427 / 1000 |

**What is in the figure.**
- At the 1.5B's weakest step the response distribution is centred slightly *positive* (+0.0091, so 13 of 20 directions widen the margin) and its most destructive direction reaches $-0.136$ — 68% of the $-0.2016$ needed. The whole distribution lives in a window of about $\pm0.15$, and the threshold sits outside it. This is the visual content of "0 of 20 flip".
- At the 14B's weakest step the response distribution is centred essentially on zero ($-3.1\times10^{-4}$, a factor 30 smaller than the 1.5B's offset) and is *wider than the margin by two orders of magnitude*: 42.7% of the mass is left of $-9.7\times10^{-4}$. The threshold line is invisible at this scale because it sits inside the peak — which is the point.
- The two panels are on the same physical axis (nats per unit radius) but different ranges; drawn on one axis the 14B's distribution would be a spike $\sim$6x narrower than the 1.5B's. The ratio of the widths (1.5B p5–p95 ≈ 0.106, 14B ≈ 0.017) is 6.2 — the opposite direction from the threshold ratio (207), which is what makes the 14B's margin crossable.

**What it shows.** The 14B's decisions are not merely near-tied; they are near-tied at a level that a typical direction's response overshoots by two orders of magnitude. The 1.5B's weakest decision is a real tie (0.02 nats) that a typical perturbation simply cannot reach at this radius — the most destructive of 20 directions still comes up 1.5x short.

**What it does not show.** It does not show what happens *after* the first flip (the prefix is still the baseline one here; free-running consequences are a separate experiment). It cannot say whether the 42.7% figure would survive a different prompt, since $t=21$ is one token of one continuation. And the grey line is a deliberate unit mismatch kept for readability of the "margin in nats" — the physically correct comparison is the red one.

---

## 7. Facts that qualify the whole comparison

1. **Different continuations, and one of them is degenerate.** Same prompt, different greedy text. Steps do not correspond across models; only distributions and mask-imposed times do. The 1.5B's continuation is the exact 4-token cycle `1576, 807, 525, 11050` from $t=143$ on; the 14B's never repeats within 300 steps. Every 1.5B statistic that aggregates over late $t$ (the late/early ratio, the shape curves after $t=150$) is therefore measured on a loop, and the 14B's is not.
2. **Different $K$ and different direction geometry.** 20 directions from one 20-d subspace (1.5B) vs 1000 from 50 disjoint ones (14B). Within-block pairs have inner product $-1/19$; cross-block pairs are orthogonal. Any tail statement about the 1.5B rests on 20 samples.
3. **The 14B's flush behaviour is not fully attributable.** The window is set identically in code, but whether it is applied per layer depends on the model config (`max_window_layers`, `layer_types`), which was not recorded. The 1.5B's flush is independently known from a controlled same-token $W=100$ vs $W=1000$ comparison; the 14B's is not.
4. **`dh_last` is not symmetric.** The 1.5B stores 3 directions, the 14B 24. Nothing in this section uses those vectors, so this does not enter any number above — it only limits what a vector-geometry follow-up could do on the 1.5B side.
5. **One radius.** Both runs use $r=0.1$. The 1.5B's usable range is $r\in[10^{-3},0.1]$ and linearity of the response in $r$ was validated there; the 14B's usable range is quoted as $[0.03,0.3]$, but no 14B linearity curve is included in this npz, so reading $r_*$ as a radius at 14B scale rests on that assumption rather than on a 14B measurement.
6. **Margins are a prefix property.** A near-tie at $t=21$ of the 14B says that one particular prefix leaves two tokens nearly tied; it is not a statement about the model's general calibration.

## 8. What follows directly

- Decision fragility scales up with model size *here* by two orders of magnitude at the worst step (9.7e-5 vs 2.0e-2 nats), and this is what makes the threshold radius 13–17x smaller. The per-direction response is 2.3x *weaker* in absolute terms and 4.6x weaker relative to the state, so the effect is on the read-out, not on the propagation.
- The window flush is structurally common ($t^*=35$ in both) but its amplitude is not: 1.70 (1.5B, L27) vs 1.11 (14B, L47) in the 5-step means, and 2.06 vs 0.80 in the single step $t=34\to35$.
- Response persistence over 300 steps is the other clear cross-scale difference: late/early 0.19 (1.5B, last) vs 0.68 (14B, last) — but the 1.5B's number is measured inside a degenerate 4-token loop, so the size of that gap is text-dependent and should not be read as a pure model-size effect.
- Anything that wants to claim "the 14B is more chaotic" has to say *which* quantity: not the response gain ($G$ is smaller), but the distance from the response to the decision threshold ($m_t$ is far smaller), i.e. the fragility is in the read-out geometry.

## 9. Reproduce

```
python scripts_spark/figs_cross_scale.py
```

Writes `notebook_big/figures/xscale_{margins,rstar,gain,shape,dm}.png` (dpi 150) and `notebook_big/figures/xscale_numbers.json`, which contains every quantile quoted above, the per-layer flush ratios, the per-layer late/early ratios, and the p10/p50/p90 of both $r_*$ normalizations and both gains. The 1.5B npz is read in place from the main checkout and is not modified.
