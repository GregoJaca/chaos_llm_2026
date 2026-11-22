# Part synthesis. Four models, one protocol: margins set the bulk, sensitivity sets the tail

## Summary

Four models were run through the same protocol: the same prompt text, 300 greedy tokens, a unit prompt-embedding direction of size $r=0.1$, and a teacher-forced continuation. The models are DeepSeek-R1-Distill-Qwen-1.5B (28 blocks, $d=1536$, $K=20$ directions), DeepSeek-R1-Distill-Qwen-14B (48 blocks, $d=5120$, $K=1000$, **true** sliding window 100), Qwen3.5-9B (32 blocks, $d=4096$, hybrid Gated-DeltaNet/full attention, $K=1000$), and phi-4 (40 blocks, $d=5120$, full attention, $K=1000$). On that protocol the **median flux-tube radius** — the largest perturbation along a direction that leaves every greedy decision intact — is

| model | median $r_*$ | relative to the 14B |
|---|---:|---:|
| DS-R1-14B | **0.1035** | 1 |
| Qwen3.5-9B | **0.7074** | 6.8x |
| DS-R1-1.5B | **1.5547** | 15.0x |
| phi-4 | **3.5865** | 34.7x |

The 14B is the reference point because it is the most fragile model in the campaign; the ordering is unchanged if the older full-attention 14B run is used (median 0.1164 instead of 0.1035, a 12% shift). The end-to-end span is **34.7x in radius**, or 1.54 decades.

**What puts each model there.** Every direction has an exact identity $r_*(u)=m_{t_*(u)}/s_{t_*(u)}$, where $m_t$ is the baseline top-1/top-2 logit gap and $s_t=|\delta m_t|/r$ is the local decision sensitivity at the controlling step. The decomposition has to be done at a *common* direction, because the marginal medians of $m_*$ and $s_*$ are taken over different directions and do not multiply. At the direction that carries the median $r_*$:

| model | $m_*$ at median-$r_*$ direction | $s_*$ at that direction | $r_*=m_*/s_*$ | margin vs 1.5B | sensitivity vs 1.5B |
|---|---:|---:|---:|---:|---:|
| DS-R1-14B | $1.144\times10^{-3}$ | $1.104\times10^{-2}$ | 0.1037 | **17.8x thinner** | 1.18x lower (helps) |
| Qwen3.5-9B | $7.271\times10^{-3}$ | $1.028\times10^{-2}$ | 0.7074 | **2.81x thinner** | 1.27x higher |
| DS-R1-1.5B | $2.041\times10^{-2}$ | $1.301\times10^{-2}$ | 1.5690 | 1 | 1 |
| phi-4 | $6.314\times10^{-2}$ | $1.759\times10^{-2}$ | 3.5899 | **3.09x fatter** | 1.35x higher |

The controlling margins span **55x** across the four models and are **monotone with $r_*$**; the co-selected sensitivities span only **1.71x** and are not monotone. That is the main result: for the bulk of directions, the margin axis places the models, and the response axis is a minor correction. The 14B sits at the bottom because its typical controlling margin is 18x thinner than the 1.5B's and its sensitivity is *lower*, which pushes $r_*$ up rather than down; no response advantage rescues it. The 1.5B sits above Qwen3.5 because its typical margin is 2.8x fatter and its sensitivity is only 1.27x lower, which does not compensate. phi-4 sits on top because its typical margin is 3.1x fatter than the 1.5B's.

The qualification is the tail. phi-4 contains a razor tie ($m=9.25\times10^{-3}$ at $t=277$, 2.2x thinner than the 1.5B's thinnest tie of $2.02\times10^{-2}$, though 95x thicker than the 14B's), yet its *most fragile* direction has $r_*=0.165$, above the 1.5B's most fragile $0.133$. The reason is that the 1.5B's most fragile direction is an unusually loud one ($s=0.903$ nats per unit radius, 16x phi-4's $0.056$) while phi-4's thin ties are quiet. So the median hierarchy is margin-driven, but the lower tail is sensitivity-driven, and the geometric-mean decomposition agrees with the median for the 14B and Qwen3.5 and with the tail for phi-4. The one-sentence version: **margins decide which model is typically fragile; sensitivity decides whether a razor tie is actually dangerous.**

**Stability validation.** The signed linear rule predicts flip *counts* across the whole range but loses the *step identity* as the flip density rises. At $r=0.1$ the true-window 14B flips **483/1000** directions in the free run against **493** predicted; 478 are exact steps, 2 are wrong steps, 13 are false alarms and 3 are misses (98.2% of all rows, 99.0% of observed flips). Qwen3.5 flips 12/1000 against 12 predicted at $r=0.1$ (12 exact) and 157 against 160 at $r=0.3$ (104 exact, 27 wrong steps, 29 false alarms, 26 misses; 91.8% of rows but only 66.2% of observed flips). phi-4 flips 0 at $r=0.1$ (0 predicted) and 4 at $r=0.3$ against 2 predicted; none of the four is predicted at the right step (one wrong step, three misses), and the 99.5% row accuracy is carried by the 995 non-flips. The 1.5B's only genuine free run covers $r\le0.01$ and is a 20/20 null; its $r=0.3$ and $r=1$ entries are teacher-forced decision flips, and at $r=1$ (outside its measured linear window) it flips 8 against 4 predicted.

**Re-synchronisation is a phi-4-only phenomenon in this sample.** Under the stated definition (20 consecutive identical tokens after the flip, or a 20-token identical suffix), **0/1186** flipped 14B rows, **0/169** Qwen3.5 rows, and **2/4** phi-4 rows return to the baseline. The 1.5B free run has no flips to test; the T4 analysis reported no re-synchronisation. Both phi-4 cases are one-token substitutions that re-join at the next token, not slow convergence: direction 276 substitutes at 219, re-joins at 220, substitutes again at 277 and re-joins at 278; direction 933 substitutes at 277 and re-joins at 278.

**Geometry.** The response-vs-state angle is measurable for the first time at 14B, from the true-window run's `h0_last` and 24 `dh_last` vectors: **89.99°** (median; 7200 direction-step pairs), against **90.08°** for Qwen3.5 and **89.85°** for phi-4. The three measured models are all exactly perpendicular within a quarter of a degree, with wide per-pair spread (14B p10-p90 77.7-102.6°). The 1.5B's ~87° is a literature value, not re-measured: that dump stores state norms and $\cos(h',h_0)$, not the baseline vectors. The depth spectrum reproduces the 1.5B's "absolute expansion is radial drift, normalized the tangent contracts" law in both models that were run: 14B $h_{\rm depth}=0.163$–$0.638$ nats with 61/64 six-block chunks contracting, phi-4 $0.154$–$0.859$ with 93/96 intervals contracting, against the 1.5B's $0.6$–$1.4$ band and worst relative exponent $-0.034$; Qwen3.5 has no depth run. Non-normality is only complete for the 14B: $\sigma_{\max}$ is 96.2 at block 0, 3.7-4.9 in the middle and 14.2 at block 47, with $\kappa=7.35,1.6-2.3,7.07$, against the 1.5B's $95$–$195$, $2.7$–$3.6$, $6$–$9$ and $\kappa=5$–$8$, $1.7$–$2.5$, $3$–$5.3$. The last block is the one place where the 14B is clearly more non-normal than the 1.5B. phi-4's non-normality run is still pending.

**Dimension and loops.** TwoNN is 5.44-6.23 for the 1.5B (four validation runs, median 6.17), 4.66-8.90 for the 14B (13 runs, median 8.19) and 6.05-8.09 for Qwen3.5 (12 runs, median 7.50). **No model has a valid scaling window**: 0/2 healthy 1.5B runs, 0/13 14B runs and 0/12 Qwen3.5 runs pass the 0.40-decade rule. The single passing flag in the 1.5B file is a spurious 0.54-decade window on a period-53 loop. Repetition loops fall sharply with model: **9 full + 2 partial of 20** 1.5B trajectories (6 full + 2 partial of 16 at the equal 1000-token horizon), **0 of 13** 14B, **0 full + 1 partial of 12** Qwen3.5. phi-4 has no D2 run.

**The round-off floor is a fixed fraction of the state norm, not a fixed absolute number.** At the smallest tested radius the per-layer response floor $F_l$ divided by the median state norm is $2.19$–$2.88\times10^{-6}$ for the 14B (18-24 fp32 ulp), $1.41$–$3.15\times10^{-6}$ for Qwen3.5 (12-26 ulp), $2.52$–$3.51\times10^{-6}$ for phi-4 (21-29 ulp) and $1.83\times10^{-6}$ for the 1.5B at its last state (15 ulp). Every model's floor is within a factor 2.1 of $2.5\times10^{-6}$, so a measured response is above round-off exactly when it exceeds about 20 ulp of its own state norm. This is what makes the "clean radius window" model-dependent in absolute units even though the normalized measurement is portable.

---

## 1. What is compared, and every quantity

### 1.1 Setup

| item | DS-R1-1.5B | DS-R1-14B | Qwen3.5-9B | phi-4 |
|---|---|---|---|---|
| blocks / width | 28 / 1536 | 48 / 5120 | 32 / 4096 | 40 / 5120 |
| attention | sliding window 100 | sliding window 100 (**true**) | 3 Gated-DeltaNet + 1 full per period; no window | full; no window |
| prompt | `interstellar_travel`, $P=65$ | same text, $P=65$ (token-identical ids) | same text, $P=64$ | same text, $P=64$ |
| directions $K$ | 20 (one simplex block) | 1000 (50 blocks of 20) | 1000 | 1000 |
| run radius | 0.1 | 0.1 | 0.1 | 0.1 |
| continuation | 300 greedy tokens, teacher-forced | same | same | same |
| free run | only $r\in\{10^{-4},10^{-3},10^{-2}\}$, 20 dirs | $r\in\{0.1,0.3\}$, 1000 dirs | $r\in\{0.05,0.1,0.3\}$, 1000 dirs | $r\in\{0.1,0.3\}$, 1000 dirs |
| calibration | 12 radii, 20 dirs | 9 radii, 8 dirs, 400 tokens | 9 radii, 8 dirs, 400 tokens | 7 radii, 8 dirs, 400 tokens |
| depth | reference only ($k=8$) | 8 tokens x 48 blocks, $k=8$ | **not run** | 8 tokens x 40 blocks, $k=8$ |
| non-normality | reference ranges | 5 tokens x 9/5 blocks | **not run** | **pending** |
| files | `chaos_llm_2026/results/local_data/w100_layers/a1.npz` (read only) | `results_big/ds14/a1_inter_w100/`, `free_inter_w100/`, `calib/`, `depth_w100/`, `nonnorm_w100/`, `d2/` | `results_big/qwen35/a1_inter_w0/`, `free_inter_w0/`, `calib/`, `d2/` | `results_big/phi4/a1_inter_w0/`, `free_inter_w0/`, `calib/`, `depth_w0/` |

Two protocol facts matter for every comparison below. (i) The **prompt text is identical but the tokenizations are not**: the DeepSeek pair uses 65 tokens and the other two 64, and the four models generate different continuations. A step index $t$ is therefore *not* a common event across columns; only distributions over steps, and the protocol, are comparable. (ii) The **direction sets differ in richness**: 20 directions give a 20-step ECDF staircase, 1000 give a smooth curve. That is a sampling-precision difference, not a bias, but it limits how finely the 1.5B's median can be resolved.

The 14B column uses the **true-window-100** run throughout. The earlier 14B runs silently used full attention because transformers 4.5x gates the window on a `layer_types` field that stock Qwen2.5 configs never set. The true-window run is the one with `window=100` genuinely applied and is the one used for the core, free-run, depth and angle numbers here; the older full-attention run is quoted only as a check (`a1_inter_full`, `free_inter_full`). The mathematical objects are the same, but they are not the same run: the 14B's median $r_*$ moves from 0.1164 to 0.1035 and its median controlling margin from $9.727\times10^{-5}$ to $1.144\times10^{-3}$ because the controlling tie moves from $t=21$ to $t=19$ for the median direction. No conclusion below depends on which of the two 14B runs is used, but a 14B number without a run label would be ambiguous.

### 1.2 Quantities

**Fragility and stability.** Notation: $u$ indexes a direction, $t\in\{0,\dots,299\}$ a generated step, $\ell^{(1)}_t,\ell^{(2)}_t$ the top two logits of the unperturbed continuation, $r=0.1$ the campaign radius.

| symbol | explicit expression | meaning | why this quantity |
|---|---|---|---|
| $m_t$ | $\ell^{(1)}_t-\ell^{(2)}_t$ | baseline top-1/top-2 gap (nats) | how close the unperturbed decision is; a flip must close it |
| $\delta m_t(u)$ | $m_t(u)-m_t$ at the run radius, stored `dm` | change of the gap under direction $u$ | response of the decision variable, measured without letting the text diverge |
| $s_t(u)$ | $\lvert\delta m_t(u)\rvert/r$ | local decision sensitivity (nats per unit radius) | converts a radius into a logit-gap change; the denominator of $r_*$ |
| $r_c(t,u)$ | $m_t r/\max(-\delta m_t(u),0)$, $+\infty$ if $\delta m_t\ge0$ | radius at which step $t$ would flip | solves $m_t+r_c\,\delta m_t/r=0$; only a gap-closing response counts |
| $r_*(u)$ | $\min_t r_c(t,u)$ | flux-tube radius | one number per direction; the largest radius that preserves every baseline argmax |
| $t_*(u)$ | $\arg\min_t r_c(t,u)$ | controlling step | identifies the weak decision |
| $m_*(u),s_*(u)$ | $m_{t_*}$, $s_{t_*}$ | controlling margin and sensitivity | the two factors of $r_*=m_*/s_*$ |
| median-$r_*$ direction | $u$ with median $\log r_*$ | a typical fragile direction | makes the median decomposition exact, because one direction's $m_*$ and $s_*$ belong together |
| extreme direction | $\arg\min_u r_*(u)$ | the most fragile direction | tests whether the conclusion survives at the tail |
| $\bar m,\bar s,\bar r$ | $\exp\langle\log m_*\rangle$, $\exp\langle\log s_*\rangle$, $\exp\langle\log r_*\rangle$ | geometric means | log-means are additive, so $\log\bar r=\log\bar m-\log\bar s$ exactly |
| $s_{\rm emb}$ | $\mathrm{median}_t\lVert h_t^{0}\rVert$ | median token-embedding norm | converts absolute radius into "fraction of a token vector" |
| flip fraction | $\#\{u:r_*(u)<\rho\}/K$ | predicted flips at radius $\rho$ | the linear prediction tested against a free run |
| exact / wrong / false alarm / miss | predicted step equals / differs from observed; predicted but no observed flip; observed but no prediction | free-run scorecard | separates count errors from step-identity errors |
| accuracy | (exact + both-none)/$K$ | row-level accuracy | includes the easy non-flips; must be read with the flip count |
| re-synchronisation | a 20-token identical run after the flip, or a 20-token identical suffix | return to the baseline continuation | distinguishes a transient from a permanent divergence |

**Geometry, dimension and round-off.**

| symbol | explicit expression | meaning | why this quantity |
|---|---|---|---|
| angle $\theta$ | $\arccos\!\big(\langle\delta h,h_0\rangle/(\lVert\delta h\rVert\lVert h_0\rVert)\big)$ at the final-norm state | response-vs-state angle (degrees) | distinguishes an angular perturbation from radial drift |
| $h_{\rm depth}$ | $\sum_i\max(\lambda_i(L),0)$, $\lambda_i(l)=\frac1{l+1}\sum_{b\le l}\log|R_{ii}(b)|$ | total positive tangent exponent over depth (nats) | absolute expansion of an infinitesimal perturbation |
| relative exponent | $\lambda_{\rm tangent}-\lambda_{\rm norm}$ on saved-state intervals | state-normalized contraction rate (nats/block) | separates real tangent growth from the state norm growing |
| $\sigma_{\max}$ | largest singular value of the block Jacobian | worst-case one-step gain | measures non-normality, i.e. transient amplification |
| $\rho$ | spectral radius of the block Jacobian | asymptotic per-step gain | the denominator that makes $\kappa$ a transient measure |
| $\kappa$ | $\sigma_{\max}/\rho$ | non-normality index | $\kappa\approx1$ is normal; larger means a rotating/amplifying direction |
| TwoNN $\hat d$ | $1/\langle\ln\mu_i\rangle$, $\mu_i=r_2^{(i)}/r_1^{(i)}$ | nearest-neighbour dimension estimate | a point-cloud dimension that does not require a scaling window |
| MLE $k=10$ | Levina-Bickel $\hat d_k$ | a second point-cloud dimension | checks whether TwoNN is estimator-specific |
| $D_2$ chord | slope of $\log C(\epsilon)$ over the admissible range | correlation-dimension chord | what one gets without a plateau; not a dimension |
| prominence | $\min_k p(k)/\mathrm{median}\,p(k\pm1,2,3)$, $p(k)=\overline{\lVert x_t-x_{t+k}\rVert}/d_{\rm typ}$ | loop detector | a drifting repetition makes a sharp lag-distance dip |
| floor $F_l$ | $\mathrm{median}_{u,t\ge100}\lVert\delta h_t^l(u)\rVert$ at the smallest tested radius | fp32 response floor at layer $l$ | the smallest measurable response; sets the lower edge of the linear window |
| ulp fraction | $F_l/(\lVert h_l\rVert\,\epsilon_{32})$, $\epsilon_{32}=1.192\times10^{-7}$ | floor in units of the state norm | tests whether the floor is absolute or scales with the state |

**Why the decomposition needs a common direction.** The medians of $m_*$ and $s_*$ are summaries of two *different* directions. For phi-4, median $m_*=0.01702$ comes from the direction controlled by $t=219$ ($s=0.00298$), while median $s_*=0.00767$ comes from the direction controlled by $t=34$ ($m=0.05674$). Dividing the two marginal medians gives $2.22$, and would place phi-4 *below* the 1.5B (whose marginal ratio is $4.78/2.42$ in the same units, $2.86$), the wrong order. The actual median $r_*$ is 3.59, and its direction ($t=253$) has $m=0.0631$, $s=0.0176$. The reason is that $m_*$ and $s_*$ are positively correlated within every model (Pearson on logs: 1.5B 0.51, 14B 0.74, Qwen3.5 0.64, phi-4 0.59): a direction that moves the logits hard also tends to move the state hard, so the ratio is less variable than either factor. Any "margin-only" or "response-only" summary that divides marginal medians is therefore invalid, and the three explicit decompositions below (median direction, geometric mean, extreme direction) replace it.

### 1.3 Four things that could fake the hierarchy

1. **The 14B attention history.** The core column is the true-window run. The older full-attention run gives median $r_*=0.1164$ instead of $0.1035$ (+12%), median $m_*=9.727\times10^{-5}$ instead of $1.144\times10^{-3}$ (−11.8x) and median $s_*=3.802\times10^{-3}$ instead of $5.536\times10^{-3}$ (−1.46x); the 14B stays last in every version. The free-run flip count at $r=0.1$ is 453/1000 (full) against 483/1000 (true window), both far above the other models.
2. **$K=20$ versus $K=1000$.** The 1.5B's ECDF is a 20-step staircase and its log-$r_*$ values are unevenly spaced. A bootstrap over its 20 directions gives a 5-95% interval of 1.41-1.99 for the median (point estimate 1.555), a factor 1.41 wide. The Qwen3.5 gap (2.20x) and the phi-4 gap (2.31x, other direction) lie outside that interval, but the bootstrap treats the 20 directions as exchangeable samples; they are one simplex block, not a random sample of all embedding directions. The 14B gap (15x) is far outside every such interval.
3. **Embedding scale.** $s_{\rm emb}$ is 1.206 (1.5B), 1.518 (14B), 0.777 (Qwen3.5) and 1.193 (phi-4). Raw $r_*$ is an absolute perturbation norm, so it is comparable by construction, but dividing by $s_{\rm emb}$ asks the stricter "fraction of a token vector" question. The normalized medians are 1.289 (1.5B), 0.068 (14B), 0.910 (Qwen3.5) and 3.006 (phi-4); the order is unchanged, the 1.5B/14B gap widens from 15.0x to 18.9x, and Qwen3.5 moves closer to the 1.5B (1.42x instead of 2.20x) because its token vectors are 36% shorter. Neither normalization changes the qualitative hierarchy.
4. **Different continuations and different gates.** The four models generate different text, so their $t_*$ values are not comparable events. What is comparable is the distribution of margins, the protocol, and the fact that each model's own near-ties are where its directions act. phi-4's and Qwen3.5's controlling steps are spread over 27 and 16 distinct steps; the 14B's over 15, with 86.3% of directions on three steps (21, 19, 297); the 1.5B's over 6, with 75% on three. A 14B conclusion is therefore about a few specific gates; a phi-4 conclusion is about a broader set of decisions.

### 1.4 Headline table

| quantity | 1.5B | DS-14B (w100) | Qwen3.5-9B | phi-4 |
|---|---:|---:|---:|---:|
| $K$ / $r$ | 20 / 0.1 | 1000 / 0.1 | 1000 / 0.1 | 1000 / 0.1 |
| median margin $m_t$ over steps | 4.778 | 1.826 | 2.461 | 3.473 |
| smallest margin | $2.016\times10^{-2}$ (123) | $9.727\times10^{-5}$ (21) | $7.271\times10^{-3}$ (267) | $9.254\times10^{-3}$ (277) |
| median $r_*$ (absolute) | 1.555 | **0.1035** | 0.7074 | 3.5865 |
| $r_*$ p10 / p90 | 0.486 / 4.124 | 0.0171 / 1.852 | 0.232 / 1.956 | 1.201 / 10.663 |
| median $m_*$ | $6.902\times10^{-2}$ | $1.144\times10^{-3}$ | $2.048\times10^{-2}$ | $1.702\times10^{-2}$ |
| median $s_*$ | $2.416\times10^{-2}$ | $5.536\times10^{-3}$ | $3.309\times10^{-2}$ | $7.671\times10^{-3}$ |
| $m_*$ at median-$r_*$ direction | $2.041\times10^{-2}$ | $1.144\times10^{-3}$ | $7.271\times10^{-3}$ | $6.314\times10^{-2}$ |
| $s_*$ at median-$r_*$ direction | $1.301\times10^{-2}$ | $1.104\times10^{-2}$ | $1.028\times10^{-2}$ | $1.759\times10^{-2}$ |
| median $r_*/s_{\rm emb}$ | 1.289 | 0.068 | 0.910 | 3.006 |
| flip fraction at $r=0.1$, predicted | 0 | 0.493 | 0.012 | 0 |
| flip fraction at $r=0.1$, observed free | 0 (20/20 at $r\le0.01$) | 0.483 | 0.012 | 0 |
| controlling steps, distinct / top-3 share | 6 / 75% | 15 / 86.3% | 16 / 59% | 27 / 62.1% |
| response-vs-state angle, median | 87° (literature) | 89.99° | 90.08° | 89.85° |
| TwoNN median | 6.17 | 8.19 | 7.50 | not run |
| valid $D_2$ window | none (0/2 healthy) | none (0/13) | none (0/12) | not run |
| repetition loops | 9 full + 2 partial / 20 | 0 / 13 | 0 full + 1 partial / 12 | not run |
| re-synchronised / flipped | 0/0 free (literature 0) | 0/1186 | 0/169 | **2/4** |
| fp32 floor / state norm | $1.83\times10^{-6}$ (last) | $2.19$–$2.88\times10^{-6}$ | $1.41$–$3.15\times10^{-6}$ | $2.52$–$3.51\times10^{-6}$ |

The 14B's smallest margin is the $t=21$ razor tie; its *median* direction is controlled by $t=19$ with $m=1.144\times10^{-3}$. Those are two different statements and both are used below.

---

## 2. Figure `syn_fragility.png` — the margin axis places the four models

![fragility](figures/syn_fragility.png)

**Setup.** The core A1 arrays for all four models at the campaign radius $r=0.1$: baseline margins $m_t$, response $\delta m_t(u)$, and the signed threshold $r_c(t,u)=m_t r/\max(-\delta m_t(u),0)$. Each direction contributes one point $(s_{t_*},m_{t_*})$ and one radius $r_*=m_{t_*}/s_{t_*}$. Panel (b) plots every direction, subsampled to 450 per model for legibility; the four stars are the median-$r_*$ directions. Panel (d) is the exact split of the median $r_*$ into a margin term and a sensitivity term, relative to the 1.5B.

**Axes.**
- (a) $x$: $r_*$ in absolute embedding units, log scale, 1.5e-3 to 40; $y$: ECDF over directions, 0-1. The dotted vertical line is the campaign radius $r=0.1$.
- (b) $x$: controlling sensitivity $s_{t_*}$, log; $y$: controlling margin $m_{t_*}$, log. Dashed diagonals are constant $r_*=0.1,1,10$. A point's position along the diagonal is set by its ratio, not by either coordinate.
- (c) $x$: $s_{t_*}$, log; $y$: ECDF. The medians are printed in the inset.
- (d) $x$: $\Delta\log_{10}$ relative to the 1.5B (dex); bars are the margin term $\Delta\log m_*$ (blue) and the sensitivity term $-\Delta\log s_*$ (orange); the black diamond is the net $\Delta\log r_*$. The three terms satisfy the identity exactly because all three are evaluated on the same median-$r_*$ direction.

**The hierarchy and its decomposition.** The ECDFs are ordered without overlap: the 14B's median is 0.1035, Qwen3.5's 0.7074, the 1.5B's 1.5547 and phi-4's 3.5865. The ratios are 6.8x, 2.20x and 2.31x; the end-to-end factor is 34.7x.

At the median-$r_*$ direction the table is:

| model | $t_*$ | $m_*$ | $s_*$ | $r_*=m_*/s_*$ | $\Delta\log_{10}r_*$ | from margin | from $1/s_*$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | 143 | $2.041\times10^{-2}$ | $1.301\times10^{-2}$ | 1.569 | 0 | 0 | 0 |
| DS-14B | 19 | $1.144\times10^{-3}$ | $1.104\times10^{-2}$ | 0.1037 | −1.180 | −1.251 | +0.071 |
| Qwen3.5 | 267 | $7.271\times10^{-3}$ | $1.028\times10^{-2}$ | 0.7074 | −0.346 | −0.448 | +0.102 |
| phi-4 | 253 | $6.314\times10^{-2}$ | $1.759\times10^{-2}$ | 3.590 | +0.359 | +0.491 | −0.131 |

Read the last three columns as factors rather than dex: the 14B's typical margin is 17.8x thinner than the 1.5B's, and its sensitivity is 1.18x *lower*, which would raise $r_*$ by that factor; the product is 15.1x lower. Qwen3.5's margin is 2.81x thinner and its sensitivity 1.27x higher; the product is 2.22x lower. phi-4's margin is 3.09x fatter but its sensitivity is 1.35x higher; the product is 2.29x higher. In all three comparisons the margin term is larger than the sensitivity term and has the sign of the net. The margin values $1.14\times10^{-3}<7.27\times10^{-3}<2.04\times10^{-2}<6.31\times10^{-2}$ are exactly monotone with $r_*$; the sensitivities span only 1.71x and are non-monotone.

**The geometric-mean check, and why phi-4 is a partial exception.** Repeating the split with geometric means (log-means, which are additive) gives

| model | $\Delta\log_{10}r_*$ | margin | sensitivity | net factor |
|---|---:|---:|---:|---:|
| 1.5B | 0 | 0 | 0 | 1 |
| DS-14B | −1.054 | −1.704 | +0.650 | 0.088 |
| Qwen3.5 | −0.324 | −0.257 | −0.067 | 0.474 |
| phi-4 | +0.387 | −0.191 | +0.578 | 2.438 |

The 14B and Qwen3.5 stories are the same: the margin term is large and negative; the 14B's sensitivity is 4.5x lower and softens the fall, while Qwen3.5's is 1.17x higher and adds a small negative term. phi-4 is different: its *geometric-mean* margin is 1.55x *thinner* than the 1.5B's, and its geometric-mean sensitivity is 3.78x lower; the net comes from the response term. This does not contradict the median table; it says phi-4's radius distribution has a wide quiet tail. The distribution is indeed wide: p10 1.20, p50 3.59, p90 10.66, and 923/1000 directions have $r_*>1$.

**The extreme direction makes the same point.** At the most fragile direction of each model:

| model | $t_*$ | $m_*$ | $s_*$ | $r_*$ | $\Delta\log_{10}r_*$ vs 1.5B | margin | sensitivity |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | 16 | $0.1200$ | $0.9031$ | 0.1329 | 0 | 0 | 0 |
| DS-14B | 21 | $9.727\times10^{-5}$ | $0.0399$ | 0.00244 | −1.737 | −3.091 | +1.355 |
| Qwen3.5 | 267 | $7.271\times10^{-3}$ | $0.1155$ | 0.0629 | −0.325 | −1.218 | +0.893 |
| phi-4 | 277 | $9.254\times10^{-3}$ | $0.0559$ | 0.1655 | +0.095 | −1.113 | +1.208 |

The 1.5B's most fragile direction is a *very* sensitive one ($s=0.903$, 37x its median $s_*=0.0242$), with a comfortable 0.12-nat margin; phi-4's most fragile direction is a genuinely thin tie ($m=9.25\times10^{-3}$) that is 16x quieter. The result is that phi-4's worst radius (0.165) is slightly *above* the 1.5B's (0.133) despite a 13x thinner margin. The 14B is again margin-dominated (a 1234x thinner margin against a 22.6x lower sensitivity). So the honest synthesis is: **the bulk ($r_*$ median and geometric mean) is placed by margins; the lower tail is placed by sensitivity, and this matters for the 1.5B-versus-phi-4 comparison, not for the 14B.**

**What it shows.** (1) The median hierarchy is monotone in the controlling margin and not in the controlling sensitivity. (2) The identity $r_*=m_*/s_*$ is exact per direction and is not reproducible from marginal medians; the naive ratio swaps phi-4 and the 1.5B. (3) $m_*$ and $s_*$ correlate within models (0.51-0.74 on logs), which is why the ratio is less variable than either factor and why a decomposition has to be stated at a common direction. (4) The 14B is the only model whose bottom position survives every summary: median, geometric mean and extreme. (5) The 1.5B-vs-Qwen3.5 order is a margin effect (2.8x thinner at a sensitivity only 1.27x higher), not a response effect.

**What it does not show.**
- It does not establish a causal mechanism. "Margin" and "sensitivity" are two readings of the same logit-gap change; a model can have thin margins because its representation is confident-but-close, or because its read-out is unstable. The decomposition says where the number comes from, not why the network produces it.
- The 20-direction 1.5B sample is a cluster sample, not an i.i.d. sample. A bootstrap gives 1.41-1.99 for its median, so the 1.5B/phi-4 (2.3x) and 1.5B/Qwen3.5 (2.2x) gaps fall outside that interval, but the 20 directions share one simplex block and the interval is not a confidence statement about all directions.
- $r_*$ is a *linear extrapolation* from one radius, not a measured nonlinear threshold. The 1.5B at $r=1.0$ flips 8/20 decisions while the linear rule predicts 4/20; $r_*$ is a local object, valid in the calibrated window ($10^{-3}$-0.1 for the 1.5B, 0.03-0.3 for the 14B, 0.01-0.3 for Qwen3.5 and phi-4). Section 3 quantifies the failure outside it.
- The models are on different texts and different tokenizations. $t_*$ is a within-model label, not a cross-model event.
- The absolute radii are not normalized by $s_{\rm emb}$ in the figure. The normalized check in 1.3 preserves the order but changes the 1.5B/14B gap from 15.0x to 18.9x and the 1.5B/Qwen3.5 gap from 2.20x to 1.42x. Both are reported; neither is "the" answer.

---

## 3. Figure `syn_stability.png` — the rule predicts counts, not late-flip identity

![stability](figures/syn_stability.png)

**Setup.** For each model, the signed rule $r_c(t,u)\le\rho$ predicts a flip at the first crossing step; the observed step is either a real free run (14B at 0.1/0.3, Qwen3.5 at 0.05/0.1/0.3, phi-4 at 0.1/0.3, 1.5B at $10^{-4}$/$10^{-3}$/$10^{-2}$) or, for the 1.5B's 0.3 and 1.0 rows, the first threshold crossing in the separate teacher-forced A1 pass at that radius. The 14B is the true-window run. A prediction and an observation that flip at different steps are counted as a *wrong step*, not as a detection, because the model's downstream text depends on the step.

**Axes.**
- (a) $x$: perturbation radius $\rho$, log, 0.04-1.25; $y$: fraction of directions that flip, linear 0-0.78. Filled circles and solid lines are observed; open squares and dashed lines are predicted; colour is model.
- (b) one row per model-radius. Dark bar: exact steps / observed flips. Light bar: exact steps / predicted flips. The number after the bar is the observed flip count; rows with zero observed flips are hatched and carry the overall row accuracy.
- (c) $x$: model; $y$: re-synchronised runs / flipped runs, 0-0.62; the labels are the raw counts.

**Numbers.**

| model | $\rho$ | observed | predicted | exact | wrong | false alarm | miss | accuracy | exact/obs | exact/pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | 0.1 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | — | — |
| 1.5B | 0.3 | 1 | 1 | 1 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 1.5B | 1.0 | 8 | 4 | 0 | 2 | 2 | 6 | 0.500 | 0.000 | 0.000 |
| DS-14B | 0.1 | 483 | 493 | 478 | 2 | 13 | 3 | 0.982 | 0.990 | 0.970 |
| DS-14B | 0.3 | 703 | 723 | 625 | 41 | 57 | 37 | 0.865 | 0.889 | 0.864 |
| Qwen3.5 | 0.05 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | — | — |
| Qwen3.5 | 0.1 | 12 | 12 | 12 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Qwen3.5 | 0.3 | 157 | 160 | 104 | 27 | 29 | 26 | 0.918 | 0.662 | 0.650 |
| phi-4 | 0.1 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | — | — |
| phi-4 | 0.3 | 4 | 2 | 0 | 1 | 1 | 3 | 0.995 | 0.000 | 0.000 |

The counts are predicted well: 493 vs 483 at $r=0.1$ (2.1% high), 723 vs 703 at 0.3 (2.8% high), 12 vs 12 at 0.1 for Qwen3.5 and 160 vs 157 at 0.3 (1.9% high), 0 vs 0 for phi-4 at 0.1. The step identity is another matter. At low flip density, the rule is nearly exact: 478/483 for the 14B at 0.1, 12/12 for Qwen3.5 at 0.1, 1/1 for the 1.5B at 0.3. At high density, the errors are mostly *late* flips: Qwen3.5 at 0.3 has 27 wrong steps against 104 exact, and the 14B at 0.3 has 41 wrong against 625 exact. The 14B's two wrong steps at 0.1 are one 19-vs-21 inversion (direction 986: observed 21, predicted 19) and one 297-vs-21 (direction 985: observed 297, predicted 21). The 19-vs-21 inversion is the first-crossing-versus-argmin distinction already seen in the state-kick analysis: the rule chooses 19 because it is the first step whose $r_c$ falls below 0.1, while the free run first departs at 21 because the response there is larger. The 297 case is a late gate reached only after the accumulated divergence, not a near-tie at 0.1.

**Re-synchronisation.** The counts are 0/1186 for the 14B (483 + 703 flipped rows across the two radii), 0/169 for Qwen3.5 (12 + 157), 0/0 for the 1.5B free run (no flips at $r\le0.01$; the T4 analysis reported none), and **2/4** for phi-4. The two phi-4 cases are directions 276 (substitution at 219, immediate re-join, 57-token identical run, second substitution at 277, re-join at 278) and 933 (substitution at 277, re-join at 278, 22-token identical suffix). Both are one-token excursions that snap back, not slow convergence. The longest shared run anywhere else is 17 tokens for the 14B (a verbatim prompt echo) and 1 token for Qwen3.5. The stored `mean_postflip_agreement` fields (e.g. 0.73 for the 14B) are averages over the non-flipped rows and are not re-synchronisation rates; they are not used here.

**What it shows.** The signed linear rule is a good *detector* across four models and three orders of magnitude of radius: the predicted flip fraction tracks the observed one to within 2-3% absolute up to $r=0.3$. It is not a *step oracle*: once many directions flip, the correct answer depends on the divergent text, and the rule's identity rate falls (99% -> 89% for the 14B, 100% -> 66% for Qwen3.5). The flip count is therefore a robust cross-model observable; the exact step is not.

**What it does not show.**
- Row accuracy is dominated by both-none when flips are rare. phi-4's 99.5% at 0.3 is 995 non-flips plus 0 exact; it is not evidence that the rule works. The exact/observed column is the honest one and it is 0 there.
- The 1.5B's $r=1.0$ row is outside its measured linear window ($10^{-3}$-0.1) and is a teacher-forced decision flip, not a free continuation. Its 8-vs-4 failure measures nonlinearity, not the rule's performance in its calibrated range. The only genuine 1.5B free run is the 20/20 null at $r\le0.01$.
- The 14B numbers use the true-window run; the 1.5B's window is 100 and is correctly applied in its source run. The full-attention 14B has 453 observed and 464 predicted at 0.1 with 449 exact. The conclusion is the same, but the counts are not interchangeable.
- The definition of re-synchronisation (20 tokens) is a threshold, not a physical time. A slower convergence with a 10-token match would not be counted. The longest 14B match is 17 tokens, so the 14B result is not sensitive to the exact cut; the phi-4 sample is 4 rows and cannot estimate a rate.
- A free run and a teacher-forced pass are different objects: the teacher-forced pass keeps the baseline attention pattern and the baseline text, while the free run lets the model condition on its own divergence. The 1.5B rows mix the two by necessity.

---

## 4. Figure `syn_angle_depth.png` — perpendicular responses, contracting tangents

![angle_depth](figures/syn_angle_depth.png)

**Setup.** (a) The angle between the last-state response $\delta h$ and the baseline last state $h_0$, at the final-norm output, over 24 saved directions x 300 steps (7200 pairs) for the three models whose A1 dump stores `h0_last` and `dh_last`: the true-window 14B, Qwen3.5 and phi-4. The 1.5B is a literature value (87°), not a measurement, because its dump stores state norms and $\cos(h',h_0)$ rather than $h_0$ vectors. (b)-(c) The depth spectrum from a JVP through all blocks with $k=8$ orthonormal tangent vectors and QR renormalisation every block; probe tokens 0, 21, 50, 100, 150, 200, 250, 299. The 1.5B band is the reference range from the committed depth section. (d)-(e) Full-block non-normality at five tokens, five blocks (0,12,24,36,47); the 1.5B reference bands are drawn at the matching relative positions.

**Axes.**
- (a) $x$: angle in degrees, 60-120; $y$: model. Thin points are a 1400-pair subsample of each distribution; the diamond is the median; the open star for the 1.5B is the literature value; the dotted line is 90°, the dashed line 87°.
- (b) $x$: probe token; $y$: $h_{\rm depth}=\sum_i\max(\lambda_i(L),0)$ in nats, log. The blue band is the 1.5B reference 0.6-1.4.
- (c) $x$: probe token; $y$: state-normalized tangent rate in nats/block (overall per-token value on the saved-state intervals). The blue dashed line is the 1.5B's least-negative value, $-0.034$.
- (d) $x$: block index; $y$: full-block $\sigma_{\max}$, log. Red line: 14B median over five tokens, band min-max. Open squares: 1.5B reference at block 0, the middle (drawn at 12/24/36) and the last block.
- (e) $x$: block index; $y$: $\kappa=\sigma_{\max}/\rho$. Red line and band: 14B; the 1.5B reference is printed as text because it is a range, not a per-block curve.

**Numbers.** The measured response-vs-state angles are **89.99°** for the 14B (mean 90.10, p10 77.66, p90 102.61, 49.9% above 90°), **90.08°** for Qwen3.5 (mean 90.25, IQR 89.51-90.91, 54.7% above 90°) and **89.85°** for phi-4 (mean 89.54, IQR 85.54-93.51, 48.7% above 90°). The 1.5B literature value is 87°. The response is $1.71\times10^{-4}$ of the state norm in the 14B, $4.94\times10^{-4}$ in Qwen3.5, and $9.71\times10^{-5}$ in phi-4; phi-4's is the smallest and its angle distribution the widest, which is consistent with a noisier direction estimate rather than a different geometry.

The depth numbers: the 14B's $h_{\rm depth}$ runs from 0.163 to 0.638 nats, with the maximum at $t=21$ (the razor tie), 1.55x the next largest and 2.5x the median of the other seven; its state-normalized rate is $-0.061$ to $-0.120$ nats/block, with 61/64 six-block chunks negative and the least-negative chunk $-0.034$. phi-4's $h_{\rm depth}$ runs from 0.154 to 0.859, with the maximum at $t=0$ (the first generated token), 2.3x the next; its normalized rate is $-0.054$ to $-0.123$ per token, with 93/96 intervals negative and a median of $-0.105$ nats/block. The 1.5B reference is $h_{\rm depth}=0.6$–$1.4$, absolute exponent 0.08-0.18 per block, worst relative $-0.034$. Qwen3.5 has no depth run; its layer profile measures the *response* growth, not the tangent contraction, and is not a substitute.

The non-normality numbers are only complete for the 14B. Full-block $\sigma_{\max}$ is 96.2 at block 0 (range 66.2-128.5), 3.7-4.9 at blocks 12-36 and 14.2 at block 47; $\kappa$ is 7.35, 1.60-2.27 and 7.07. The MLP alone at block 47 has $\sigma_{\max}=117.7$ (94.4-146.5), against the 1.5B's 58-93. The 1.5B reference is $\sigma_{\max}=95$–$195$, $2.7$–$3.6$, $6$–$9$ and $\kappa=5$–$8$, $1.7$–$2.5$, $3$–$5.3$. The middle blocks match; the last block is 1.6-2.4x more non-normal in the 14B. The principal angle between the top singular directions of blocks 4 apart is 86.96° (14B) against 83.6° (1.5B); that is a different quantity from the response-vs-state angle in panel (a) and should not be conflated with it.

**What it shows.** (1) At 9-14B the response is perpendicular to the state, not radial: the three measured medians agree to 0.23° and the per-pair distributions straddle 90°. This sharpens the T4 1.5B statement ("about 87°") into "exactly perpendicular" at the larger scales, with the caveat that the 1.5B number is not measured on the same objects. (2) The depth spectrum reproduces the 1.5B law in both models that were run: absolute tangent volume grows (up to 46x in the 14B, 73x in phi-4) but the state norm grows far more (655-833x, 1621x), so the normalized tangent contracts almost everywhere. (3) The razor tie is an absolute hotspot in the 14B (largest $h_{\rm depth}$) but not a normalized one (it contracts least, -0.061); in phi-4 the absolute hotspot is the first generated token, not either tie. (4) The 14B's last block is the one clearly non-normal place relative to the 1.5B; the middle is ordinary.

**What it does not show.**
- The 1.5B angle is literature. The 3° difference from 90° is not established, and the 1.5B's own principal-angle number (83.6°) is a different measurement.
- The full-attention 14B run has no `h0_last`, so the angle cannot be cross-checked against it; the true-window run is the only 14B angle. The 24 directions are 24 out of 1000; the angle is a 24-direction sample, not a census.
- phi-4's angle is measured on responses that are only $9.7\times10^{-5}$ of the state norm, so its direction estimate is noisier; its IQR (8°) is twice Qwen3.5's (1.4°). The medians agree, but the shapes are not equally well measured.
- The depth run uses 8 probe tokens and $k=8$; phi-4's token list does not include its ties (219, 277), so the phi-4 tie-hotspot question is unanswered. The 1.5B reference ranges come from a different dump with a different depth resolution.
- Qwen3.5's depth is simply absent. The layer profile shows a 159x response growth and saturation after block 19; that is not the same as a contracting tangent and is not used as one.
- Non-normality is one model. The phi-4 fused-MLP Jacobian is still generating and Qwen3-14B is pending; the cross-model statement "the last block is more non-normal at 14B" is a 14B-vs-1.5B statement only.
- The 14B's relative exponent is a 6-block average because `norm_h0` saves only nine states; per-block contraction is not measured. The three positive chunks are at the resolution floor (+0.0003 to +0.0083 nats/block), not a qualitative failure.

---

## 5. Figure `syn_dimension.png` — no window, no loops, one floor

![dimension](figures/syn_dimension.png)

**Setup.** (a) TwoNN $\hat d=1/\langle\ln\mu_i\rangle$ on each 1000-token last-layer trajectory (the 1.5B validation file uses 4095-token runs; the 14B and Qwen3.5 use 1000). One point per trajectory; the horizontal bar is the median. The grey band is the 1.5B validation range 5.44-6.23. (b) Repetition-loop counts from the lag-distance prominence screen: full loops (prominence <0.6), partial (0.6-0.8) and clean. The 1.5B count is the 20 local trajectories at 1000 tokens; the equal-horizon subset is 16 runs (6 full + 2 partial). (c) The fp32 floor fraction $F_l/(\lVert h_l\rVert\epsilon_{32})$ at the smallest tested radius, per layer, for the models whose calibration and norms are available.

**Axes.**
- (a) $x$: model; $y$: TwoNN, 3.8-11.1. Each dot is one trajectory; the bar is the median. The "none" labels are the valid-window verdict; phi-4 has no D2 run.
- (b) $x$: model; $y$: number of trajectories. Grey is clean, red is a full loop, orange is a partial loop; the label is full+partial/total.
- (c) $x$: block index; $y$: $F_l/\lVert h_l\rVert$ in fp32 ulp units, log, 5-60. Dashed horizontals are 10, 20 and 30 ulp. The 1.5B star is its last saved state. The 14B uses the matched true-window state norms; the full-attention normalization gives 21-25 ulp instead of 18-24 and is quoted in the JSON.

**Numbers.** TwoNN: 1.5B **5.44-6.23** (median 6.17, four validation runs; two of those are loops), 14B **4.66-8.90** (median 8.19, 13 runs), Qwen3.5 **6.05-8.09** (median 7.50, 12 runs). The MLE at $k=10$ gives 7.9-10.1 (1.5B healthy), 11.0-15.9 (14B) and 8.9-12.9 (Qwen3.5); the T4-style chords are 9.20-11.35 (1.5B healthy), 8.69-11.75 (14B) and 12.58-22.36 (Qwen3.5). **No model has a valid scaling window**: 0/2 healthy 1.5B runs, 0/13 14B and 0/12 Qwen3.5 pass the 0.40-decade rule. The single passing flag in the 1.5B file is a spurious 0.54-decade window on the period-53 loop; the two healthy 1.5B runs fail. The loops are 9 full + 2 partial / 20 for the 1.5B (6 full + 2 partial / 16 at the equal 1000-token horizon), 0 / 13 for the 14B, and 0 full + 1 partial / 12 for Qwen3.5 (the partial is `great_emu_war`, a late period-7 repeat with 13.8% coverage and prominence 0.849).

The floor fractions are: 14B $2.19$–$2.88\times10^{-6}$ (18-24 ulp; the established full-attention normalization is $2.57$–$3.03\times10^{-6}$, 21-25 ulp), Qwen3.5 $1.41$–$3.15\times10^{-6}$ (12-26 ulp), phi-4 $2.52$–$3.51\times10^{-6}$ (21-29 ulp), 1.5B $1.83\times10^{-6}$ at the last state (15 ulp). The 14B's per-layer values are 18.4, 23.4, 24.2 and 20.8 ulp at layers 12, 24, 36 and 47 (layer 0 is exactly zero because the perturbation enters the prompt, not generated state 0); the 14B is flat within 1.3x across depth. Qwen3.5 rises from 11.9 ulp at layer 8 to 26.5 at layer 31, a 2.2x drift; phi-4 is 29.4 at layer 20 and 21.1 at layer 39. The order of magnitude is the same in every model: about 20 ulp, or $2$–$3\times10^{-6}$ of the state norm.

**What it shows.** (1) The point-cloud estimators are not converged and no model has a scaling region; the 1.5B's "dimension" 5.44-6.23 is not a low-dimensional exception to a high-dimensional 14B, because the 14B's TwoNN spans the same band (4.66-8.90) and its MLE is higher. (2) The 14B and Qwen3.5 have no repetition loops at the same horizon where the 1.5B has 6 full + 2 partial of 16; Qwen3.5 has one partial late repeat, not a full loop. (3) The fp32 floor is not an absolute number: it scales with the state norm, at about 20 ulp, in all four models. That is why the floor is a property of the measurement, not of a model, and why the lower edge of the linear window moves in absolute radius.

**What it does not show.**
- TwoNN is not a dimension. It is a nearest-neighbour estimator with a finite-$N$ bias (the 14B's `great_emu_war` moves 7.82 -> 6.30 -> 4.66 as $N$ goes 500 -> 750 -> 1000). The "4.66-8.90" band is a range of estimates, not a dimension.
- The 1.5B validation used 4095-token runs and $d=1536$; the 14B and Qwen3.5 used 1000-token runs and $d=5120/4096$. The ranges are not like-for-like, and the 1.5B's four validation trajectories include two loops. The healthy 1.5B range is two points (6.12, 6.23), not a distribution.
- phi-4 has no D2 run. The panel says "no D2 run" rather than drawing a number.
- The loop screen is a state-recurrence test. It can miss a period longer than the 600-lag search, and it does not inspect the token text; a text repetition that leaves the state more than $0.1d_{\rm typ}$ away is not flagged. The 1.5B's 9/20 is at 1000 tokens; the equal-horizon 6+2/16 is the correct comparison to the 14B's 0/13 and Qwen3.5's 0+1/12.
- The floor fraction is measured at the smallest tested radius. For phi-4 that is $10^{-3}$, and the $3\times10^{-3}$ value is 22% higher, so its $F$ is an upper bound on the true floor. For the 14B the $3\times10^{-4}/10^{-4}$ ratio is 1.001-1.008 and for Qwen3.5 1.06-1.09 (per layer), so those floors are at the plateau; the established proportional fit $F=\kappa\lVert h\rVert$ gives $\kappa=2.610\times10^{-6}=21.9$ ulp.
- "Fixed fraction" is approximate: the spread is a factor 2.1 across models and Qwen3.5 drifts 2.2x with depth. It is a law of order of magnitude, not a constant.

---

## 6. What the campaign says

### 6.1 The hierarchy, in one paragraph

The four models are ordered by how large a prompt-embedding perturbation their greediest decisions tolerate: 14B (0.104) < Qwen3.5-9B (0.707) < 1.5B (1.555) < phi-4 (3.587). The ordering is not a size ordering (the 9B is between the two DeepSeek models, the 40-block phi-4 is the most robust) and it is not a response-strength ordering (the 14B's median per-unit-radius last-state displacement is 2.3x *smaller* than the 1.5B's). At the typical fragile direction it is a **margin ordering**: the controlling gaps are $1.14\times10^{-3}$, $7.27\times10^{-3}$, $2.04\times10^{-2}$ and $6.31\times10^{-2}$ nats, spanning 55x and monotone with $r_*$, while the co-selected sensitivities span 1.71x and are non-monotone. The 14B is fragile because its decisions are close, not because its response is strong; the same sentence with "Qwen3.5" substituted holds against the 1.5B. phi-4 is robust because its typical margin is fat, and its thin ties are dynamically quiet.

The tail is the exception that defines the rule. The most fragile direction of the 1.5B is extremely sensitive ($s=0.903$), so its worst radius (0.133) is only 1.24x below phi-4's (0.165), even though phi-4's worst margin is 13x thinner. A one-factor summary cannot capture both the bulk and the tail; the identity $r_*=m_*/s_*$, evaluated at a stated direction, does.

### 6.2 What replicates across models

1. **The signed top-2 threshold rule predicts flip counts.** 493 vs 483 (14B, 0.1), 723 vs 703 (14B, 0.3), 12 vs 12 and 160 vs 157 (Qwen3.5), 0 vs 0 (phi-4, 0.1). It is a count rule, not a step oracle.
2. **A flip is a top-2 word swap at a near-tie.** The 14B's gates are 19/21 (`" reference"` vs `" research"`, `"Alright"` vs `"Okay"`) plus 297; Qwen3.5's are 267 (`Ex` -> `Specific`) and 69 (`Distance` -> `The`); phi-4's are 219 and 277. Every observed flip takes the second token.
3. **No model except phi-4 re-synchronises.** 0/1186 (14B), 0/169 (Qwen3.5) and 0/0 (1.5B free run; no flipped row to test); phi-4's 2/4 one-token re-joins are the exception, not the rule.
4. **The response is perpendicular to the state.** 89.99°, 90.08°, 89.85° at 14B/9B/phi-4; the 1.5B literature 87° is consistent, not contradictory.
5. **Absolute tangent growth is radial drift; normalized it contracts.** 61/64 chunks (14B), 93/96 intervals (phi-4), worst $-0.034$ (1.5B). The largest absolute exponent is at a near-tie or the first token, not a generic late step.
6. **No scaling window.** 0/2, 0/13, 0/12. The 1.5B's old low-dimension headline is not a dimension and does not replicate.
7. **The fp32 floor is a fixed fraction of the state norm,** about 20 ulp (12-29 across models), not a fixed absolute number.
8. **Loops are common only in the 1.5B:** 6 full + 2 partial / 16 at the equal horizon (1.5B), 0 / 13 (14B), 0 + 1 / 12 (Qwen3.5). The 14B-to-Qwen3.5 difference is one partial and is not a scale trend; the 1.5B is the outlier. phi-4 has no D2 run and is not in the comparison.

### 6.3 What does not replicate, or is model-specific

- **The 14B's last-block non-normality.** The 14B's block-47 $\sigma_{\max}$ (14.2) and $\kappa$ (7.07) are 1.6-2.4x the 1.5B's; the middle blocks are ordinary. This is a 14B-vs-1.5B statement; phi-4 and Qwen3.5 have no non-normality run.
- **The 14B's razor tie is a single gate.** 86.3% of its directions are controlled by three steps (21, 19, 297), against 59-62% for Qwen3.5 and phi-4 and 75% for the 20-direction 1.5B. The 14B's fragility is concentrated; the others' is spread.
- **phi-4's re-synchronisation.** 2/4 versus 0 everywhere else. With 4 flips, this is a lead, not a rate.
- **The 1.5B's repetition loops.** 9+2/20 (6+2/16 at equal horizon) is far above the 14B's 0/13 and Qwen3.5's 0+1/12. The 1.5B is the only model in which loops are common enough to affect the late-time statistics.
- **phi-4's absolute depth hotspot is the first token, not a tie; the 14B's is the tie.** The normalized statement is universal; the absolute hotspot is not.

### 6.4 Limitations

1. **Four models, four prompt tokenizations, four continuations.** The prompt text is the same, but the token sequences and the generated text are not. $t$ is not a common event; only distributions and protocol are comparable. The 1.5B and 14B share prompt ids; the other two have $P=64$.
2. **The 1.5B has 20 directions.** Its quantiles are coarse and its ECDF is a staircase. A bootstrap over those 20 gives a median interval 1.41-1.99; the 1.5B-vs-Qwen3.5 and 1.5B-vs-phi-4 gaps (2.2x and 2.3x) sit just outside it, but the directions are one simplex block and the interval is not a population confidence statement.
3. **$r_*$ is a linear object.** It is computed from one radius and calibrated in a finite window; at $r=1$ the 1.5B's observed decision flips (8) exceed the linear prediction (4). Do not read the far tail as measured.
4. **The 14B has two runs.** The core column is true-window-100. The full-attention run gives 0.1164 instead of 0.1035 and a different controlling gate (21 instead of 19 for the median). The hierarchy is unchanged, but a 14B number without a run label is ambiguous.
5. **The angle is measured on 24 directions, not 1000,** and only for the models that store `h0_last`. The 1.5B value is literature; the full-attention 14B cannot be measured.
6. **Non-normality is one model.** phi-4's is pending; Qwen3.5 has none. The "last block is more non-normal at 14B" is not a four-model law.
7. **Depth is absent for Qwen3.5 and sampled at 8 tokens.** The layer profile is a response measurement, not a tangent measurement, and is not substituted.
8. **D2 is absent for phi-4.** TwoNN is not a dimension and is not converged; the 1.5B and 14B/Qwen3.5 use different $N$, $d$ and prompt sets.
9. **The floor's normalization has a history.** The established 14B law used full-attention state norms; the matched true-window norms (which differ by up to 19% at layer 12) give a fraction up to 16% lower. The order-of-magnitude conclusion is unchanged.
10. **This is a synthesis of four models, not a scaling law.** The 9B is not "between" the two DeepSeek models in any monotone size sense; architecture, tokenizer, training data and prompt length all differ. The one variable that orders the median radius here is the controlling margin, and that is a description, not a derivation.

### 6.5 Pending

- **Qwen3-14B.** `results_big/qwen314/a1_inter_w0/a1.npz` was absent at run time. The script checks for it and for `free_inter_w0`; when present, it is added to every applicable panel and to `syn_numbers.json` automatically. The column is omitted rather than guessed.
- **phi-4 non-normality.** `nonnorm_w0/nonnorm.npz` is still generating (fused-MLP Jacobian fix). The panel marks it pending.
- **Qwen3.5 depth.** No JVP run exists; the layer profile is not a substitute.
- **phi-4 D2.** No distance matrices in `results_big/phi4`.
- **The 1.5B response-vs-state angle.** The T4 dump did not store `h0_last`; a re-run that does would replace the literature 87° with a measured value.

### 6.6 Reproduction

`python scripts_spark/figs_synthesis.py` reads the local npz/json files in place, writes `notebook_big/figures/syn_{fragility,stability,angle_depth,dimension}.png` (dpi 170) and `syn_numbers.json`, and prints the headline table. The script is deterministic and does not modify any result file. The 1.5B A1 file and its free-run file are read from the read-only `chaos_llm_2026` tree. The Qwen3-14B column is conditional on `os.path.exists(results_big/qwen314/a1_inter_w0/a1.npz)` (and the matching free-run file), so a later pull needs no code change, only a rerun.

The headline synthesis table, repeated for the reader who wants it without the prose:

| | 1.5B | DS-14B | Qwen3.5-9B | phi-4 |
|---|---:|---:|---:|---:|
| median $r_*$ | 1.555 | **0.1035** | 0.707 | **3.587** |
| median controlling margin $m_*$ | $6.90\times10^{-2}$ | $1.14\times10^{-3}$ | $2.05\times10^{-2}$ | $1.70\times10^{-2}$ |
| median controlling sensitivity $s_*$ | $2.42\times10^{-2}$ | $5.54\times10^{-3}$ | $3.31\times10^{-2}$ | $7.67\times10^{-3}$ |
| margin at median-$r_*$ direction | $2.04\times10^{-2}$ | $1.14\times10^{-3}$ | $7.27\times10^{-3}$ | $6.31\times10^{-2}$ |
| sensitivity at median-$r_*$ direction | $1.30\times10^{-2}$ | $1.10\times10^{-2}$ | $1.03\times10^{-2}$ | $1.76\times10^{-2}$ |
| factor from margin (vs 1.5B) | 1 | $1/17.8$ | $1/2.81$ | 3.09 |
| factor from sensitivity (vs 1.5B) | 1 | $1/0.85$ | 1.27 | 1.35 |
| net $r_*$ factor (vs 1.5B) | 1 | $1/15.1$ | $1/2.22$ | 2.29 |
| flips at $r=0.1$ (free) | 0/20 at $r\le0.01$ | 483/1000 | 12/1000 | 0/1000 |
| re-synchronised | 0 (literature) | 0/1186 | 0/169 | 2/4 |
| angle | 87° (lit.) | 89.99° | 90.08° | 89.85° |
| TwoNN | 5.44-6.23 | 4.66-8.90 | 6.05-8.09 | not run |
| loops | 9+2/20 | 0/13 | 0+1/12 | not run |
| floor, ulp | 15 (last) | 18-24 | 12-26 | 21-29 |

The single sentence to carry away: **the 14B is fragile because its margins are thin, the 1.5B is robust because its margins are fatter, and phi-4 is the most robust because its thin ties sit on quiet responses; the response axis matters at the tail, where the 1.5B's worst direction is unusually loud.**
