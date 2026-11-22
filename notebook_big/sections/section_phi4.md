# phi-4: a thin tie that does not matter, and the widest flux tube of the four models

## Summary

phi-4 is a **40-block, width-5120 dense-attention model** (`Phi3ForCausalLM`, vocabulary 100352) with **no sliding window** (documented null; the window field is 0, so window experiments are undefined). On the same prompt and protocol as the other three models — `interstellar_travel`, $P=64$ tokens, 300 greedy tokens, $r=0.1$, 1000 directions — it is the **least flippable model in the campaign**.

Its calibration run (400 tokens) has a smallest top-1/top-2 gap of $9.24\times10^{-3}$ nats at $t=277$, then $1.70\times10^{-2}$ at 219, $2.77\times10^{-2}$ at 53, $5.67\times10^{-2}$ at 34 and $5.94\times10^{-2}$ at 186. That minimum is **95x thicker than the 14B's** $9.73\times10^{-5}$ but only **1.3x thicker than Qwen3.5-9B's** $7.27\times10^{-3}$: on the margin spectrum alone phi-4 is a sibling of Qwen3.5, not of the 14B. Its fp32 floor is $1.51\times10^{-3}$ and the linear window is $r\in[0.01,0.3]$.

What separates it is the **response**. Its per-direction gain $G$ has median **0.383**, the **lowest of the four** (14B 0.583, Qwen3.5 0.822, 1.5B 1.333), and its calibration plateau at the last saved state is 0.375 (Qwen3.5 0.502 on the same measure). At the steps that actually control the threshold the local sensitivity is $0.0077$ per unit radius, **4.3x below Qwen3.5's 0.0331** while the controlling margin is almost identical ($0.0170$ vs $0.0205$ nats). The flux-tube radius $r_*$ therefore has median **3.59** (p10 1.20, p90 10.66): **31x the 14B's 0.116, 5.1x Qwen3.5's 0.707 and 2.3x the 1.5B's 1.56**. **No direction in 1000 flips anything at $r=0.1$**; two flip at $r=0.3$; 92% need $r>1$.

The layer profile is monotone but weaker per block than Qwen3.5's: **131x growth over 38 blocks** (1.137x/block) across the 13 saved points, and all **12** intervals are positive. The state norm grows **196x** over the same span, so the **state-normalized response peaks at block 2 and then decays 2.4x** — the absolute growth is mostly radial/state-scale growth, the same lesson as the 1.5B and 14B.

Free-run validation is a null at $r=0.1$: **0 observed, 0 predicted**. At $r=0.3$ there are **4 observed flips vs 2 signed predictions** (5 sign-blind), a 99.5% classification that is carried by the 995 both-none directions. All four flips are top-2 word swaps, and — unlike Qwen3.5 (0/169) and the 14B (0/19), where no flipped row re-synchronised — **2 of the 4 are one-token substitutions that do re-join the baseline**, while the other 2 diverge permanently.

The response-vs-state angle is **89.85°** (IQR 85.5-93.5°), consistent with the near-perpendicular picture, but noisier here because the response is only $9.7\times10^{-5}$ of the state norm.

**Pending in this draft.** `state_w0/`, `depth_w0/`, `nonnorm_w0/`, `a1_childhood_w0/` and `a1_cyber_w0/` were still running on the GPU when this was written. Their sections are marked **pending**; `scripts_spark/figs_core_phi4.py` detects them with `Test-Path`-equivalent checks and will render `phi4_state.png`, `phi4_depth.png`, `phi4_nonnorm.png`, `phi4_generality.png` and fill the corresponding JSON blocks on the next run.

---

## 1. What was measured

### 1.1 Setup

| item | value |
|---|---|
| model | `phi-4`, `Phi3ForCausalLM`, FP32 |
| blocks / width | 40 / $d=5120$; vocabulary 100352 |
| attention | full; sliding window **none** (documented null, `window=0`) |
| prompt | `interstellar_travel`, $P=64$ tokens |
| calibration | 8 directions x 7 radii, 400 greedy tokens, prompt perturbation |
| A1 | 1000 directions, $r=0.1$, $T=300$, teacher-forced on the baseline tokens |
| free run | same 1000 directions, greedy, own KV cache, $\rho\in\{0.1,0.3\}$ |
| files used here | `results_big/phi4/calib/calib.npz`, `results_big/phi4/a1_inter_w0/a1.npz`, `results_big/phi4/free_inter_w0/{free_run.npz,summary.json}` |
| files pending | `state_w0/state2.npz`, `depth_w0/depth.npz`, `nonnorm_w0/nonnorm.npz`, `a1_{childhood,cyber}_w0/a1.npz` |

The A1 margins are self-consistent with the baseline tokens: `top1[t] == gen_ids[t]` for all 300 steps. The calibration and A1 runs produce **bit-identical generated tokens** (both greedy from the same prompt) and margins agreeing to $4.96\times10^{-5}$ nats, so they describe one trajectory.

### 1.2 Indexing: 40 blocks, 14 saved states

This is the most important deviation from the Qwen3.5 and 14B runs, which saved 32 and 9 states.

| object | convention (verified against `src/chaos_llm/big_engine.py:Capture`) |
|---|---|
| state $0$ | embedding output; **identically zero response** (the perturbation touches prompt positions only) |
| state $l\ge1$ | output of **block $l-1$** |
| saved states | $0,1,2,3,5,8,12,16,20,24,28,32,36,39$ |
| saved blocks | $0,1,2,4,7,11,15,19,23,27,31,35,38$ |
| block 39 (output state 40) | **not in `dh_norm`**; its post-final-norm response is available for 24 directions in `dh_last` |
| `dh_last` / `h0_last` | final-norm output, 24 directions x 300 steps x 5120 |

All panels and tables label by **block index $b$**, not state. Because the saved states are sparse, the layer profile has 13 points and the per-block growth rate is computed per *saved interval* (dividing the log ratio by the number of blocks in the interval). No statement is made about unsampled blocks.

### 1.3 Quantities

| symbol | explicit expression | meaning | why this quantity |
|---|---|---|---|
| $m_t$ | $\ell_t^{(1)}-\ell_t^{(2)}$ | top-1 minus top-2 logit gap of the unperturbed run (nats) | how close a greedy decision was |
| $\delta m_t(u)$ | $m_t(u)-m_t(0)$ at $r=0.1$ | change of the gap under direction $u$ | first-order response of the decision |
| $r_c^-(t,u)$ | $m_t\,r/\max(-\delta m_t(u),0)$; $+\infty$ if $\delta m_t\ge0$ | radius at which step $t$ would flip | signed threshold; a positive $\delta m$ widens the gap and cannot flip |
| $r_*(u)$ | $\min_t r_c^-(t,u)$ | flux-tube radius | smallest radius at which $u$ flips anything in 300 tokens |
| $t_*(u)$ | $\arg\min_t r_c^-(t,u)$ | controlling step | which decision is the weak point |
| $s(u)$ | $-\delta m_{t_*}(u)/r$ | local sensitivity at the controlling step | the denominator of $r_*=m_{t_*}/s$ |
| $G(u)$ | $\mathrm{median}_t\|\delta h_t^{\rm last}(u)\|/r$ | response gain per unit radius | how strongly a direction moves the last saved state |
| $\mathrm{angle}(u,t)$ | $\arccos\!\big(\langle\delta h_t^{\rm last},h_{0,t}^{\rm last}\rangle/(\|\delta h\|\,\|h_0\|)\big)$ | angle between response and baseline state | tests the near-perpendicular claim |

The npz's stored `r_c` and `pred_T` use $|\delta m|$ (sign-blind). Everything below uses the **signed** rule recomputed from $m$ and $\delta m$; the sign-blind variant is reported only as a comparison.

---

## 2. Figure `phi4_calib.png` — a 0.0092-nat tie, a low floor, and the weakest gain of the four

![calibration](figures/phi4_calib.png)

**Setup.** Eight unit directions on the prompt embedding, seven radii from $10^{-3}$ to 1, 400 greedy tokens, teacher-forced on the model's own continuation. Response read at state 39 = output of block 38. Left panels: unperturbed margins. Right panels: $\mathrm{median}_{u,t\in[100,400)}\|\delta h_t^{\rm last}\|$; gain is that divided by $r$.

**Axes.** (a) $x$ generated step $t$; $y$ $m_t$ (nats, log). (b) $x$ $m_t$ (log); $y$ fraction of steps with margin $\le m$. (c) log-log: $x$ radius $r$; $y$ median $\|\delta h\|$; dotted guides have slope 1 through the $r=0.1$ point. (d) $x$ radius; $y$ $\|\delta h\|/r$; shaded band = clean window $r\in[0.01,0.3]$, red dotted line = campaign radius.

| $r$ | $10^{-3}$ | $3\times10^{-3}$ | $10^{-2}$ | $3\times10^{-2}$ | $0.1$ | $0.3$ | $1$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| $\|\delta h\|$ (state 39) | 1.51e-3 | 1.84e-3 | 4.00e-3 | 1.13e-2 | 3.75e-2 | 1.13e-1 | 4.10e-1 |
| $\|\delta h\|/r$ | 1.51 | 0.614 | **0.400** | **0.377** | **0.375** | **0.377** | 0.410 |

- **No razor tie, but the minimum is not the interesting number.** Over 400 steps: median **3.782**, mean 6.212, p10 0.516, minimum $9.24\times10^{-3}$ at 277; 11 steps below 0.1 nats and **1 below 0.01**. The five smallest are 277 (` energy`/` reactors`, 9.24e-3), 219 (` notable`/` historical`, 1.70e-2), 53 (` metrics`/` parameters`, 2.77e-2), 34 (`'s`/`,`, 5.67e-2), 186 (` further`/` significant`, 5.94e-2). There is no 14B-style knife edge, but the thinnest tie is only 1.3x Qwen3.5's, so on margins alone the two models are close.
- **A log discrepancy, flagged rather than copied.** The run log quoted a calibration median of "~1.2-1.5". The file's 25th percentile is 1.355 and its median is **3.782** (A1: p25 1.238, median 3.473); 1.2-1.5 is the quartile, not the median. All values in this section are the file's values.
- **Floor and window.** At $r\le3\times10^{-3}$ the last-state response is pinned at $1.5$-$1.8\times10^{-3}$; this is the fp32 floor, 24.9x below the $r=0.1$ signal (14B 17.5x, Qwen3.5 82x). The gain is flat to 1% across $[0.01,0.3]$; at $10^{-2}$ it is 6.1% above the plateau and at $r=1$ 8.7% above it, so the honest window is $[0.01,0.3]$ and the campaign radius sits in the middle.
- **Gain.** The plateau is **0.377** nats/unit at the last saved state, against Qwen3.5's 0.502 and the 14B's 0.583 at their last saved states. This is the quantitative statement that phi-4 is "hard to move", not just "well decided". The same flat plateau appears at the other saved states: 0.0218 (block 9), 0.0618 (block 19), 0.177 (block 29) at $r=0.1$.

---

## 3. Figure `phi4_core.png` — a 3.6-unit flux tube explained by a 4x lower sensitivity

![margins and thresholds](figures/phi4_core.png)

**Setup.** A1 run only ($r=0$ margins; $r=0.1$ responses), 1000 directions, 300 tokens. Signed threshold throughout.

**Axes.** (a) $x$ step $t$; $y$ $m_t$ (log); five smallest marked. (b) $x$ $r_*$ (log); $y$ fraction of directions with $r_*\le r$; dashed red = run radius; dotted lines = the other models' medians. (c) horizontal bars: steps controlling the most directions. (d) $x$ $G(u)$ (log); left $y$ counts, right $y$ ECDF; red line median, band IQR. (e) $x$ $\cos(\delta h,h_0)$ at the final-norm state; $y$ direction x step pairs. (f) $x$ local sensitivity $s$ at the controlling step (log); $y$ controlling margin $m_{t_*}$ (log); dashed lines are iso-$r_*$; diamonds are the medians of phi-4, Qwen3.5 and the 14B computed with the same signed rule.

**Margins.** A1 median 3.473, mean 5.884, p10 0.488; 11/300 steps below 0.1 and 1/300 below 0.01. The five smallest are the calibration's five (277, 219, 53, 34, 186).

**Flux tube.**

| $r_*$ | p10 | p25 | p50 | p75 | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| value | 1.201 | 2.009 | **3.587** | 6.467 | 10.66 | 14.43 | 0.165 | 45.29 |

**0/1000 directions cross at $r=0.1$ and 2/1000 at $r=0.3$; 923/1000 need $r>1$ and 111/1000 need $r>10$.** The 14B's p50 is 31x smaller, Qwen3.5's 5.1x smaller, the 1.5B's 2.3x smaller.

**Danger ladder.** The thresholds are controlled by **27 distinct steps**, and the five thinnest steps control **79.9%** of all directions:

| step | $m_t$ | all directions | flip at $r=0.1$ |
|---:|---:|---:|---:|
| 219 | 1.70e-2 | 273 | 0 |
| 277 | 9.25e-3 | 253 | 0 |
| 34 | 5.67e-2 | 95 | 0 |
| 186 | 5.94e-2 | 91 | 0 |
| 53 | 2.76e-2 | 87 | 0 |
| 50 | 9.80e-2 | 44 | 0 |
| 142 | 6.28e-2 | 36 | 0 |
| 147 | 8.21e-2 | 35 | 0 |
| 253 | 6.30e-2 | 17 | 0 |

The thinnest tie is not the most controlling one: **219 controls more directions than 277** because the response at 277 is smaller. That inversion is the first hint that the margin is not the whole story.

**Gain.** $G$: p5 0.209, p25 0.287, p50 **0.383**, p75 0.584, p95 1.052; min 0.155, max 3.14, a **20.3x spread** — the widest of the four (Qwen3.5 15.1x, 14B 13.7x, 1.5B 11.8x). The median is 1.5x below the 14B, 2.1x below Qwen3.5 and 3.5x below the 1.5B.

**The decomposition (panel f) is the answer to "why".** Per direction $r_*=m_{t_*}/s$ exactly. The population medians are:

| model | median $m_{t_*}$ (nats) | median $s$ (per unit $r$) | median $r_*$ |
|---|---:|---:|---:|
| DS-R1-14B | 1.14e-3 | 5.54e-3 | 0.103 |
| Qwen3.5-9B | 2.05e-2 | 3.31e-2 | 0.707 |
| **phi-4** | **1.70e-2** | **7.67e-3** | **3.59** |

phi-4 and Qwen3.5 have **almost the same controlling margin** (0.017 vs 0.020 nats, 17% apart), but phi-4's sensitivity is **4.3x lower**, which is what produces the 5.1x wider flux tube. The 14B is different in both terms: its controlling margin is 15x thinner, and that is the dominant term. The thin tie at 277 is shared by 253 directions but its typical sensitivity is only $3.4\times10^{-3}$ per unit radius, so a typical direction there needs $r\approx2.7$ to flip it. The model is not fragile where it is uncertain; it is uncertain where it is hard to move.

**Gain structure.** Across the 24 stored final-norm response vectors the mean pairwise cosine is $-0.003$, but **25.7%** of pairs exceed $|0.1|$ (Qwen3.5: 6.9%) and the first PC carries **12.6%** of the variance against 4.2% for an isotropic 24-vector cloud — a modest shared mode, stronger than Qwen3.5's 6.9%. The temporal cosine is 0.342 at lag 1, 0.128 at lag 10 and 0.074 at lag 50: direction decorrelates within about ten steps.

**Angle.** The median angle between $\delta h$ and the baseline final-norm state is **89.85°** (mean 89.54°), IQR 85.54-93.51°, p1-p99 62.9-116.4°, with **48.7% beyond 90°** and 52.4% inside $87^\circ\pm5^\circ$. The per-direction medians span 84.1-98.2° and the per-step medians 76.5-106.0°. The perpendicular picture survives, but it is **less sharp than Qwen3.5's 90.08° (IQR 89.5-90.9°)**: here $\|\delta h\|/\|h_0\|$ is only $9.7\times10^{-5}$, so the angle of a small vector against a large one is noisier. The stored $\cos(h',h_0)$ stays at 0.999988 or above, so that field cannot be used for the angle; the 24 `dh_last` vectors and `h0_last` can.

---

## 4. Figure `phi4_layer_profile.png` — 131x monotone growth, but the normalized response decays

![layer profile](figures/phi4_layer_profile.png)

**Setup.** A1 run, 1000 directions at $r=0.1$, 300 tokens, the 14 saved states. Plotted: $\mathrm{median}_u\|\delta h_t^b\|/r$ with $b$ the block whose output is state $b+1$. State 0 is identically zero and is omitted.

**Axes.** (a) heatmap: $x$ step $t$; $y$ saved state (block); colour $\log_{10}$ median response. (b) $x$ block index $b$; $y$ median $\|\delta h\|/r$ (log); band = 10-90% over $t$. (c) $x$ saved-state interval; $y$ response ratio per block; dashed line = mean. (d) $x$ block; $y$ $\|\delta h\|/\|h_0\|$ ($\times10^{-3}$).

| block | 0 | 1 | 2 | 4 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| median $\|\delta h\|/r$ | 0.0031 | 0.0075 | 0.0107 | 0.0170 | 0.0265 | 0.0370 | 0.0527 | 0.0743 | 0.112 | 0.171 | 0.251 | 0.337 | 0.406 |
| growth/block to next | 2.43 | 1.42 | 1.26 | 1.16 | 1.088 | 1.093 | 1.089 | 1.109 | 1.112 | 1.100 | 1.076 | 1.064 | — |

- **Growth is monotone and front-loaded.** The response rises **131x from block 0 to block 38**, a mean of **1.137x per block**, and **12 of 12 saved intervals are positive**. The two largest increments are the first two blocks (2.43x, 1.42x); from block 7 onward the per-block rate is 1.06-1.11. This is the same qualitative shape as Qwen3.5 (159x over 30 blocks, 1.18x/block), slightly weaker per block.
- **The normalized response turns over.** The state norm grows **196x** over the same 38 blocks, but the response grows only 131x. $\|\delta h\|/\|h_0\|$ rises from $1.02\times10^{-3}$ at block 0 to a peak of $1.66\times10^{-3}$ at block 2, then **falls monotonically to $6.8\times10^{-4}$ at block 38, a 2.4x decline from the peak**. The absolute growth is not an angular instability: after the first two blocks the residual stream grows faster than the perturbation.
- **Time decay is weaker at depth.** The ratio of the median response at $t\in[250,300)$ to $t\in[0,50)$ is 0.20 at block 0, 0.33 at block 2, and 0.48-0.51 at blocks 19-38. Near the prompt the response is 2-5x larger than late; the late response is relatively better preserved deep in the network. Qwen3.5 showed a roughly constant 0.2-0.34 at all depths.
- **Block 39 is missing.** The saved profile stops at block 38 (state 39). Block 39's output is state 40 and is not in `dh_norm`; its post-final-norm response exists for 24 directions, but the RMSNorm makes it not comparable to the pre-norm profile. No claim is made about the last block.

---

## 5. Figure `phi4_free.png` — stable at $r=0.1$, four one-token events at $r=0.3$

![free validation](figures/phi4_free.png)

**Setup.** Same 1000 directions and prompt perturbation, now generated greedily with the model's own KV cache for 300 tokens at $\rho=0.1$ and $0.3$. `obs` is the first generated token that differs from the baseline; `pred` is recomputed here with the signed rule from A1. The npz's stored `pred` is sign-blind and is not used for the scorecard.

**Axes.** (a) stacked class counts per radius; labels give observed/predicted flips and both-none. (b) observed vs signed vs sign-blind flip percentage. (c) longest identical post-flip run per flipped direction (re-sync threshold 20). (d,e) survival $S(t)=P(\text{first flip}>t)$, $y$ zoomed to 0.994-1.0005; black observed, orange signed prediction; dotted verticals at 171/219/277. (f) black where the free continuation differs from the baseline, for the four flipped rows.

| | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|
| observed flips / 1000 | **0** | **4** |
| predicted, signed | 0 | 2 |
| predicted, sign-blind | 0 | 5 |
| exact / wrong step / false alarm / miss | 0 / 0 / 0 / 0 | 0 / 1 / 1 / 3 |
| both none | 1000 | 995 |
| classification accuracy | **100% (trivial)** | **99.5%** |
| observed mean / median flip step | — | 209.5 / 195 |
| max survival deviation | 0.000 | 0.003 (at $t=219$) |
| rows with a $\ge20$-token identical run | — | **2 / 4** |

All four observed flips are **top-2 word swaps**, decoded with the phi-4 tokenizer:

| step | baseline $\to$ free | directions | A1 predicted step | outcome |
|---:|---|---:|---:|---|
| 171 | ` the` $\to$ ` high` | 84, 701 | — | permanent divergence (128 later differences each) |
| 219 | ` notable` $\to$ ` historical` | 276 | 277 | one-token substitution; re-syncs for 57 tokens, then diverges again at **277** |
| 277 | ` energy` $\to$ ` reactors` | 933 | — (A1 $r_*=0.452$) | one-token substitution; identical tail for 22 tokens |

- **At $\rho=0.1$ there are no events at all**, so the signed rule is perfect by construction and carries no step information. The informative null is that it does not hallucinate a flip: 1000/1000 both-none, survival exactly 1.000 at every $t$.
- **At $\rho=0.3$ the signed rule underpredicts the count (2 vs 4) while the sign-blind rule overpredicts (5 vs 4).** The signed rule's two predictions are directions 276 and 652 at $t=277$. Direction 276 does flip, but first at 219 as a transient (counted as "wrong step"); direction 652 does not flip (false alarm). The three misses are 84 and 701 at 171 and 933 at 277. So the event set is not reproduced even though the classification accuracy is 99.5%.
- **Two misses are far outside the linear rule.** Directions 84 and 701 cross at $t=171$, where their teacher-forced thresholds are $r_c=2.54$ and $1.69$ — 5.6x and 8.5x above the free radius. A 0.3 perturbation would need a 6-8x larger local response at that step than the A1 pass measured, which is a nonlinear/compounding effect, not a small threshold error. Both flip to the same token (` high`) but then follow different continuations. The 933 flip is a near-threshold case ($r_*=0.452$, $r=0.3$, 1.5x short).
- **Re-synchronisation is not zero, contrary to the expectation.** 2 of 4 flipped rows have a $\ge20$-token identical post-flip run: direction 276 for 57 tokens, direction 933 for 22 (the remaining tail). Both are **single-token substitutions**, not a return to the same trajectory — direction 276 re-diverges at 277. The other two rows (84, 701) diverge permanently. So the correct statement is "no generic re-sync, but 2/4 events at this radius are one-token substitutions that do re-join". This is a small-sample result: four events cannot support a rate.
- **The post-flip agreement averages 0.497**, but that is an artefact of averaging two permanent zeros with two nearly-identical rows; the median is not meaningful at $n=4$. The late agreement ($t>{\rm obs}+45$) is 0.324 for the same reason.

---

## 6. Pending: depth, non-normality, state kicks, generality

These blocks were still running when this draft was written. The analysis script already implements them and will produce the figures when the files appear; no numbers are quoted here.

**6.1 `phi4_depth.png` — pending.** The run is `depth_w0/depth.npz`: 8 tokens $0,21,50,100,150,200,250,299$, $k=8$ tangent directions, 40 blocks, QR after every block. Planned content: $\lambda_i(l)$ per token, $h_{\rm depth}$, per-block local gain against state-norm growth, the absolute and relative exponents, and the final spectrum. The relative test is better resolved for phi-4 than for the 14B: `norm_h0` has 14 states, giving **12 intervals** (1, 1, 2, 3, then 4s, last 3 blocks) rather than eight 6-block chunks, so "contracts" can be tested at 1-4 block resolution over blocks 0-38.

**Protocol caveat for the razor-tie question.** The brief asks whether "the razor-tie token" is a tangent hotspot like the 14B's $t=21$. For phi-4 the thin ties are at **277 and 219**, and neither is among the eight depth tokens; token 21 is in the depth set but is not a phi-4 tie ($m_{21}$ is an ordinary margin). The hotspot question can therefore only be answered by proxy (e.g. token 250, the closest to 277) unless the depth run is repeated with 219/277 in the token list. This is a protocol mismatch, not a data problem.

**6.2 `phi4_nonnorm.png` — pending.** `nonnorm_w0/nonnorm.npz`: MLP analytic SwiGLU Jacobian at 5 tokens x 9 blocks, full-block power iteration/Arnoldi at 5 tokens x 5 blocks, principal angles 4 blocks apart. Planned: $\sigma_{\max}$, $\rho$ and $\kappa=\sigma_{\max}/\rho$ by block, the $\sigma_{\max}$-$\rho$ plane, and the angle histogram against the 1.5B's 83.6° and the 14B's 86.96°. The reference pattern to test is the U-shape: extreme at block 0, normal in the middle, extreme again at the last block.

**6.3 `phi4_state.png` — pending.** `state_w0/state2.npz`: single-position kicks at $s=8,16,32,50,75,100,150,200,250,299$, 24 directions, radii 0.03 and 0.1, read at the last saved state. Planned: immediate response, half-life, late/immediate ratio, the fraction of directions whose first crossing is at $r\le1$, the median first-crossing step (the latency question), and response trajectories. Because phi-4 has no near-tie at 219/277 before the late sequence, the late gate at 277 is the natural target for kicks at $s\ge250$.

**6.4 `phi4_generality.png` — pending.** `a1_childhood_w0/` and `a1_cyber_w0/`: 200 directions at $r=0.1$ on two other prompts. Planned: margin spectra and $r_*$ ECDFs against the interstellar run of the same model, plus the flip fraction at 0.1/0.3. The question is whether the **gates move** (the controlling margin and its step) when the text changes, and whether the low sensitivity is a property of the model or of this prompt.

**6.5 Trajectories.** `traj/` is for the separate D2 analysis and is not part of this section.

---

## 7. Cross-model comparison

All rows use the same prompt, greedy decoding, FP32 and $r=0.1$ unless stated. 1.5B and 14B values are from the published sections; Qwen3.5-9B from its section; phi-4 from this one.

| quantity | 1.5B | DS-R1-14B | Qwen3.5-9B | **phi-4** |
|---|---:|---:|---:|---:|
| blocks / width | 28 / 1536 | 48 / 5120 | 32 / 4096 | **40 / 5120** |
| attention | full | full | 3 linear : 1 full | **full** |
| sliding window | 100 | 100 / full* | none | **none (null)** |
| $K$ directions | 20 | 1000 | 1000 | **1000** |
| margin min (step) | 2.02e-2 (123) | 9.73e-5 (21) | 7.27e-3 (267) | **9.24e-3 (277)** |
| margin median | 4.78 | 3.10 | 2.46 | **3.47** |
| $r_*$ p50 / p90 | 1.56 / 4.12 | 0.116 / 2.59 | 0.707 / 1.96 | **3.59 / 10.66** |
| fp32 floor (last saved) | 1.0e-3 | 2.7e-3 | 6.1e-4 | **1.51e-3** |
| clean radius window | [1e-3, 0.1] | [0.03, 0.3] | [0.01, 0.3] | **[0.01, 0.3]** |
| gain $G$ p50 (range) | 1.33 (11.8x) | 0.583 (13.7x) | 0.822 (15.1x) | **0.383 (20.3x)** |
| flip fraction at $r=0.1$ | 0/20 | 45.3% | 1.2% | **0/1000** |
| free flip fraction 0.1 / 0.3 | 0 / — | 45.3% / 64.3% | 1.2% / 15.7% | **0% / 0.4%** |
| response-vs-state angle | ~87° (claimed) | not recoverable | 90.08° | **89.85°** |
| prediction at $r=0.1$ | 19/20 exact | 98.2% acc | 12/12 exact | **0/0 (100% trivial)** |

\* the 14B's window-100 setting was silently inactive in the runs analysed before the fix; see the correction note in `section_a1_deep.md`. phi-4 and Qwen3.5 have no window at all, so no window comparison is meaningful.

Reading the table:

- **Margin thickness.** phi-4's median margin (3.47) is 12% thicker than the 14B's and 40% thicker than Qwen3.5's, but 27% thinner than the 1.5B's. Its minimum is 95x the 14B's yet only 1.3x Qwen3.5's. So by margins it is "mid-pack": it has neither the 14B knife edge nor the 1.5B's very thick minimum.
- **Threshold radius.** It has the **widest** flux tube: p50 31x the 14B, 5.1x Qwen3.5 and 2.3x the 1.5B. At the campaign radius this is the difference between 45% (14B), 1.2% (Qwen3.5), 0/20 (1.5B) and **0/1000**.
- **Gain.** It has the **lowest** response gain of the four: 1.5x below the 14B, 2.1x below Qwen3.5, 3.5x below the 1.5B. Its 20.3x spread is the widest, so direction choice matters more than in any other model.
- **Floor.** Its floor (1.51e-3) sits between Qwen3.5's (6.1e-4) and the 14B's (2.7e-3); signal-to-floor at $r=0.1$ is 24.9x (14B 17.5x, Qwen3.5 82x).
- **Angle.** The near-perpendicular response is reproduced (89.85°), but with an 8° IQR rather than Qwen3.5's 1.4°.

---

## 8. Where phi-4 sits in the fragility hierarchy, and why

**Ordering (most fragile first):** DS-R1-14B $\gg$ Qwen3.5-9B $>$ 1.5B $>$ **phi-4**. In one number per pair: flips at $r=0.1$ are 45.3%, 1.2%, 0/20 and **0/1000**; $r_*$ medians are 0.116, 0.707, 1.56 and **3.59**.

The decomposition $r_*=m_{t_*}/s$ shows that these two factors are not interchangeable:

| | 14B | Qwen3.5 | phi-4 |
|---|---:|---:|---:|
| controlling margin $m_{t_*}$ | 1.1e-3 | 2.0e-2 | 1.7e-2 |
| local sensitivity $s$ | 5.5e-3 | 3.3e-2 | 7.7e-3 |
| $r_*$ (median) | 0.10 | 0.71 | 3.59 |

- **The 14B is fragile because its decisions are thin.** Its controlling margin is 15x below phi-4's, and its sensitivity is also slightly lower (5.5e-3 vs 7.7e-3); the margin term dominates. The razor tie at $t=21$ ($9.7\times10^{-5}$) is the extreme case.
- **phi-4 is robust because its decisions are hard to move.** Its controlling margin is within 17% of Qwen3.5's, but its local sensitivity is **4.3x lower**. The same ~0.02-nat gap that Qwen3.5 crosses at $r\approx0.7$ needs $r\approx3.6$ here. The model is not "better decided" than Qwen3.5 in any strong sense; it is **less responsive**.
- **The 1.5B is intermediate for a third reason.** It has the thickest median margin (4.78) and the highest gain (1.33), and its $r_*$ (1.56) reflects both. phi-4 beats it on $r_*$ because its population gain is 3.5x lower even though its margins are thinner.

This is a clean confirmation of the cross-scale lesson: **raw margin is a weak predictor of flippability across models**; the ratio of margin to local response is what the perturbation radius measures. The same corpus, prompt and radius produce a 45% flip rate in one model and 0% in another, and the difference is mostly the denominator.

---

## 9. Prediction scorecard

| | $r=0.1$ | $r=0.3$ |
|---|---:|---:|
| observed flips | 0 | 4 |
| signed predicted | 0 | 2 |
| sign-blind predicted | 0 | 5 |
| exact / wrong step / false alarm / miss | 0 / 0 / 0 / 0 | 0 / 1 / 1 / 3 |
| accuracy | 100% (trivial) | 99.5% |

The scorecard is **uninformative at $r=0.1$**: with zero observed and zero predicted events, 100% accuracy is a null, not a validation. It is still useful as a stability test — the signed rule does not invent an event where the model is stable, which the 14B's sign-blind rule did at low radius.

At $r=0.3$ the count is wrong in both directions depending on the sign convention (signed 2, sign-blind 5, observed 4), and the step-level classification is carried by the 995 both-none directions. The one "wrong step" (direction 276) actually does flip at the predicted gate 277 after a transient at 219, so the substantive prediction is better than the table suggests. The two 171 flips are genuine failures of the linear rule and are the interesting cases: they are large-margin decisions ($m_{171}=0.083$) that a 0.3 perturbation can only reach through a nonlinear amplification.

The honest one-line verdict: **phi-4 is so far outside the linear rule's danger zone at $r=0.1$ that the campaign cannot test the rule there; at $r=0.3$ four events are too few to establish a rate, but they show the same qualitative failure mode as the other models (early flips at moderate margins) plus a new one (one-token substitutions that re-sync).**

---

## 10. Caveats and what this does not show

1. **The layer profile is sparse.** Only 14 states are saved, covering blocks 0,1,2,4,7,11,15,19,23,27,31,35,38. The 131x growth and the "12/12 positive intervals" are statements about those 12 intervals, not about every block. A block between two saved states could contract without being seen. The last block (39) is absent.
2. **The state-normalized response uses the median baseline norm per saved state**, not per step. This is the same compromise as the 14B, but at finer resolution (12 intervals instead of 8).
3. **The log's calibration median is not the file's.** The file gives median 3.782 (p25 1.355), not ~1.2-1.5. If the run log aggregated a different statistic, that statistic is not in `calib.npz`.
4. **The angle is a 24-direction, one-state measurement.** It uses the final-norm state only. The IQR of 8° and the per-step range of 30° show that a single direction's angle can be noisy when its response is $10^{-4}$ of the state norm; the median is stable, the tails are not.
5. **The free-run scorecard has almost no power.** Zero events at 0.1, four at 0.3, and two of those are one-token substitutions. "99.5% accuracy" is 995 both-none plus five classifications; it is not a 99.5% success rate on events.
6. **The 2/4 re-sync result contradicts the "0 re-sync expected" prior.** It is reported as measured. With four events it cannot distinguish "phi-4 sometimes produces transient substitutions" from "these two directions happened to be near a one-token boundary".
7. **The decomposition uses each model's own A1 file.** The 14B point is computed from `results_big/ds14/a1_inter_w100/a1.npz` with the same signed rule; it gives a median $r_*$ of 0.103 where the published cross-scale value is 0.116 (a run-version difference), so the ordering, not the third digit, is the claim.
8. **phi-4 has no sliding window.** Window experiments are undefined (null by construction); nothing here says how a window would change the margins.
9. **Depth, non-normality, state kicks and multi-prompt generality are pending**, as described in §6. The conclusions about the fragility hierarchy rest on the calibration, A1 and free runs and do not depend on those blocks.
10. **The 40-block count and $d=5120$ are metadata facts**, not measured in the npz here; the npz confirms $d=5120$ through `dh_last` and the vocabulary 100352 through the tokenizer. The depth run, when it lands, will give an independent count of the blocks.
