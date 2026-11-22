# phi-4: a thin tie that does not matter, and the widest flux tube of the four models

## Summary

phi-4 is a **40-block, width-5120 dense-attention model** (`Phi3ForCausalLM`, vocabulary 100352) with **no sliding window** (documented null; the window field is 0, so window experiments are undefined). On the same prompt and protocol as the other three models — `interstellar_travel`, $P=64$ tokens, 300 greedy tokens, $r=0.1$, 1000 directions — it is the **least flippable model in the campaign**.

Its calibration run (400 tokens) has a smallest top-1/top-2 gap of $9.24\times10^{-3}$ nats at $t=277$, then $1.70\times10^{-2}$ at 219, $2.77\times10^{-2}$ at 53, $5.67\times10^{-2}$ at 34 and $5.94\times10^{-2}$ at 186. That minimum is **95x thicker than the 14B's** $9.73\times10^{-5}$ but only **1.3x thicker than Qwen3.5-9B's** $7.27\times10^{-3}$: on the margin spectrum alone phi-4 is a sibling of Qwen3.5, not of the 14B. Its fp32 floor is $1.51\times10^{-3}$ and the linear window is $r\in[0.01,0.3]$.

What separates it is the **response**. Its per-direction gain $G$ has median **0.383**, the **lowest of the four** (14B 0.583, Qwen3.5 0.822, 1.5B 1.333), and its calibration plateau at the last saved state is 0.375 (Qwen3.5 0.502 on the same measure). At the steps that actually control the threshold the local sensitivity is $0.0077$ per unit radius, **4.3x below Qwen3.5's 0.0331** while the controlling margin is almost identical ($0.0170$ vs $0.0205$ nats). The flux-tube radius $r_*$ therefore has median **3.59** (p10 1.20, p90 10.66): **31x the 14B's 0.116, 5.1x Qwen3.5's 0.707 and 2.3x the 1.5B's 1.56**. **No direction in 1000 flips anything at $r=0.1$**; two flip at $r=0.3$; 92% need $r>1$.

The layer profile is monotone but weaker per block than Qwen3.5's: **131x growth over 38 blocks** (1.137x/block) across the 13 saved points, and all **12** intervals are positive. The state norm grows **196x** over the same span, so the **state-normalized response peaks at block 2 and then decays 2.4x** — the absolute growth is mostly radial/state-scale growth, the same lesson as the 1.5B and 14B.

Free-run validation is a null at $r=0.1$: **0 observed, 0 predicted**. At $r=0.3$ there are **4 observed flips vs 2 signed predictions** (5 sign-blind), a 99.5% classification that is carried by the 995 both-none directions. All four flips are top-2 word swaps, and — unlike Qwen3.5 (0/169) and the 14B (0/19), where no flipped row re-synchronised — **2 of the 4 are one-token substitutions that do re-join the baseline**, while the other 2 diverge permanently.

The response-vs-state angle is **89.85°** (IQR 85.5-93.5°), consistent with the near-perpendicular picture, but noisier here because the response is only $9.7\times10^{-5}$ of the state norm.

**What the later runs add.** The depth, state-kick and multi-prompt runs have landed. The depth spectrum (8 tokens x 40 blocks, $k=8$) shows the usual split: the tangent volume grows $2.2$-$73\times$ absolutely, but the state-normalized tangent **contracts in 93 of 96 token-intervals** (median $-0.105$ nats/block), a stronger version of the 14B's 61/64. The largest absolute tangent volume is at the **first generated token** ($h_{\rm depth}=0.859$, 2.3x the next), not at either phi-4 tie — and the depth token list does **not** contain 219 or 277, so this run cannot answer the tie-hotspot question (a re-run with the tie tokens is scheduled). The single-position kicks confirm the low responsiveness: immediate responses of 1.56-8.78 per unit radius, half-lives of 1-4.5 steps, and **no flip at $r=0.1$ in any of the 240 (position, direction) kicks**. Only the $s=250$ kick reaches a gate, crossing the $t=277$ tie for 2/24 directions 27 steps later; by the crossing step its response has fallen to 9-12% of the immediate value, which makes the one-token re-synchronisations in the free run plausible. The two extra prompts leave the margin and $r_*$ distributions nearly unchanged ($r_*$ p50 2.98 and 3.16 vs 3.59; 0 flips at $r=0.1$), but the **controlling steps and the vulnerable directions move completely** — the Spearman correlation of $\log r_*$ with the interstellar run on the same 200 directions is 0.09 and $-0.07$. Non-normality is still generating (fused-MLP Jacobian fix) and remains pending.

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
| state kicks | 10 positions x 24 directions x $r\in\{0.03,0.1\}$, teacher-forced |
| depth | 8 tokens x 40 blocks, $k=8$ tangent directions, QR every block |
| multi-prompt | 200 directions at $r=0.1$ on `childhood` ($P=56$) and `cyber` ($P=61$) |
| files used here | `results_big/phi4/calib/calib.npz`, `a1_inter_w0/a1.npz`, `free_inter_w0/{free_run.npz,summary.json}`, `state_w0/state2.npz`, `depth_w0/depth.npz`, `a1_{childhood,cyber}_w0/a1.npz` |
| files pending | `nonnorm_w0/nonnorm.npz`; `traj/` (D2 distance matrices, separate analysis) |

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

## 6. Figure `phi4_state.png` — a one-position kick dies in 1-4.5 steps, and only $s=250$ reaches a gate

![state kicks](figures/phi4_state.png)

**Setup.** `state_w0/state2.npz`: one random unit vector is added to the embedding output (state 0) at a single generated position $s\in\{8,16,32,50,75,100,150,200,250,299\}$, 24 directions per position, at $r=0.03$ and $r=0.1$. The continuation is teacher-forced on the baseline tokens and the response is read at state 39 (output of block 38). The baseline margins are the A1 margins, so the thinnest decision in range is $m_{277}=0.00925$. This is a single-token input change, not a prompt-wide perturbation, and it is the cleanest test of how far one position's state can travel.

**Axes.** (a) $x$ kick position $s$; $y$ immediate response $\|\delta h_s^{39}\|/r$ (log); filled circles $r=0.1$, open triangles $r=0.03$. (b) Left $y$: median number of steps until the response first falls below half its value at $s$; right $y$ (red, log): largest later response divided by the immediate response. (c) Left $y$: percentage of the 24 directions whose smallest later threshold is $r_c\le1$; right $y$ (red, log): median and minimum $r_*$ over directions, with $s=299$ omitted because it has no later step. (d) $x$ generated step; $y$ response norm for $s=8,100,250$; dotted verticals mark the kicks, dashed the two late ties at 219 and 277.

| $s$ | immediate $/r$ | half-life | decaying | late/immediate | flips / 24 | median $r_*$ | min $r_*$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8.78 | 1 | 24/24 | 0.32 | 0 | 7.38 | 2.84 |
| 16 | 2.44 | 3 | 24/24 | 0.89 | 0 | 7.27 | 2.40 |
| 32 | 1.56 | 4.5 | 24/24 | 1.64 | 0 | 3.07 | 1.37 |
| 50 | 2.40 | 2 | 24/24 | 0.61 | 0 | 5.65 | 1.67 |
| 75 | 3.64 | 3 | 24/24 | 0.69 | 0 | 7.48 | 3.59 |
| 100 | 3.40 | 2 | 24/24 | 0.90 | 0 | 9.07 | 2.95 |
| 150 | 3.02 | 2 | 24/24 | 0.72 | 0 | 9.84 | 3.39 |
| 200 | 4.45 | 1 | 24/24 | 0.68 | 0 | 11.13 | 1.26 |
| 250 | 6.65 | 1 | 24/24 | 0.67 | **2** | 3.61 | **0.70** |
| 299 | 2.74 | — | 0/24 | — | 0 | — | — |

- **The immediate response varies 5.6x across positions** (8.78 at $s=8$ to 1.56 at $s=32$) and is linear in the radius: the per-unit values at $r=0.03$ and $r=0.1$ agree within 3.5% at every position (ratios 0.966-1.028). The 14B's two radii agreed to 19% at the worst position, so the small-radius extrapolation is better controlled here.
- **A kick is absorbed fast.** The response falls below half its immediate value in **1-4.5 steps** at every position, and all 24/24 directions decay at every position except $s=299$ (no later step). The late maximum is only **0.32-1.64x** the immediate value, against **1.8-98x** for the 14B (17-98x at five of its ten positions): there is no global late burst, only a local bump. At $s=250$ the bump is at $t\approx271$ (median 3.31 per unit radius, 50% of the immediate spike) and dies to 0.58 by $t=277$.
- **Nothing flips at the campaign radius.** At $r=0.1$ **0 of 240** (position, direction) kicks flip any token, and even the extrapolated $r_c\le1$ count is 2/240. The median flux-tube radius of a single-position kick is 3.1-11.1 across positions; only $s=250$ has a minimum below 1 (0.70), and its median (3.61) is the second-lowest after $s=32$ (3.07).
- **The latency question has one clean answer: $s=250\to t=277$, 27 steps later.** It is the only kick that reaches a gate, and it reaches it for **2 of 24 directions** with thresholds 0.70 and 0.995. At $t=219$ the response is non-positive for all 24 directions ($r_c=+\infty$), so the 219 tie — the step that controls the most prompt-wide directions in A1 — is *widened*, not narrowed, by this kick. The 277 crossing is carried by the secondary bump at 270-271 and then by the decay tail. This is the same first-crossing-versus-argmin distinction seen in the 14B: the reachable gate is not the thinnest one ($m_{277}=0.0093$ vs $m_{219}=0.0170$), because the response at 277 is the one that survives.
- **The decay makes the free-run re-syncs plausible.** At the 277 crossing the median response is **8.7%** of the immediate value, and the two flipping directions are at 9.2% and 11.9%; by $t=299$ the median has fallen to **6.3%**. A perturbation that is already at a tenth of its initial amplitude when the token changes has little left with which to steer the following tokens, so a one-token substitution that rejoins the baseline is what a decaying local kick should produce. This is a consistency argument, not a mechanism proof: the free-run events were prompt-wide perturbations with a persistent component, and direction 276 did re-diverge at 277 because that component was still there. A one-position kick has no such persistent component.

---

## 7. Figure `phi4_depth.png` — absolute growth in 40 blocks, contraction in 93/96 normalized intervals

![depth spectrum](figures/phi4_depth.png)

**Setup.** `depth_w0/depth.npz`: $k=8$ orthonormal tangent vectors at one generated token, propagated through all **40 blocks** with a JVP and QR-renormalised after every block (Benettin). Tokens are $0,21,50,100,150,200,250,299$. $\lambda_i(l)=(1/(l+1))\sum_{b\le l}\log|R_{ii}(b)|$ is the running mean per-block tangent exponent, and $h_{\rm depth}=\sum_i\max(\lambda_i(39),0)$ is the total positive final exponent. The state normalization uses the baseline `norm_h0` from the A1 run at its 14 saved states, giving **12 intervals** of 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4 and 3 blocks. The final block (39) is in the tangent growth but not in the norm growth — state 40 is not saved — so the last interval's contraction is if anything slightly conservative.

**Axes.** (a,b) $x$ block $l$; $y$ running exponent $\lambda_i(l)$ (symlog, linear within $\pm10^{-2}$ so the sign is visible); one line per token, median over the 8 directions. (c) $x$ probe token; $y$ $h_{\rm depth}$ (nats); token 0 in red. (d) $x$ block; $y$ growth per block (log): blue = median local tangent gain with the 25-75% band over tokens, red = state-norm growth per block from the saved baseline states. (e) $x$ block; $y$ log gain per block: grey = absolute exponent $e_l$, blue = relative exponent (tangent minus state-norm growth) at interval centres. (f) $x$ tangent direction; $y$ final $\lambda_i$ per token, token 0 red.

### 7.1 The numbers per token

| token $t$ | $h_{\rm depth}$ | median final $\lambda_i$ | tangent growth | state norm growth | endpoint relative (nats/block) |
|---:|---:|---:|---:|---:|---:|
| 0 | **0.859** | **0.107** | **73.2x** | 1621x | −0.077 |
| 21 | 0.319 | 0.040 | 4.92x | 466x | −0.114 |
| 50 | 0.154 | 0.018 | 2.16x | 550x | −0.139 |
| 100 | 0.216 | 0.027 | 2.94x | 574x | −0.132 |
| 150 | 0.229 | 0.028 | 3.14x | 484x | −0.126 |
| 200 | 0.284 | 0.033 | 4.13x | 422x | −0.116 |
| 250 | 0.376 | 0.048 | 6.56x | 413x | −0.104 |
| 299 | 0.194 | 0.023 | 2.64x | 422x | −0.127 |

The absolute tangent volume grows **2.2-73x** over 40 blocks, while the per-token baseline state norm grows **413-1621x**. The final individual $\lambda_i$ are all positive and nearly flat across the 8 sampled directions: per token the final spectrum spans 0.005-0.031 nats, with the median 0.018-0.107. The local tangent gain has a median of **1.033** over all token-block cells, and 85.9% of cells (85% of block medians) are above 1: the absolute tangent is mildly expansive almost everywhere, not only at block 0. The last column is the endpoint rate $\frac{1}{40}\log(\text{tangent growth}/\text{state norm growth})$; every token is negative, −0.077 to −0.139. The chunk test below uses the same ratio over each saved interval and is the finer-grained version.

The state-norm column is **per token**; §4's 196x is the ratio of the *median over t* at state 0 to the median at state 39. The two aggregations differ because the per-token norm growth is heavy-tailed (the median per-token growth is 475x). The verdict does not depend on which is used: every per-token endpoint rate and 93/96 chunk cells are negative.

**The absolute hotspot is the first generated token, not a tie.** Token 0 has $h_{\rm depth}=0.859$, **2.3x** the next largest (token 250, 0.376) and **2.7x** the median of the other seven (0.244). Its final $\lambda_i$ are 0.104-0.113, the only values inside the 1.5B's 0.08-0.18 band; its first-block exponent is 0.674 against −0.008 to 0.269 for the others; its block-0 local gain is 2.05 against 1.00-1.31; and its tangent grows 73x against 2.2-6.6x. But its state norm also grows the most (1621x), so normalized it is the **least** contracting token (−0.077). The two thinnest tokens in the depth set are not hotspots: token 50 ($m_{50}=0.098$, 11th thinnest of 300 steps) has the **smallest** $h_{\rm depth}$ (0.154) and token 100 ($m=0.204$, 18th thinnest) has 0.216. Token 250 has the second-largest $h_{\rm depth}$ (0.376) but an ordinary margin ($m_{250}=10.73$, 238th of 300). Raw margin and absolute tangent volume are not aligned in this run.

### 7.2 The normalized verdict

The 12 chunk medians are:

| chunk | 0→1 | 1→2 | 2→4 | 4→7 | 7→11 | 11→15 | 15→19 | 19→23 | 23→27 | 27→31 | 31→35 | 35→38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| relative | −0.360 | −0.261 | −0.227 | −0.168 | −0.128 | −0.092 | −0.103 | −0.084 | −0.071 | −0.045 | **−0.033** | −0.117 |

**93 of 96 token-intervals contract** (96.9%), with an overall median of **−0.105 nats/block**. The three positive cells are token 0 in 0→1 (+0.153) and 31→35 (+0.0054), and token 200 in 23→27 (+0.0204). The first is a real first-block effect: the tangent grows 1.79x while the *median* baseline norm grows 1.54x across that one interval, a 1.17x relative expansion. The other two are at the resolution floor, exactly like the 14B's three exceptions.

The shape is **an early contraction that weakens toward zero and then re-accelerates at the very end**: −0.360 in the first block, a monotone approach to −0.033 in blocks 31-35 (the 1.5B's least-negative value is −0.034, so this is the saturation regime), and then −0.117 in the last three blocks. phi-4 therefore does **not** saturate flatly like Qwen3.5 over its last third; its final three blocks are the second-strongest late contraction after the first block. But the verdict is unambiguous: at 40 blocks, the absolute growth is radial/state-scale growth, and the state-normalized tangent contracts for every token; the only positive cells are the single first-block token-0 cell and two resolution-floor cells.

### 7.3 Comparison and the protocol caveat

| | 1.5B | DS-R1-14B | Qwen3.5-9B | **phi-4** |
|---|---:|---:|---:|---:|
| blocks | 28 | 48 | 32 | **40** |
| $h_{\rm depth}$ | 0.6-1.4 | 0.16-0.64 | — | **0.15-0.86** |
| final median $\lambda_i$ | 0.08-0.18 | 0.020-0.079 | — | **0.018-0.107** |
| normalized verdict | every block contracts, worst −0.034 | 61/64 chunks negative, −0.061 to −0.120 | contracts, saturates | **93/96 negative, median −0.105** |

The 14B's 61/64 = 95.3% is slightly weaker than phi-4's 93/96 = 96.9%, and both have three near-zero exceptions. The difference is in the range: phi-4's first-block chunk reaches −0.360, close to the 14B's first 6-block chunk (−0.408) but at 1-block instead of 6-block resolution; its late contraction is −0.033 to −0.117 rather than the 14B's −0.034 to −0.100. The 1.5B's "every block contracts" is not literally reproduced, but the three violations are +0.5%, +2% and +17% per block (the last is the single first-block token-0 cell); the other 93 cells contract.

**The tie-hotspot question cannot be answered by this run, and the text must say so.** The eight depth tokens are **0, 21, 50, 100, 150, 200, 250, 299**. phi-4's thin ties are at **219 and 277**, and **neither is in the list**. Token 21 is in the list but is not a phi-4 tie: $m_{21}=2.89$ nats, an ordinary margin (it is the 14B's razor tie, not phi-4's). Token 250 is the closest probe (27 steps before 277) and happens to have the second-largest $h_{\rm depth}$, but its margin is 10.73 nats, so it says nothing about the tie. The scheduled re-run with 219 and 277 in the token list is required before any claim that "the tie token is/is not a tangent hotspot" can be made.

---

## 8. Figure `phi4_generality.png` — the distributions generalise, the gates do not

![multi-prompt generality](figures/phi4_generality.png)

**Setup.** `a1_childhood_w0/a1.npz` and `a1_cyber_w0/a1.npz`: 200 directions at $r=0.1$, 300 teacher-forced tokens, 14 saved states, on the `childhood` ($P=56$) and `cyber` ($P=61$) prompts. Both use the same seed-69 direction generator as the interstellar A1 run, so their 200 directions are the first 200 of the 1000-direction run and are directly comparable direction-by-direction. The interstellar curve is the full $K=1000$ run.

**Axes.** (a) $x$ baseline margin $m_t$ (log); $y$ fraction of steps at or below it. (b) $x$ $r_*$ (log); $y$ fraction of directions with $r_*\le r$; dashed line is the run radius. (c) $r_*$ quantiles per prompt; thick bar = 25-75%, thin = 10-90%, circle = median. (d) percentage of directions that flip at $r=0.1$ (dark) and $r=0.3$ (light).

| prompt | $K$ | margin min (step) | margin p50 | $r_*$ p10 / p50 / p90 | flips at 0.1 / 0.3 | top controlling steps (directions) | median $m_{t_*}$ | median $s$ |
|---|---:|---:|---:|---|---:|---|---:|---:|
| interstellar | 1000 | 9.25e-3 (277) | 3.47 | 1.20 / **3.59** / 10.66 | 0 / 0.2% | 219 (273), 277 (253), 34 (95) | 0.0170 | 7.67e-3 |
| childhood | 200 | 8.89e-3 (52) | 3.67 | 0.79 / **2.98** / 11.18 | 0 / 1.0% | 32 (52), 52 (51), 183 (39) | 0.0116 | 5.89e-3 |
| cyber | 200 | 4.87e-3 (224) | 2.89 | 0.76 / **3.16** / 9.34 | 0 / 1.0% | 224 (60), 27 (38), 25 (36) | 0.0348 | 7.09e-3 |

- **The margin spectrum barely moves.** The median margin is 2.89-3.67 and the 10th percentile 0.35-0.58 across the three prompts. Every prompt has one sub-0.01 nat step (277, 52, 224), and the cyber minimum (4.87e-3) is actually **1.9x thinner** than interstellar's (9.25e-3). The ECDFs in panel (a) are almost on top of each other.
- **The $r_*$ distribution is equally stable.** Median 2.98-3.59, p10 0.76-1.20, p90 9.34-11.18. **All three prompts have 0 flips at $r=0.1$**; at $r=0.3$ the two 200-direction prompts have 1.0% (2/200 each) against 0.2% (2/1000) for interstellar. The model-level "hard to move" property generalises; the two new prompts are not more fragile at the campaign radius despite their thin minima.
- **The gates themselves move completely.** The controlling steps are prompt-specific: 219/277/34/186/53 for interstellar, 32/52/183/148 for childhood, 224/27/25/45 for cyber. The number of distinct controlling steps drops from 27 to 12 and 11, and the five thinnest steps carry 79.9%, 86.5% and 72.5% of the directions. Nothing about *which* step is the weak point transfers.
- **The vulnerable directions move with the prompt.** On the same 200 directions, the Spearman correlation of $\log r_*$ is **0.092** (p=0.20) for childhood vs interstellar and **−0.074** (p=0.30) for cyber vs interstellar; between the two new prompts it is 0.121 (p=0.09). The shared-flip counts at $r\le1$ are 3 of (15, 32) and 0 of (15, 27). Direction vulnerability is a direction $\times$ prompt property, not a fixed property of the random direction.
- **What stays is the low response.** The median local sensitivity at the controlling step is 5.9e-3 (childhood) and 7.1e-3 (cyber) against 7.7e-3 (interstellar), and the median controlling margin is 0.012-0.035. The ratio that sets $r_*$ is therefore stable within a factor of 1.3. The "uncertain where it is hard to move" reading is a property of phi-4, not of the interstellar prompt.

---

## 9. Still pending: non-normality and trajectories

**9.1 `phi4_nonnorm.png` — pending.** `nonnorm_w0/nonnorm.npz` is still generating; the fused-MLP Jacobian path is being fixed. Planned: MLP analytic SwiGLU Jacobian at 5 tokens x 9 blocks, full-block power iteration/Arnoldi at 5 tokens x 5 blocks, principal angles 4 blocks apart; $\sigma_{\max}$, $\rho$ and $\kappa=\sigma_{\max}/\rho$ by block, the $\sigma_{\max}$-$\rho$ plane, and the angle histogram against the 1.5B's 83.6° and the 14B's 86.96°. The reference pattern to test is the U-shape: extreme at block 0, normal in the middle, extreme again at the last block. No numbers are quoted and no figure is drawn until the file exists.

**9.2 Trajectories.** `traj/` is for the separate D2 analysis; the distance matrices will arrive later under `results_big/phi4/d2/`. Nothing in this section depends on them.

---

## 10. Cross-model comparison

All rows use the same prompt, greedy decoding, FP32 and $r=0.1$ unless stated. 1.5B and 14B values are from the published sections; Qwen3.5-9B from its section; phi-4 from this one.

| quantity | 1.5B | DS-R1-14B | Qwen3.5-9B | **phi-4** |
|---|---:|---:|---:|---:|
| blocks / width | 28 / 1536 | 48 / 5120 | 32 / 4096 | **40 / 5120** |
| depth $h_{\rm depth}$ | 0.6-1.4 | 0.16-0.64 | — | **0.15-0.86** |
| depth, normalized | every block contracts (worst −0.034) | 61/64 chunks negative | contracts, saturates | **93/96 negative** |
| single-position kick flips at $r\le1$ | — | 22/24† | — | **2/24** |
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

† the 14B's kick row is the maximum over its 10 positions ($s=8$ and $s=150$, 22/24 at $r\le1$) from `section_state_depth.md`; the 1.5B and Qwen3.5 have no state-kick run in this campaign.

Reading the table:

- **Margin thickness.** phi-4's median margin (3.47) is 12% thicker than the 14B's and 40% thicker than Qwen3.5's, but 27% thinner than the 1.5B's. Its minimum is 95x the 14B's yet only 1.3x Qwen3.5's. So by margins it is "mid-pack": it has neither the 14B knife edge nor the 1.5B's very thick minimum.
- **Threshold radius.** It has the **widest** flux tube: p50 31x the 14B, 5.1x Qwen3.5 and 2.3x the 1.5B. At the campaign radius this is the difference between 45% (14B), 1.2% (Qwen3.5), 0/20 (1.5B) and **0/1000**.
- **Gain.** It has the **lowest** response gain of the four: 1.5x below the 14B, 2.1x below Qwen3.5, 3.5x below the 1.5B. Its 20.3x spread is the widest, so direction choice matters more than in any other model.
- **Depth.** phi-4 has the widest absolute $h_{\rm depth}$ range (0.15-0.86, driven by token 0) but the most consistently negative normalized tangent (93/96 intervals). The absolute/radial split is the same story as in the other three models, not a phi-4 exception.
- **State kicks.** The 14B's kicks reach its early razor pair at 22/24 directions; phi-4's strongest kick reaches its late tie at 2/24, and only after 27 steps. The kick result is the dynamical counterpart of the static flux-tube result: one-position changes do not travel far enough to threaten phi-4's decisions.
- **Floor.** Its floor (1.51e-3) sits between Qwen3.5's (6.1e-4) and the 14B's (2.7e-3); signal-to-floor at $r=0.1$ is 24.9x (14B 17.5x, Qwen3.5 82x).
- **Angle.** The near-perpendicular response is reproduced (89.85°), but with an 8° IQR rather than Qwen3.5's 1.4°.

---

## 11. Where phi-4 sits in the fragility hierarchy, and why

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

## 12. Prediction scorecard

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

## 13. Caveats and what this does not show

1. **The layer profile is sparse.** Only 14 states are saved, covering blocks 0,1,2,4,7,11,15,19,23,27,31,35,38. The 131x growth and the "12/12 positive intervals" are statements about those 12 intervals, not about every block. A block between two saved states could contract without being seen. The last block (39) is absent.
2. **The state-normalized response uses the median baseline norm per saved state**, not per step. This is the same compromise as the 14B, but at finer resolution (12 intervals instead of 8).
3. **The log's calibration median is not the file's.** The file gives median 3.782 (p25 1.355), not ~1.2-1.5. If the run log aggregated a different statistic, that statistic is not in `calib.npz`.
4. **The angle is a 24-direction, one-state measurement.** It uses the final-norm state only. The IQR of 8° and the per-step range of 30° show that a single direction's angle can be noisy when its response is $10^{-4}$ of the state norm; the median is stable, the tails are not.
5. **The free-run scorecard has almost no power.** Zero events at 0.1, four at 0.3, and two of those are one-token substitutions. "99.5% accuracy" is 995 both-none plus five classifications; it is not a 99.5% success rate on events.
6. **The 2/4 re-sync result contradicts the "0 re-sync expected" prior.** It is reported as measured. With four events it cannot distinguish "phi-4 sometimes produces transient substitutions" from "these two directions happened to be near a one-token boundary".
7. **The decomposition uses each model's own A1 file.** The 14B point is computed from `results_big/ds14/a1_inter_w100/a1.npz` with the same signed rule; it gives a median $r_*$ of 0.103 where the published cross-scale value is 0.116 (a run-version difference), so the ordering, not the third digit, is the claim.
8. **phi-4 has no sliding window.** Window experiments are undefined (null by construction); nothing here says how a window would change the margins.
9. **Non-normality is still pending.** `nonnorm_w0/nonnorm.npz` is still generating (fused-MLP Jacobian fix); §9 quotes no numbers for it. The D2 trajectory distance matrices are also still to arrive under `results_big/phi4/d2/`. The conclusions about the fragility hierarchy rest on the calibration, A1, free, depth, state-kick and multi-prompt runs and do not depend on those two blocks.
10. **The 40-block count and $d=5120$ are metadata facts**, and the depth npz independently confirms 40 block entries; the npz also confirms $d=5120$ through `dh_last` and the vocabulary 100352 through the tokenizer.
11. **The depth run cannot answer the tie-hotspot question.** Its tokens are $0,21,50,100,150,200,250,299$; the phi-4 ties at 219 and 277 are absent. Token 21 is present but is an ordinary-margin token in phi-4 ($m_{21}=2.89$), and token 250 is 27 steps before 277 with a large margin ($m=10.73$). No proxy in this run is a tie. A re-run with 219/277 is scheduled.
12. **The depth spectrum is top-8, not full-rank.** $h_{\rm depth}$ sums the 8 sampled tangent directions; a 5120-dimensional Jacobian can have larger directions outside the sampled subspace. The final spectrum is flat across the 8 directions, which is consistent with, but not proof of, a flat full spectrum. The state normalization uses the A1 `norm_h0` at 14 saved states (12 intervals), not per-block norms.
13. **The state kicks are a single-position, embedding-level injection.** They are not the prompt-wide perturbation of the A1/free runs, and the readout is state 39. The flip count uses $r_c\le1$ recomputed from the baseline $m$ and the $r=0.1$ response; at the actual $r=0.1$ no kick flips anything. The half-life is measured from the immediate response at $s$, not from a global maximum. The re-sync argument is a consistency argument, not a mechanism proof.
14. **The multi-prompt runs have 200 directions and one prompt each.** The same seed-69 generator makes the 200 directions comparable to the first 200 interstellar directions, but the Spearman correlations have $n=200$ and $p=0.09$-0.30, so they establish "no strong shared ordering", not a precise zero. The 1% flip rate at $r=0.3$ is 2/200 per prompt and cannot be compared as a rate with the 2/1000 interstellar value.
