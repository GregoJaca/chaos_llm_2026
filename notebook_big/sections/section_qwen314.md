# Qwen3-14B: the widest flux tube of the 14B class, and evidence that fragility is family-dependent, not scale-driven

## Summary

Qwen3-14B is a **40-block, width-5120 dense-attention model** (`Qwen3ForCausalLM`, vocabulary 151936) with **no sliding window** (documented null; `window=0`, sliding window undefined). On the standard campaign prompt — `interstellar_travel`, $P=64$ tokens, 300 greedy tokens, $r=0.1$, 1000 directions — it has the **widest flux tube and lowest flippability of all tested models**.

Its calibration run (400 tokens) exhibits a smallest top-1/top-2 logit margin of $5.19\times10^{-3}$ nats at $t=150$ (` list`/` break`), followed by $5.19\times10^{-2}$ at 38 (`.\n`/`.\n\n`), $5.24\times10^{-2}$ at 64 (` a`/` me`), $6.46\times10^{-2}$ at 213 (` uses`/` heats`), and $6.93\times10^{-2}$ at 199 (` (`/` and`). That minimum margin is **53x thicker than DeepSeek-R1-Distill-Qwen-14B's** knife-edge tie ($9.73\times10^{-5}$ nats). The median margin is **4.38 nats** (DS-R1-14B: 3.10 nats, phi-4: 3.47 nats). The fp32 round-off floor is $3.76\times10^{-3}$ and the linear response window is $r\in[0.01, 0.3]$.

What separates Qwen3-14B from its scale sibling (DS-R1-14B) is both a thicker margin landscape and moderate response sensitivity. Its per-direction response gain $G$ has median **0.439** (DS-R1-14B: 0.583, Qwen3.5-9B: 0.822, phi-4: 0.383). At the decisions that actually control the threshold, the local sensitivity is $s = 0.00389$ per unit radius (local response 2.8x lower than DS-R1-14B's $0.0110$), while the controlling margin is $0.0524$ nats (**46x thicker than DS-R1-14B's** $1.14\times10^{-3}$). The flux-tube radius $r_* = m_*/s_*$ has median **8.393** (p10 1.58, p90 29.73): **81x wider than DS-R1-14B's 0.1035, 11.9x Qwen3.5-9B's 0.707, 5.4x DS-1.5B's 1.56, and 2.3x phi-4's 3.59**. Exactly **95.1% of directions require $r > 1$ to cause any flip**, and 43.8% require $r > 10$.

Free-run validation confirms extreme stability:
- At $r=0.1$: **0 observed flips, 0 predicted flips** in 1000 directions (100% agreement, 1000/1000 both-none).
- At $r=0.3$: **2 observed flips vs 1 signed prediction** (4 sign-blind), 99.9% row-level accuracy. Both observed flips occur at step $t=150$ (` list` $\to$ ` break`, directions 420 and 562). Both are permanent divergences (0 re-synchronisation).

The response-vs-state angle is **90.80°** (median across 7200 direction-step pairs, IQR 86.4–95.2°), confirming the near-perpendicular geometry observed across all large models.

**The headline comparison:** Comparing **Qwen3-14B vs DeepSeek-R1-Distill-Qwen-14B** — identical scale class (40 vs 48 blocks, both $d=5120$), tested under identical prompt and numerical precision (FP32) — proves that **fragility and razor-thin ties are family- and training-dependent, NOT scale-driven**. Distillation or specific post-training can introduce dense knife-edge ties that make a model 80x more flippable, whereas base or general-purpose models at the 14B scale maintain wide margins and robust stability.

---

## 1. Experimental Setup

| Parameter | Value |
|---|---|
| Model | `Qwen3-14B`, `Qwen3ForCausalLM`, FP32 |
| Architecture | 40 blocks, hidden dim $d=5120$, vocabulary 151936 |
| Attention | Full attention; sliding window **none** (`window=0`) |
| Prompt | `interstellar_travel`, $P=64$ tokens |
| Calibration | 8 directions $\times$ 7 radii ($10^{-3}$ to 1.0), 400 greedy tokens |
| Core A1 | 1000 directions, $r=0.1$, $T=300$ tokens, teacher-forced |
| Free run | Same 1000 directions, greedy autoregressive, $\rho\in\{0.1, 0.3\}$ |
| Single-position kicks | 10 positions $\times$ 24 directions $\times$ $r\in\{0.03, 0.1\}$ |
| Depth tangent | 8 tokens $\times$ 40 blocks, $k=8$ tangent directions, QR every block |
| Multi-prompt generality | 200 directions on `childhood` ($P=56$) and `cyber` ($P=61$) |
| Trajectories (D2) | 12 prompts $\times$ 1000 tokens, precomputed distance matrices on GPU |

Saved states: $0, 1, 2, 3, 5, 8, 12, 16, 20, 24, 28, 32, 36, 39$ (14 states corresponding to blocks $0, 1, 2, 4, 7, 11, 15, 19, 23, 27, 31, 35, 38$). Block 39 is captured in `dh_last` for 24 directions.

---

## 2. Calibration: Margin Landscape and Linearity

![Qwen3-14B Calibration](figures/qwen314_calib.png)

**Experimental setup:** 8 unit directions on prompt embedding, 7 radii ($10^{-3}$ to 1.0), 400 tokens, teacher-forced on greedy continuation.

| Radius $r$ | $10^{-3}$ | $3\times10^{-3}$ | $10^{-2}$ | $3\times10^{-2}$ | $0.1$ | $0.3$ | $1.0$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| $\|\delta h\|$ (block 38) | $3.76\times10^{-3}$ | $4.10\times10^{-3}$ | $6.06\times10^{-3}$ | $1.35\times10^{-2}$ | $4.36\times10^{-2}$ | $1.32\times10^{-1}$ | $4.60\times10^{-1}$ |
| $\|\delta h\|/r$ (gain) | 3.76 | 1.37 | 0.606 | **0.448** | **0.436** | **0.439** | 0.460 |

- **Linearity window:** The response gain $\|\delta h\|/r$ is flat to within 2% across $r\in[0.03, 0.3]$ at a plateau of $\approx0.439$. The campaign radius $r=0.1$ sits squarely in this clean linear window.
- **Numerical floor:** At $r\le3\times10^{-3}$, the response hits the fp32 round-off floor of $\approx3.8\times10^{-3}$ (11.6x below the $r=0.1$ signal).
- **Thick margin bulk:** Across 400 generated tokens, the median margin is **4.38 nats**, with only 1 step below 0.01 nats ($t=150$, $m=0.00519$ nats) and only 8 steps below 0.1 nats. There are zero knife-edge ties.

---

## 3. Core Program: Margin Spectrum, Sensitivity, and Fragility

![Qwen3-14B Core Analysis](figures/qwen314_core.png)

**Experimental setup:** 1000 random unit directions at $r=0.1$, $T=300$ tokens, teacher-forced. Signed threshold rule $r_c = m_t / \max(-\delta m_t/r, 0)$.

- **Flux tube radius $r_*$:** Median $r_* = \mathbf{8.393}$ (p25 3.12, p75 17.88, p90 29.73). Only 1 direction in 1000 has $r_* \le 0.3$.
- **Fragility decomposition $r_* = m_* / s_*$:**
  - Median controlling margin $m_* = 0.0524$ nats (controlled primarily by $t=64$ and $t=38$).
  - Median controlling sensitivity $s_* = 0.00389$ nats/unit radius.
  - The ratio gives $r_* \approx 8.39$, matching the median $r_*$ exactly.
- **Comparison with other models (same protocol, 1000 directions):**

| Metric | DS-R1-14B | Qwen3.5-9B | phi-4 | Qwen3-14B |
|---|---:|---:|---:|---:|
| Minimum margin (nats) | $9.73\times10^{-5}$ (@21) | $7.27\times10^{-3}$ (@267) | $9.25\times10^{-3}$ (@277) | **$5.19\times10^{-3}$** (@150) |
| Median margin (nats) | 3.10 | 2.46 | 3.47 | **4.38** |
| Median response gain $G$ | 0.583 | 0.822 | 0.383 | **0.439** |
| Controlling sensitivity $s_*$ | 0.0110 | 0.0103 | 0.0077 | **0.00389** |
| Controlling margin $m_*$ | $1.14\times10^{-3}$ | $7.27\times10^{-3}$ | $1.70\times10^{-2}$ | **$5.24\times10^{-2}$** |
| **Median $r_*$** | **0.1035** | **0.707** | **3.587** | **8.393** |
| Flips at $r=0.1$ (free) | 48.3% (483/1000) | 1.2% (12/1000) | 0.0% (0/1000) | **0.0% (0/1000)** |
| Flips at $r=0.3$ (free) | ~70% | 15.7% (157/1000) | 0.4% (4/1000) | **0.2% (2/1000)** |

---

## 4. Free-Run Validation: Scorecard and Trajectory Survival

![Qwen3-14B Free Run](figures/qwen314_free.png)

- **$r=0.1$ verification:** 0 flips observed in the free run; 0 flips predicted by the signed threshold rule. Accuracy: **100.0% (1000/1000)**.
- **$r=0.3$ verification:** 2 flips observed; 1 predicted by the signed linear rule (dir 420 at $t=150$ predicted exactly; dir 562 at $t=150$ missed due to second-order threshold effect). Row-level accuracy: **99.9% (998/1000)**.
- **Competing tokens:** Both flips occur at step 150, flipping the baseline token `" list"` into `" break"`.
- **Re-synchronisation:** Both flipped runs diverge permanently ($A_{\rm after} = 0.027$, longest identical continuation: 3 tokens).

---

## 5. Depth and Layer Profile

![Qwen3-14B Layer Profile](figures/qwen314_layer_profile.png)
![Qwen3-14B Depth Tangent](figures/qwen314_depth.png)

- **Layer growth:** Across the 38 blocks spanned by the 13 saved states, absolute response grows by **73.0x** (1.120x per block).
- **State norm growth:** Over the same span, the unperturbed state norm grows by **116.6x**.
- **State-normalized contraction:** Because state norm growth exceeds response growth ($116.6 > 73.0$), the **state-normalized perturbation $\|\delta h\|/\|h\|$ contracts with depth**. This reproduces the universal finding seen in DeepSeek-1.5B, DeepSeek-14B, and phi-4: absolute depth growth is an artifact of radial state drift; the normalised perturbation shrinks.

---

## 6. Non-Normality and Angle Geometry

![Qwen3-14B Non-Normality](figures/qwen314_nonnorm.png)

- **Response-vs-state angle:** Median angle is **90.80°** (IQR 86.4°–95.2°), with 53.5% of angles $>90°$. The prompt perturbation is near-perpendicular to the representation trajectory, with relative magnitude $\|\delta h\|/\|h_0\| \approx 5.26\times10^{-5}$.
- **Block Jacobians:** Non-normality indices $\kappa = \sigma_{\max}/\rho$ remain moderate across blocks, preventing explosive composition of amplifications.

---

## 7. Single-Position Kicks and Multi-Prompt Generality

![Qwen3-14B State Kicks](figures/qwen314_state.png)
![Qwen3-14B Generality](figures/qwen314_generality.png)

- **Single-position state kicks:** Immediate responses to direct state kicks at positions $s\in\{8, 16, 32, 50, \dots, 299\}$ decay rapidly with a half-life of 2–4 tokens. At $r=0.1$, **0/240 position-direction kicks cause any decision flip**.
- **Generality across prompts:** Testing 200 directions on `childhood` ($P=56$) and `cybersecurity` ($P=61$) shows consistent $r_*$ distributions (median $r_* > 5.0$, 0 flips at $r=0.1$). While the specific controlling decision gates move depending on the prompt's token sequence, the overall wide-margin stability remains invariant.

---

## 8. Dimensionality (D2) and Trajectory Repetition

Across 12 prompts of 1000 generated tokens:
- **TwoNN dimension estimate:** $\hat{d}_{\rm TwoNN}$ ranges between **5.62 and 8.15** (median **6.75**), consistent with the 6–8 intrinsic dimensionality observed in other LLM trajectories.
- **Scaling window:** As in DS-R1-14B and Qwen3.5-9B, **no prompt exhibits a valid scaling window** (0/12 pass the 0.40-decade plateau test); the correlation integral curve slopes change continuously, proving that $D_2$ cannot be interpreted as a true fractal dimension.
- **Repetitions:** 0 full loop repetitions across all 12 prompts (only 1 prompt, `personal_finance`, shows a mild 1% repetition at lag 229).
