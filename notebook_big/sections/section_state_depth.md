# Part state kicks, depth spectrum and non-normality (DS-R1-14B, true window 100)

## Summary

A single random unit kick in the residual stream at one generated position is absorbed quickly — the response at the last saved state falls to half its value at the injection point in **1–5 steps at seven of ten positions**, and in 24 steps at $s=32$ — but it does not disappear: for kicks at $s\ge32$ the largest response occurs **41–259 steps later, at $t\approx291$**, where it is **2.4–98 times** the immediate response. The late maximum is a property of the baseline trajectory, not of the kick position (it is at the same $t$ for every kick from $s=16$ onward), and it is not a state-norm artefact (the state norm at $t=291$ is $0.82$ of the median). Every kick except the one at the last position $s=299$ can flip some token at radius $\le1$: **42–92% of the 24 directions**, depending on $s$. Kicks at $s=8$ and $16$ flip the **razor ties at $t=19/21$** ($m_{21}=9.7\times10^{-5}$, $m_{19}=1.1\times10^{-3}$): 11 and 3 steps later at $t=19$ for most directions (14/24 and 9/24), or 13 and 5 steps later at $t=21$ for the rest. Kicks after $t=21$ can only act through the late ties at $t=271$/$276$ or the razor tie at $t=297$ ($m_{297}=4.8\times10^{-3}$).

The depth spectrum confirms the 1.5B verdict, and more strongly. The **absolute** tangent volume grows only $2.7$–$45.9\times$ over 48 blocks ($h_{\rm depth}=0.16$–$0.64$ nats), while the baseline state norm grows $655$–$833\times$; dividing by state growth, the tangent **contracts in 61 of 64 token-chunks**, with an overall rate of $-0.061$ to $-0.120$ nats/block. The 1.5B's "absolute expansion is radial drift; normalized every block contracts" is therefore reproduced. The three exceptions are $+0.0003$, $+0.0052$ and $+0.0083$ nats/block — at the resolution of the 6-block normalization, not a qualitative failure. The token at the razor tie, **$t=21$, is the single most expansive token in the absolute spectrum**: $h_{\rm depth}=0.638$ (1.55× the next largest, 2.5× the median of the other seven), final $\lambda_i\approx0.08$ (the only token reaching the 1.5B range $0.08$–$0.18$), first-block exponent $2.40$ (vs $0.98$–$1.42$) and total tangent growth $45.9\times$ (vs $2.7$–$11.8\times$). Normalized by state growth it is not exceptional — it contracts least ($-0.061$ vs $-0.090$ to $-0.120$) because its state norm also grows most, and two of the three slightly positive chunks are at $t=21$.

The 14B Jacobians have the same U-shaped non-normality as the 1.5B — extreme at block 0, normal in the middle, extreme at the last block — but the last block is more extreme. Full-block $\sigma_{\max}$ is $96.2$ at block 0 (range $66.2$–$128.5$), $3.7$–$4.9$ at blocks 12–36 and $14.2$ at block 47; $\kappa=\sigma_{\max}/\rho$ is $7.35$ at block 0, $1.60$–$2.27$ in the middle and $7.07$ at block 47. The 1.5B reference is $\sigma_{\max}=95$–$195$, $\rho=11$–$39$, $\kappa=5$–$8$ at block 0; $\sigma_{\max}=2.7$–$3.6$, $\kappa=1.7$–$2.5$ in the middle; $\sigma_{\max}=6$–$9$, $\kappa=3$–$5.3$ at the last block. The middle matches; the 14B's last block has **1.6–2.4× larger $\sigma_{\max}$ and $\kappa$ ≈ 7 instead of 3–5.3**. The MLP alone at block 47 has $\sigma_{\max}=118$ (94–147), against the 1.5B's 58–93. Principal angles are **86.96° overall** (per-token medians 84.1–87.8°), slightly more orthogonal than the 1.5B's 83.6°: consecutive blocks' top singular directions are nearly unrelated in both models.

---

## 1. What was measured

### 1.1 Setup

| item | value |
|---|---|
| model | `DeepSeek-R1-Distill-Qwen-14B`, 48 blocks, $d=5120$ |
| precision | FP32, no autocast |
| prompt | `interstellar_travel`, $P=65$ tokens |
| baseline | 300 greedy tokens from the **true window-100** A1 run, teacher-forced in every perturbed pass |
| state kicks | one random unit vector in the 5120-d residual stream at the **embedding of generated position $s$**, 24 directions per cell, radii $r=0.03$ and $0.1$; $s=8,16,32,50,75,100,150,200,250,299$ |
| saved states for kicks | 0, 12, 24, 36, 47; the response is read at layer 47 (input to the final norm) |
| depth | 8 orthonormal tangent vectors at one generated token, JVP through each of the 48 blocks, QR-renormalised after every block (Benettin); tokens $0,21,50,100,150,200,250,299$ |
| non-normality | MLP analytic SwiGLU Jacobian at 5 tokens × 9 blocks ($0,6,\dots,42,47$); full-block power iteration + Arnoldi at 5 tokens × 5 blocks ($0,12,24,36,47$); principal angles between blocks 4 apart |
| non-normality tokens | $21,50,100,189,273$ |
| baseline state norms | `a1_inter_w100/a1.npz`, `norm_h0`, states $0,6,12,\dots,42,47$ (9 values per generated step) |
| data | `results_big/ds14/state_w100/state2.npz`, `depth_w100/depth.npz`, `nonnorm_w100/nonnorm.npz`, `a1_inter_w100/a1.npz` |

The state margins are bit-identical to the A1 margins (checked with `array_equal`), so the kick run is on exactly the same baseline trajectory as the depth and non-normality runs. The true window is 100: the A1 metadata says `window=100`, and the margins used here are the stored ones from that run.

### 1.2 Quantities

| symbol | explicit expression | meaning | why this quantity |
|---|---|---|---|
| $\delta h^l_t$ | $h^l_t(u)-h^l_t(0)$ | displacement of the state at depth $l$, step $t$ | the object a state kick changes |
| immediate | $\lVert\delta h^{47}_s\rVert/r$, median over 24 directions | response at the injection token and the last saved state, per unit radius | measures how strongly position $s$ itself reacts |
| half-life | per direction, first $t>s$ with $\lVert\delta h^{47}_t\rVert<\tfrac12\lVert\delta h^{47}_s\rVert$ | initial decay time of the response | separates "absorbed at once" from "persists" |
| late ratio | $\max_{t\ge s}\lVert\delta h^{47}_t\rVert\,/\,\lVert\delta h^{47}_s\rVert$ | largest later response relative to the immediate one | tests whether the response really decays |
| $r_c^-(t,u)$ | $m_t\,r/\max(-\delta m_t(u),0)$; $+\infty$ if $\delta m_t\ge0$ | radius at which step $t$ would flip | one number per direction and later step |
| flip count | number of directions with $\min_{t\ge s}r_c^-(t,u)\le1$ | directions for which some radius $\le1$ flips a token | the latency question in one count |
| $\lambda_i(l)$ | $(1/(l+1))\sum_{b\le l}\log|R_{ii}(b)|$ after QR each block | running mean per-block tangent exponent | the depth spectrum; **not** the local gain and **not** a cumulative sum |
| $e_l$ | $(l+1)\lambda(l)-l\lambda(l-1)$ | per-block tangent exponent | the actual block-by-block growth |
| $h_{\rm depth}$ | $\sum_i\max(\lambda_i(47),0)$ | total positive tangent exponent | a scalar "how expansive" per token |
| $n_l$ | $\lVert h^l_t(0)\rVert$ at saved states | baseline state norm | the scale the tangent is compared with |
| relative exponent | $e_l-\frac1{6}\log(n_{l+6}/n_l)$ on 6-block chunks | tangent growth per block minus state-norm growth per block | the 1.5B "normalized" test |
| $\sigma_{\max}$ | top singular value of the block Jacobian | largest instantaneous gain | how much a direction can be stretched in one block |
| $\rho$ | largest $|\text{eigenvalue}|$ of the block Jacobian | long-term spectral radius | distinguishes transient from sustained growth |
| $\kappa$ | $\sigma_{\max}/\rho$ | non-normality ($1$ = normal) | high $\kappa$ means large transient amplification |
| angle | $\arccos|\langle v_l,v_{l+4}\rangle|$, degrees | angle between top singular directions 4 blocks apart | whether the amplifying direction rotates with depth |

### 1.3 The one normalization that could not be done per block

`norm_h0` stores the baseline norm at **nine states only**: $0,6,12,18,24,30,36,42,47$. The relative exponent is therefore computed on **eight 6-block chunks** (the last one is 5 blocks, $42$–$46$, because state 48 is not saved). Per-block state normalization would need all 48 norms; it is not available, and no per-block statement below should be read as measured. The last transformer block (47) is included in the tangent growth but not in the norm growth, because its output state 48 was not saved; this makes the reported contraction slightly **conservative** (the missing final norm growth would make it more negative, not less). The 1.5B's per-block normalization cannot be reproduced at the same resolution; the comparison is made on the 6-block medians.

---

## 2. `ds14sd_state_kicks.png` — a kick is absorbed quickly, then re-amplifies late

![state kicks](figures/ds14sd_state_kicks.png)

**(a) Immediate response against position.** $x$: generated position $s$ at which the kick is injected. $y$: $\lVert\delta h^{47}_s\rVert/r$, the response at the same token and the last saved state, per unit radius, on a log scale; filled circles are $r=0.1$, open squares $r=0.03$, error bars are the 25–75% range over the 24 directions. The response varies by a factor **4.5** across positions: $12.07$ at $s=50$, $10.15$ at $s=8$, $10.06$ at $s=100$, down to $2.67$ at $s=32$ and $2.94$ at $s=250$. The two radii agree per unit radius to within 6% at seven positions, by 7% and 10% at $s=50$ and $75$, and by 19% at $s=200$; the immediate response is close to linear at these radii. The variation across $s$ is a property of the token at $s$, not of the radius.

**(b) Initial half-life and late re-amplification.** $x$: kick position $s$. Left $y$: median over directions of the number of steps until the response first falls below half its value at $s$; bars, hatched when not every direction crosses half. Right $y$ (red triangles, log): largest later response divided by the immediate response. Most positions decay in **1–5 steps**; $s=32$ takes **24 steps**; at $s=250$ only **4 of 24 directions** ever fall below half, and at $s=299$ there is no later step. The red triangles are the surprise: at $s=32,75,150,200,250$ the late maximum is **17×, 17×, 31×, 17×, 98×** the immediate response, while at $s=8,16,50,100$ it is only 2.5×, 1.8×, 2.4×, 5.0×. The late maximum is at $t=291$ for every kick except $s=8$ (where it is at $t=114$) and $s=299$ (only $t=299$). It is therefore a property of the trajectory around $t=291$, not of the kick position.

**(c) Which later tie is crossed.** $x$: kick position $s$. $y$: fraction of the 24 directions. Bars are stacked by the **earliest** later step at which $r_c^-(t)\le1$: red = $t=19$ or $21$ (the razor ties), orange = $t=145$–$276$ (late burst / late near-ties), purple = $t=297$ (razor tie), grey = no flip at any radius $\le1$. Black dots are the total. The pattern is sharp:

| $s$ | immediate $/r$ ($r=0.1$) | half-life | directions decaying | flips / 24 | earliest flips (19/21, 145–276, 297, none) |
|---:|---:|---:|---:|---:|---|
| 8 | 10.15 | 3 | 24/24 | 22 | 20, 1, 1, 2 |
| 16 | 8.43 | 2 | 24/24 | 21 | 17, 1, 3, 3 |
| 32 | 2.67 | 24 | 24/24 | 15 | 0, 15, 0, 9 |
| 50 | 12.07 | 4 | 24/24 | 10 | 0, 9, 1, 14 |
| 75 | 4.49 | 3 | 24/24 | 19 | 0, 17, 2, 5 |
| 100 | 10.06 | 1 | 24/24 | 16 | 0, 14, 2, 8 |
| 150 | 5.86 | 5 | 24/24 | 22 | 0, 22, 0, 2 |
| 200 | 4.94 | 4.5 | 24/24 | 20 | 0, 19, 1, 4 |
| 250 | 2.94 | 6 | 4/24 | 22 | 0, 22, 0, 2 |
| 299 | 3.24 | — | 0/24 | 0 | 0, 0, 0, 24 |

The counts at $r=0.03$ agree with these within one direction at every position (the largest differences are $s=32$: 16 vs 15, and $s=200$: 19 vs 20). The median smallest threshold over directions ranges from $0.023$ at $s=250$ to $1.087$ at $s=50$; the single smallest is $0.0041$ at $s=250$. A kick at $s=8$ or $16$ reaches the $t=19/21$ razor pair **before** the baseline would decide it, and 20/24 or 17/24 directions flip there — **mostly at $t=19$** (14/24 at $s=8$, 9/24 at $s=16$), the rest at $t=21$. The first crossing is at 19 even though 21 has the smaller margin, because the response at 19 is larger; this is the same first-crossing-versus-argmin distinction as in the free-run part. A kick at $s\ge32$ cannot reach $t=21$; it must act through the late near-ties at $t=271$ ($m=0.052$), $t=276$ ($m=0.165$) or $t=297$ ($m=0.0048$), or through the late burst at $t=262$ (where the baseline margin is $1.465$ and the flip requires a margin change of order one nat). This is why the total flip fraction is not monotone in $s$.

**(d) Response trajectories.** $x$: generated step $t$. $y$: $\lVert\delta h^{47}_t\rVert$, median over directions, log scale; four kick positions. The dotted vertical line marks each injection, the dashed line $t=291$. The earlier kicks ($s=8,50,150$) drop by an order of magnitude in the first 5–20 steps and then sit at $0.1$–$1$; the late kick $s=250$ instead grows immediately ($0.29$ at $s$, $7.7$ at $t=260$). All curves then show the common burst at $t\approx291$, where the late kicks jump to $10$–$30$. The state norm at $t=291$ is $905.9$, below the median $1100.7$ of the 300 steps, so the burst is not the state growing; it is a tangent amplification of the difference.

**What the figure does not show.** The half-life is measured from the immediate response at $s$; it is not a global half-life. For $s\ge32$ the global maximum is late, so a half-life measured from the global maximum would not exist inside the 300-step window. The flip counts use the linear threshold $r_c^-$ from `dm` at $r=0.1$; the measured `dm` at $r=0.1$ is $2.74\times$ the value at $r=0.03$, not the $3.33\times$ of a linear response, so the extrapolation is already 18% sublinear at the upper radius. The counts at $r=0.03$ and $0.1$ agree within one direction, which bounds this effect on the count but not on the exact threshold. The late burst is measured only as the norm of the layer-47 difference; the figure does not decompose it into a direction or a token-level event.

---

## 3. `ds14sd_depth_spectrum.png` — absolute growth is one block plus a late tail; normalized, it contracts

![depth spectrum](figures/ds14sd_depth_spectrum.png)

**(a), (b) The depth spectrum.** $x$: block index $l$, 0 = first block. $y$: $\lambda_i(l)=(1/(l+1))\sum_{b\le l}\log|R_{ii}(b)|$, the running mean per-block tangent exponent, log scale; one line per token (median over the 8 tangent directions), shaded band = 25–75% over the 8 directions. Panel (a) has tokens $0,21,50,100$; panel (b) $150,200,250,299$. Every token has the same shape: a huge first block, then a rapid drop, then a slow decay to a final running mean of $0.02$–$0.09$ nats/block. Token 21 is visibly the highest curve throughout; token 0 is the lowest over most of the depth.

**(c) Total tangent growth per token.** $x$: probe token. $y$: $h_{\rm depth}=\sum_i\max(\lambda_i(47),0)$, the sum of the eight final positive exponents, in nats. Bars: token 21 red. The grey band is the 1.5B reference range $0.6$–$1.4$. Token 21 reaches the bottom of the 1.5B band at $0.638$; the other seven are $0.163$–$0.412$.

**(d) Tangent gain against state-norm growth.** $x$: block index. $y$: growth factor per block, log scale. Blue: median over tokens of `local_gain`, the mean $\lVert Jq_i\rVert$ over the 8 tangent directions, with the 25–75% band over tokens. Red: state-norm growth $\exp(\frac16\log(n_{l+6}/n_l))$ from the A1 `norm_h0` at the 6-block chunk centres. The tangent gain is **3.37** at block 0 (range 3.2–11.1), then hovers at **1.00** (0.93–1.07) through the middle, and rises to **1.04–1.10** in the last 8 blocks; 58% of blocks have a median gain above 1. The state-norm line is $1.77$–$1.91$ per block in the first chunk and $1.02$–$1.15$ afterward. Over the full depth the tangent volume grows $2.7$–$45.9\times$ while the state norm grows $655$–$833\times$; the tangent therefore grows **16–280× less** than the state.

**(e) Absolute and relative exponents.** $x$: block index (the relative points are at 6-block chunk centres). $y$: log gain per block, nats/block. Grey: absolute per-block exponent $e_l$, median over tokens with the 25–75% band; block 0 is $+1.22$ (range $+0.98$ to $+2.40$) and is off the plotted range. Blue: relative exponent, tangent minus state-norm growth, median over tokens with the 25–75% range over tokens. The absolute exponent is near zero or slightly negative through the middle and becomes $+0.03$ to $+0.09$ in the last blocks. The relative exponent is negative at every chunk centre: the median over tokens ranges from $-0.408$ at blocks 0–5 to $-0.034$ at blocks 18–24; every other chunk lies between $-0.100$ and $-0.048$. The red dotted line is the 1.5B's least-negative relative exponent, $-0.034$; the 14B's chunk medians are at or below it except for three tiny positive cells.

**(f) Final spectrum per token.** $x$: tangent direction $i=1,\dots,8$. $y$: final $\lambda_i$, nats/block. Each grey line is one token; token 21 is red. The spectra are flat across directions (the 8 values span at most $0.017$–$0.086$), which is why the scalar $h_{\rm depth}$ is meaningful. The grey band is the 1.5B range $0.08$–$0.18$: only token 21 reaches it.

### 3.1 The numbers per token

| token $t$ | $h_{\rm depth}$ | median final $\lambda_i$ | tangent volume growth | state norm growth | relative mean (nats/block) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.258 | 0.033 | 4.71× | 655× | −0.107 |
| 21 | **0.638** | **0.079** | **45.9×** | 733× | −0.061 |
| 50 | 0.412 | 0.048 | 11.83× | 723× | −0.090 |
| 100 | 0.395 | 0.048 | 10.67× | 760× | −0.093 |
| 150 | 0.271 | 0.032 | 5.10× | 833× | −0.110 |
| 200 | 0.216 | 0.025 | 3.66× | 689× | −0.112 |
| 250 | 0.163 | 0.020 | 2.66× | 742× | −0.120 |
| 299 | 0.199 | 0.023 | 3.29× | 691× | −0.114 |

The final individual $\lambda_i$ run from $0.0171$ to $0.0859$, all positive; the median per token is $0.020$–$0.079$. The block-0 exponent is $0.98$–$2.40$; the late-block exponent (blocks 44–47) is $0.03$–$0.15$. The middle blocks are not uniformly expansive: their per-block exponents scatter around zero, and the median `local_gain` is below 1 in 42% of blocks. The absolute expansion is therefore **not "every block expands"** in the 14B; it is "block 0 plus a late tail", with a middle that is nearly neutral.

### 3.2 The relative test, chunk by chunk

The eight chunks and the median over the 8 tokens:

| chunk | blocks | tangent exponent | norm exponent | relative |
|---|---|---:|---:|---:|
| 0→6 | 0–5 | +0.201 | +0.611 | **−0.408** |
| 6→12 | 6–11 | −0.015 | +0.078 | −0.100 |
| 12→18 | 12–17 | −0.040 | +0.024 | −0.064 |
| 18→24 | 18–23 | −0.014 | +0.019 | **−0.034** |
| 24→30 | 24–29 | +0.010 | +0.066 | −0.048 |
| 30→36 | 30–35 | +0.038 | +0.100 | −0.070 |
| 36→42 | 36–41 | +0.049 | +0.094 | −0.051 |
| 42→47 | 42–46 | +0.052 | +0.140 | −0.080 |

(Tangent and norm are medians over tokens; the tangent exponent is the average over the 8 directions.) The first chunk is the strongest contraction: the tangent grows $e^{0.201}=1.22\times$ per block while the state norm grows $e^{0.611}=1.84\times$ per block, a relative $-0.408$ nats/block. After that the tangent is nearly flat ($-0.040$ to $+0.052$) while the norm keeps growing $e^{0.019}$–$e^{0.140}$ per block, so the relative exponent stays negative.

**61 of the 64 token-chunks contract.** The three positive cells are:

| token | chunk | relative |
|---:|---|---:|
| 21 | 30–36 | +0.0052 |
| 21 | 36–42 | +0.0003 |
| 50 | 36–42 | +0.0083 |

All three are at most $+0.008$ nats/block, i.e. a relative expansion of at most 0.8% per block, at the resolution of a 6-block average. The overall relative rate (mean over all chunks) is $-0.061$ at token 21 and $-0.090$ to $-0.120$ at the other tokens.

### 3.3 Comparison with the 1.5B depth result

| quantity | 14B (this run) | 1.5B reference | verdict |
|---|---|---|---|
| final $\lambda_i$ | 0.017–0.086 nats/block | 0.08–0.18 | 14B is **less** expansive; only token 21 reaches the 1.5B range |
| $h_{\rm depth}$ | 0.16–0.64 nats | 0.6–1.4 | 14B is lower; token 21 (0.638) is at the bottom of the 1.5B band |
| absolute tangent growth | 2.7–45.9× over 48 blocks | positive exponents, $e^{0.08}$–$e^{0.18}$ per block | 14B's absolute growth is much weaker |
| state norm growth | 655–833× | ×30 in block 0, ×1.05–1.18 afterward | same qualitative pattern; the 14B does not isolate block 0 because norms are only saved every 6 states |
| normalized relative exponent | −0.061 to −0.120 overall; 61/64 chunks negative; least-negative chunk median −0.034 | every block contracts, worst −0.034 | **confirmed**, with three near-zero exceptions at 6-block resolution |
| what the growth is | tangent volume grows 2.7–45.9× while the norm grows 655–833× | "absolute expansion is radial drift; normalized contracts" | **confirmed, more strongly in the 14B** |

The one place where the 14B does not literally reproduce the 1.5B sentence is the word "every": at 6-block resolution three token-chunks are positive. The violations are $+0.0003$ to $+0.0083$ nats/block, far below the $\pm0.03$–$0.1$ spread between tokens in the same chunk, and they are exactly the chunks where the 14B's tangent is flat and the state norm growth is modest. A per-block test could in principle resolve them, but `norm_h0` does not contain the 48 per-block norms needed for it. The honest statement is: **the 14B confirms normalized contraction at 6-block resolution, and the three exceptions are at the resolution floor, not a qualitative contradiction.**

### 3.4 Is the token at the razor tie special?

Yes, in the absolute spectrum. Token 21's $h_{\rm depth}$ is $0.638$, **1.55×** the next largest (token 50, $0.412$) and **2.5×** the median of the other seven ($0.258$). Its first-block exponent is $2.40$ against $0.98$–$1.42$ for the other tokens; its first-block local gain is $11.13$ against $2.72$–$4.26$; its total tangent growth is $45.9\times$ against $2.7$–$11.8\times$; its final $\lambda_i$ are $0.076$–$0.086$, at the bottom of the 1.5B range; no other token exceeds $0.066$. The depth spectrum therefore agrees with the non-normality run, where token 21 also has the largest block-0 full-block $\sigma_{\max}$ ($128.5$).

It is not special in the **relative** spectrum. Its state norm grows by $733\times$, almost the same as the other tokens ($655$–$833\times$), and its relative rate is the **least negative** ($-0.061$ vs $-0.090$ to $-0.120$). The large absolute tangent growth is accompanied by a large state norm, so normalized it is the mildest contraction, not an expansion. Two of the three positive chunks are at token 21. The correct reading is: **the razor-tie token is the dynamical hotspot of the absolute tangent (largest gain, largest final spectrum), but it still contracts when divided by its own state growth.** This is exactly the 1.5B lesson — a large absolute number at a sensitive token is mostly the state scale.

---

## 4. `ds14sd_nonnorm.png` — the same U-shape as the 1.5B, with a more extreme last block

![non-normality](figures/ds14sd_nonnorm.png)

**(a) MLP block gain.** $x$: block index. $y$: Jacobian gain, log scale. Blue: $\sigma_{\max}$ of the MLP block, median over the 5 tokens with min–max range; orange: spectral radius $\rho$. Nine measured blocks ($0,6,\dots,42,47$). The MLP gain is $9.45$ at block 0, drops to $4.82$ at block 6, rises slowly from $9.57$ at block 12 to $19.36$ at block 42, then jumps to **$117.65$ (94.36–146.54)** at block 47. The spectral radius follows the same shape, $4.02\to1.41\to4.96\to\dots\to22.94$.

**(b) Non-normality $\kappa$.** $x$: block index. $y$: $\kappa=\sigma_{\max}/\rho$, linear scale. Blue: MLP block (9 blocks, median + min–max); red dashed: full block (5 blocks, $0,12,24,36,47$). The grey band is the 1.5B mid-layer range $1.7$–$2.5$; the purple band the 1.5B last-block range $3$–$5.3$. The 14B is U-shaped: $\kappa=7.35$ (full, block 0), $1.60$ (block 12), $2.14$ (24), $2.27$ (36), $7.07$ (47). The MLP alone has a milder block 0 ($\kappa=2.34$) but a large last block ($5.23$).

**(c) $\sigma_{\max}$ against $\rho$.** $x$: spectral radius $\rho$, log; $y$: $\sigma_{\max}$, log; one point per token per full block, colour = block. The grey lines are $\kappa=1$ (normal), $3$ and $10$. Block 0 sits at $\rho\approx8.5$–$17.5$, $\sigma_{\max}\approx66$–$128$ ($\kappa=5.2$–$11.5$); block 47 sits at $\rho\approx1.5$–$2.1$, $\sigma_{\max}\approx10.5$–$14.7$ ($\kappa=6.9$–$9.8$). The last block has a **small spectral radius but a large top singular value**: its gain is a transient, not sustained growth. Blocks 12–36 are the normal middle ($\kappa=1.5$–$2.8$).

**(d) Principal angles.** $x$: angle between the top singular directions of blocks 4 apart, degrees; $y$: number of block pairs (5 tokens × 12 pairs). Blue bars: histogram; vertical lines: 14B median $86.96°$ and 1.5B median $83.6°$. Per-token medians are $84.14°$ (21), $87.76°$ (50), $87.66°$ (100), $87.01°$ (189), $85.42°$ (273). The full range is $69.8$–$89.8°$. Consecutive-block amplifying directions are nearly orthogonal in both models; the 14B is 3.4° more orthogonal.

### 4.1 The profile in numbers

| block | MLP $\sigma_{\max}$ (range) | MLP $\rho$ | MLP $\kappa$ | full $\sigma_{\max}$ (range) | full $\rho$ (range) | full $\kappa$ (range) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.45 (8.87–12.08) | 4.02 | 2.34 | 96.22 (66.2–128.5) | 12.67 (8.5–17.5) | 7.35 (5.23–11.50) |
| 6 | 4.82 (4.53–11.70) | 1.41 | 3.53 | — | — | — |
| 12 | 9.57 (7.52–18.62) | 4.96 | 2.24 | 4.12 (3.1–5.0) | 2.57 (1.7–3.3) | 1.60 (1.52–1.85) |
| 18 | 10.41 (9.19–12.12) | 4.56 | 2.40 | — | — | — |
| 24 | 14.97 (13.35–16.35) | 4.47 | 3.26 | 4.85 (2.9–5.2) | 1.75 (1.4–2.6) | 2.14 (1.93–2.77) |
| 30 | 14.78 (12.15–16.23) | 4.27 | 3.38 | — | — | — |
| 36 | 15.44 (13.46–20.17) | 6.28 | 2.96 | 3.70 (2.9–4.2) | 1.61 (1.4–1.6) | 2.27 (1.76–2.63) |
| 42 | 19.36 (14.93–23.80) | 7.19 | 2.69 | — | — | — |
| 47 | 117.65 (94.36–146.54) | 22.94 | 5.23 | 14.23 (10.5–14.7) | 1.68 (1.5–2.1) | 7.07 (6.91–9.76) |

The token at the razor tie is again at the edge: token 21 has full block-0 $\sigma_{\max}=128.46$ (the largest of the 5 tokens), $\rho=17.47$, $\kappa=7.35$; block 47 $\sigma_{\max}=13.22$, $\rho=1.68$, $\kappa=7.87$. Its principal-angle median, $84.14°$, is the lowest of the five and the closest to the 1.5B's $83.6°$. Token 273 at block 47 is the one quoted in the run log: $\sigma_{\max}=14.23$, $\rho=1.457$, $\kappa=9.76$.

### 4.2 Comparison with the 1.5B non-normality

| quantity | 14B | 1.5B | verdict |
|---|---|---|---|
| full block 0 $\sigma_{\max}$ | 96.2 (66–128) | 95–195 | same range, 14B at the low end |
| full block 0 $\rho$ | 12.7 (8.5–17.5) | 11–39 | same range, 14B at the low end |
| full block 0 $\kappa$ | 7.35 (5.2–11.5) | 5–8 | same, 14B at the high end |
| full mid $\sigma_{\max}$ | 3.7–4.9 | 2.7–3.6 | 14B slightly higher (1.2–1.4×) |
| full mid $\kappa$ | 1.60–2.27 | 1.7–2.5 | **same normal middle** |
| full last $\sigma_{\max}$ | 14.2 (10.5–14.7) | 6–9 | 14B **1.6–2.4× higher** |
| full last $\kappa$ | 7.07 (6.9–9.8) | 3–5.3 | 14B **≈1.4–2.3× higher** |
| MLP last $\sigma_{\max}$ | 117.7 (94–147) | 58–93 | 14B **1.3–2.0× higher** |
| MLP last $\kappa$ | 5.23 | not quoted | — |
| median principal angle | 86.96° (per-token 84.1–87.8°) | 83.6° | 14B slightly more orthogonal |

The U-shape is the same; the difference is concentrated at the very end. The 1.5B has 28 blocks, the 14B 48, so the comparison is in fractional depth: the 14B minimum is at block 12 ($0.25$ of depth, $\kappa=1.60$), and the middle stays at $\kappa\approx2.1$–$2.3$ out to $0.75$; the spike is confined to block 47. The 1.5B's last-block $\kappa=3$–$5.3$ is below the 14B's $7.07$, but its block-0 range already overlaps the 14B. The scale-dependent change is therefore **not** a uniform increase of non-normality with model size; it is a stronger transient amplifier at the final block of the deeper model.

---

## 5. Summary comparison table

| quantity | 14B (true W=100) | 1.5B reference | same? |
|---|---|---|---|
| depth final $\lambda_i$ | 0.017–0.086 | 0.08–0.18 | lower in 14B |
| depth $h_{\rm depth}$ | 0.16–0.64 | 0.6–1.4 | lower in 14B; only token 21 reaches the band |
| absolute tangent growth | 2.7–45.9× / 48 blocks | positive, $e^{0.08}$–$e^{0.18}$/block | lower in 14B |
| state norm growth | 655–833× | ×30 block 0 + ×1.05–1.18 after | same qualitative |
| normalized relative exponent | −0.061 to −0.120; 61/64 chunks negative | every block negative, worst −0.034 | **confirmed**, 3 near-zero exceptions |
| absolute expansion is radial drift | tangent 2.7–45.9× vs norm 655–833× | stated | **confirmed, stronger** |
| full block 0 $\sigma_{\max}$ | 96.2 | 95–195 | same |
| full block 0 $\kappa$ | 7.35 | 5–8 | same |
| full mid $\sigma_{\max}$ | 3.7–4.9 | 2.7–3.6 | slightly higher |
| full mid $\kappa$ | 1.60–2.27 | 1.7–2.5 | same |
| full last $\sigma_{\max}$ | 14.2 | 6–9 | **higher in 14B** |
| full last $\kappa$ | 7.07 | 3–5.3 | **higher in 14B** |
| MLP last $\sigma_{\max}$ | 117.7 | 58–93 | **higher in 14B** |
| principal angle median | 86.96° | 83.6° | slightly more orthogonal |

No 1.5B single-position kick run exists in the reference set, so the state-kick part is not compared numerically to the 1.5B. Its mechanism, however, is the one the 1.5B depth result predicts: the raw response can grow late while the state-normalized tangent does not. The late burst at $t\approx291$ is an absolute response, and the depth run shows that such absolute growth coexists with normalized contraction.

---

## 6. What this does not show

- **No per-block state normalization.** `norm_h0` is saved only every 6 states. The relative exponent is a 6-block average; the "every block contracts" statement is tested at that resolution only. The last transformer block (47) is excluded from the relative chunks because state 48 is not saved.
- **The late burst is not decomposed.** The figure shows the norm of the layer-47 difference. It does not say which direction, head, or token causes the $t\approx291$ amplification. The coincidence of the burst time across kick positions and its absence of a state-norm spike make a trajectory/tangent explanation likely, not proven.
- **The flip counts are linear extrapolations.** $r_c^-$ uses `dm` at $r=0.1$; `dm` is already 18% sublinear between $r=0.03$ and $0.1$. The counts at the two radii agree within one direction, but the exact thresholds at $r_c>0.1$ are less certain. The $t=262$ flips, which require a margin change of order one nat, are the most extrapolation-sensitive.
- **The half-life is an initial decay time.** For late kicks the global maximum is at $t\approx291$, not at $s$; a literal "half of the global peak" half-life does not exist within 300 steps. The figure uses the immediate response as the peak and says so.
- **Only 8 tangent directions.** `h_depth` is the top-8 volume, not the full tangent spectrum; a 5120-dimensional Jacobian can have larger directions outside the sampled subspace. The spectrum is flat across the 8 sampled directions, which is consistent with, but not proof of, a flat full spectrum.
- **The non-normality is sampled.** Full-block metrics exist at 5 blocks and 5 tokens; MLP at 9 blocks. The curves between measured blocks are interpolation. The power iteration and Arnoldi use finite tolerances; $\rho$ is a largest-eigenvalue estimate, and $\kappa$ inherits that error. The 1.5B numbers are quoted from the reference set, not recomputed here.
- **The comparison mixes model size and attention span.** The 14B run is true window 100; the 1.5B reference protocol is whatever the T4 notebook used. The 28-vs-48 block difference is handled in fractional depth, but the attention window is not.
- **Token sets differ.** Depth uses $0,21,50,100,150,200,250,299$; non-normality uses $21,50,100,189,273$; state kicks use 10 positions. Statements about "the token at 21" are cross-run only where 21 is present in both.

## 7. What it implies

1. **A razor tie is a dynamical hotspot, not just a small number.** The token at $t=21$ has the largest absolute tangent growth ($h_{\rm depth}=0.638$, $45.9\times$ volume growth), the largest block-0 full-block $\sigma_{\max}$ ($128.5$), and the smallest principal angle ($84.1°$, closest to the 1.5B). A kick at $s=8$ or $16$ flips the $t=19/21$ razor pair 3–13 steps later — most directions first cross at $t=19$, the rest at $t=21$. The $t=21$ token is where the model is both decisive (smallest margin) and most amplificatory (largest tangent), which is why a perturbation injected much earlier can control it.
2. **Raw response is not the right measure of persistence.** The immediate response is largest at $s=50$ ($12.07$ per unit radius), but the late response is largest for $s=250$ ($98\times$ its immediate value). The absolute norm mixes the state scale and the tangent; the depth run shows the state norm grows $655$–$833\times$ while the tangent grows only $2.7$–$45.9\times$. Any claim that a deep model "amplifies perturbations" must specify whether it is absolute or state-normalized.
3. **The last block is a transient amplifier.** Full block 47 has a modest spectral radius, $\rho=1.68$ (growth $1.68\times$ per block in the long-time limit), but $\sigma_{\max}=14.2$ and $\kappa=7.07$; the MLP alone reaches $\sigma_{\max}=118$. A small kick can be stretched by an order of magnitude in one block even though its sustained growth per block is only $1.68\times$. This is the natural mechanism for the $t\approx291$ re-amplification and for the late flips at $t=262$–$276$.
4. **The 1.5B depth picture survives at 14B, with the last block more extreme.** The middle of the 14B is as normal as the 1.5B's ($\kappa\approx1.6$–$2.3$); the last block is more non-normal ($\kappa\approx7$ vs $3$–$5.3$). The normalized tangent contracts in 61 of 64 chunks at 6-block resolution; the absolute growth is a radial/state-scale effect plus a late tail. The open question is whether the last-block spike is a property of depth (48 vs 28 blocks) or of this particular model; answering it needs the same measurement on another deep model, not another analysis of this one.
