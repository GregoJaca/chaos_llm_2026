# Qwen3.5-9B: margins, thresholds, and the hybrid-attention layer profile

## Summary

Qwen3.5-9B is a **9B hybrid-attention model**: 32 blocks, hidden size 4096, and a repeating period of four in which three **Gated DeltaNet** (linear-attention) blocks are followed by one **full-attention** block (blocks $3,7,11,\dots,31$). It has no sliding window (the run metadata stores `window=0`). On the same prompt and protocol as the 14B — `interstellar_travel`, $P=64$ tokens, 300 greedy tokens, $r=0.1$, 1000 directions — its behaviour is **quantitatively between the 1.5B and the 14B, but much closer to the 1.5B in flippability**.

The calibration run (400 tokens) contains **no razor-thin tie**. Its smallest top-1/top-2 gap is $6.87\times10^{-3}$ nats at $t=327$, then $7.27\times10^{-3}$ at $t=267$, $1.74\times10^{-2}$ at $t=160$, $1.89\times10^{-2}$ at $t=322$ and $2.05\times10^{-2}$ at $t=69$. The smallest margin is **75x thicker than the 14B's $9.73\times10^{-5}$** and 2.8x thinner than the 1.5B's $2.02\times10^{-2}$. Its fp32 response floor is $6.1\times10^{-4}$, **4.4x below the 14B's $2.7\times10^{-3}$**, so the linear window is wider than the 14B's on the low side: $r\in[0.01,0.3]$.

The core A1 run ($K=1000$, $r=0.1$, $T=300$) has one step below 0.01 nats ($t=267$, $m=7.27\times10^{-3}$), 12 steps below 0.1, median margin 2.46 nats. The flux-tube radius has median **$r_*=0.707$** (14B: 0.116; 1.5B: 1.56), so only **12/1000 directions (1.2%) flip at $r=0.1$** — **38x fewer than the 14B's 45.3%** — while 35% need $r>1$ to flip anything in 300 tokens.

The new measurement is the **per-block layer profile**. The median response $\|\delta h_t^b\|/r$ grows **monotonically by 159x from block 0 to block 30** (30/30 positive increments), with no visible compression at the three linear-attention blocks in each period. The seven saved full-attention blocks grow on average **1.24x per block against 1.18x for the linear blocks** — a 5.3% excess per block — but a permutation over the 7 block labels gives **$p=0.18$** ($p=0.15$ after removing the linear depth trend), and two of the seven (blocks 7 and 11) show no excess at all. Only blocks 3, 15, 19 and 27 produce visible local spikes. The response also **decays by 3-4.5x along generated time** at every depth, and **relative to the baseline state norm it saturates after block 19** ($\|\delta h\|/\|h_0\|\approx4.0\times10^{-3}$ from block 19 to 30); most of the late absolute growth is radial/state-norm growth, not a growing angular perturbation.

Free-run validation is decisive. The signed top-2 threshold rule predicts **0/12/160 flips** at $\rho=0.05/0.1/0.3$ against **0/12/157 observed**; at $\rho=0.1$ all 12 flips are exact, and at $\rho=0.3$ the classification is **91.8% accurate** (104 exact, 27 wrong step, 29 false alarms, 26 misses). The sign-blind rule would predict 20 and 305 flips and is rejected. Every observed flip is a **top-2 word swap** (`Ex`$\to$`Specific` at 267, `Distance`$\to$`The` at 69, …), and **0/169 flipped rows re-synchronise**; post-flip agreement is 2.9% at $\rho=0.1$ and 2.0% at $\rho=0.3$ (0.55% after the echoed task sentence).

Finally, because `h0_last` is stored in this run, the response-vs-state angle is measurable for the first time at this scale: the median angle between $\delta h$ and the baseline final state is **90.08°** (IQR 89.51-90.91°), with 80.2% of pairs inside $87^\circ\pm5^\circ$ and 54.7% beyond 90°. The 1.5B's "responses sit about 87° from the state" is reproduced and sharpened: at 9B the response is, if anything, **exactly perpendicular**.

---

## 1. What was measured

### 1.1 Setup

| item | value |
|---|---|
| model | `Qwen3.5-9B`, `Qwen3_5ForConditionalGeneration`, FP32 |
| blocks / width | 32 / $d=4096$; vocabulary 248320 (padded) |
| attention | period 4: three **Gated DeltaNet** linear-attention blocks, then one full-attention block |
| full-attention blocks | $3,7,11,15,19,23,27,31$ (8 total) |
| sliding window | **none** (documented null; metadata `window=0`) |
| prompt | `interstellar_travel`, $P=64$ tokens |
| calibration | 8 directions x 9 radii, 400 greedy tokens, prompt perturbation |
| A1 | 1000 directions, $r=0.1$, $T=300$, teacher-forced on the baseline tokens |
| free run | same 1000 directions, greedy, own KV cache, $\rho\in\{0.05,0.1,0.3\}$ |
| files | `results_big/qwen35/calib/calib.npz`, `results_big/qwen35/a1_inter_w0/a1.npz`, `results_big/qwen35/free_inter_w0/{free_run.npz,summary.json}` |

The A1 margins are self-consistent with the baseline tokens: `top1[t] == gen_ids[t]` for all 300 steps, and the in-batch duplicate $r=0$ rows are bit-identical (stored flag). The free-run `summary.json` fields `mean_postflip_agreement` (1.000 / 0.998 / 0.941) are **whole-run agreement including the 988 and 843 non-flipped rows**; they are not post-flip means and are not used as such below.

### 1.2 Indexing: states versus blocks

This matters because the npz stores 32 state slots but the model has 32 blocks, and the last full-attention block is not among them.

| object | convention (verified against `src/chaos_llm/big_engine.py:Capture`) |
|---|---|
| state $0$ | embedding output |
| state $l\ge1$ | output of **block $l-1$** |
| saved states | $0,1,\dots,31$; therefore **blocks $0,\dots,30$** |
| block $31$ (full attention) | output is state $32$, **not saved** |
| full-attention blocks visible in the profile | $3,7,11,15,19,23,27$ (outputs = states $4,8,12,16,20,24,28$) |
| `dh_last` / `h0_last` | final-norm output (after block 31 and RMSNorm), i.e. state 32; **24 directions only** |

All figure panels and tables below label by **block index $b$**, not state. State 0 is identically zero in both the calibration and the A1 run: the perturbation touches only the 64 prompt positions, so nothing can appear at generated positions before the first attention mix.

### 1.3 Quantities

| symbol | explicit expression | meaning | why this quantity |
|---|---|---|---|
| $m_t$ | $\ell_t^{(1)}-\ell_t^{(2)}$ | top-1 minus top-2 logit gap of the unperturbed run (nats) | the only scalar that says how close a greedy decision was |
| $\delta m_t(u)$ | $m_t(u)-m_t(0)$ at $r=0.1$ | change of the gap under direction $u$ | first-order response of the decision |
| $r_c^-(t,u)$ | $m_t\,r/\max(-\delta m_t(u),0)$; $+\infty$ if $\delta m_t\ge0$ | radius at which step $t$ would flip | signed threshold; a positive $\delta m$ widens the gap and cannot flip |
| $r_*(u)$ | $\min_t r_c^-(t,u)$ | flux-tube radius | smallest radius at which direction $u$ flips anything in 300 tokens |
| $t_*(u)$ | $\arg\min_t r_c^-(t,u)$ | controlling step | which decision is the weak point |
| $T_{\rm pred}(\rho,u)$ | $\min\{t:\rho\ge r_c^-(t,u)\}$ | predicted first flip | comparable with the free run's `obs` |
| $G(u)$ | $\mathrm{median}_t\|\delta h_t^{\rm last}(u)\|/r$ | response gain per unit radius | how strongly a direction moves the final state |
| $\mathrm{angle}(u,t)$ | $\arccos\!\big(\langle\delta h_t^{\rm last},h_{0,t}^{\rm last}\rangle/(\|\delta h\|\,\|h_0\|)\big)$ | angle between the response and the baseline state | tests the 1.5B claim that responses are near-perpendicular to the state |

For the layer profile, the response is $\|\delta h_t^b\|/r$ with $b$ the **block** whose output is state $b+1$; the division by $r$ is exact in the measured linear window and makes the profile comparable across radii. Percentile bands are over the 1000 directions (or over $t$ where stated).

**One discrepancy with the brief, flagged rather than copied.** The brief quotes a calibration median margin of "~3.99". The saved `calib.npz` margins over 400 steps have **median 2.634 and mean 4.143**; 3.99 is not a median of this array (it is closest to the mean). The values below are the file's values. The A1 300-token margins have median 2.461 and mean 4.054, consistent with the first 300 steps of the calibration run.

---

## 2. Figure `q35_calib.png` — no razor tie, a wider linear window, a lower floor

![calibration](figures/q35_calib.png)

**Setup.** Eight unit directions on the prompt embedding, nine radii from $10^{-4}$ to $1$, 400 greedy tokens, teacher-forced on the model's own continuation. The response is read at the last saved state (state 31 = output of block 30). The left panels use the unperturbed margins; the right panels use $\mathrm{median}_{u,t\in[100,400)}\|\delta h_t^{\rm last}\|$.

**Axes.** (a) $x$: generated step $t$; $y$: $m_t$ on a log scale; the dotted line is the 300-token A1 horizon. (b) $x$: $m_t$ on a log scale; $y$: fraction of steps with margin $\le m$. (c) log-log: $x$ radius $r$, $y$ median response $\|\delta h\|$; dotted lines are slope-1 guides anchored at $r=0.1$; the dashed horizontal line is the fp32 floor. (d) $x$: radius; $y$: $\|\delta h\|/r$; the shaded band is the clean window $r\in[0.01,0.3]$, the red dotted line is the campaign radius $r=0.1$.

| $r$ | $10^{-4}$ | $3\times10^{-4}$ | $10^{-3}$ | $3\times10^{-3}$ | $10^{-2}$ | $3\times10^{-2}$ | $0.1$ | $0.3$ | $1$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $\|\delta h\|$ (state 31) | 6.12e-4 | 6.45e-4 | 8.28e-4 | 1.67e-3 | 5.07e-3 | 1.50e-2 | 5.02e-2 | 1.54e-1 | 6.96e-1 |
| $\|\delta h\|/r$ | 6.12 | 2.15 | 0.828 | 0.558 | **0.507** | **0.501** | **0.502** | **0.512** | 0.696 |

The same plateau appears at the other saved states: $0.0265$ (state 8), $0.0614$ (state 16), $0.214$ (state 24) at $r=0.1$, each flat to a few percent across $r\in[0.01,0.3]$.

- **The floor is lower than the 14B's, not higher.** At $r\le3\times10^{-4}$ the last-state response is pinned at $6.1$-$6.4\times10^{-4}$; the 14B floor was $2.7\times10^{-3}$, the 1.5B floor $10^{-3}$. This is the opposite of a naive "more layers, more rounding" expectation. Two things can explain it and are not separated here: the 32-block depth is smaller than the 14B's 48, and the Gated DeltaNet blocks round a fixed-size recurrent state rather than a growing KV cache. At $r=1$, the response is 1.39x the plateau, the usual superlinear saturation.
- **Clean window.** The gain is flat to $\le1\%$ from $10^{-2}$ to $0.3$; at $3\times10^{-3}$ it is already 11% high, and at $10^{-3}$ 65% high, so the honest window is $[0.01,0.3]$. The campaign radius $0.1$ sits in the middle, with signal-to-floor $=0.0502/0.000612\approx82$ (the 14B's ratio was 17.5).
- **Margins.** Over 400 steps: median 2.634, mean 4.143, p10 0.329, minimum $6.87\times10^{-3}$ at $t=327$; 16 steps below 0.1 nats and 2 below 0.01. The five smallest are 327 ($6.87\times10^{-3}$), 267 ($7.27\times10^{-3}$), 160 ($1.74\times10^{-2}$), 322 ($1.89\times10^{-2}$), 69 ($2.05\times10^{-2}$). There is **no tie anywhere near the 14B's $9.7\times10^{-5}$**; the thinnest gap is 75x thicker. The A1 300-token run confirms the same spectrum and makes $t=267$ the controlling minimum because the 327 step is outside its horizon.

---

## 3. Figure `q35_layer_profile.png` — monotone 159x growth, no linear-block compression, a weak full-attention excess

![layer profile](figures/q35_layer_profile.png)

**Setup.** Same A1 run: 1000 directions at $r=0.1$, 300 tokens, all 32 saved states. For each block $b$ and step $t$, the plotted quantity is $\mathrm{median}_u\|\delta h_t^b\|/r$; $b$ runs from 0 to 30 (the output of block $b$ is state $b+1$). The seven full-attention blocks whose outputs are saved are marked in red/cyan; block 31's output (state 32) is not in the file.

**Axes.** (a) Heatmap: $x$ generated step $t$, $y$ block index $b$, color $\log_{10}$ of the median response; cyan horizontal rules and cyan tick labels mark full-attention blocks. (b) $x$ block index $b$, $y$ median $\|\delta h\|/r$ on a log scale; the band is the 10-90% range over $t$; red dots are full-attention blocks, blue dots linear blocks. (c) $x$ block index $b$, $y$ the ratio of the median response at state $b+1$ to that at state $b$; red bars are full-attention blocks; the dashed line is the mean over blocks 1-30.

**Depth growth is monotone and large.** The median-over-$t$ response rises from $5.3\times10^{-3}$ at block 0 to $0.843$ at block 30, a factor **159x**; every one of the 30 block-to-block intervals is positive (30/30). There is no periodic sawtooth at the 3:1 pattern in this median-over-$t$ profile: the three linear blocks between two full-attention blocks always add response, never subtract it. The first finite block-to-block jump is 2.00x at block 1, then 1.37x at block 2, 1.45x at block 3 (full), 1.35x at block 15 (full), 1.32x at block 19 (full) and 1.19x at block 22 (linear); the late blocks are in the 1.09-1.17x range regardless of type. (Per individual step some blocks do contract; the statement here is about the median over the 1000 directions and 300 steps.)

| block $b$ | 3 | 7 | 11 | 15 | 19 | 23 | 27 | mean full | mean linear (1-30) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| growth through block | 1.451 | 1.148 | 1.111 | 1.353 | 1.315 | 1.159 | 1.159 | **1.242** | **1.180** |

The full-attention mean is 5.3% higher per block (1.242/1.180). A permutation test over the 7 full-attention labels among blocks 1-30 gives **$p=0.179$**; after removing a linear depth trend, **$p=0.155$**. The per-step version is consistent in sign: 76.7% of the 300 steps have a positive full-minus-linear growth difference, with mean $+0.043$ nats/block. So the effect is **suggestive but not significant with seven blocks**, and it is not uniform: blocks 3, 15, 19 and 27 spike, while blocks 7 and 11 are indistinguishable from their linear neighbours. This is the honest answer to "is there a visible signature at the full-attention blocks": a weak one, not a clean periodic one.

**Time decay.** At every depth the response is strongest near the prompt and decays with generated step. The ratio of the median over $t\in[250,300)$ to the median over $t\in[0,50)$ is 0.32 at state 1, 0.30 at state 4, 0.32 at state 8, 0.34 at state 12, 0.31 at state 16, 0.23 at state 20, 0.22 at state 24, 0.24 at state 28 and 0.28 at state 31. The heatmap shows the same pattern as vertical banding: the perturbed prompt's influence is diluted as generation proceeds, with individual late tokens acting as local amplifiers. This decay is not specific to the linear blocks; it is present at shallow and deep states alike.

**Relative response.** Dividing by the baseline state norm $\|h_0\|$ removes the radial growth. The relative response rises from $8.6\times10^{-4}$ at block 0 to $4.22\times10^{-3}$ at block 19 and then stays at $3.9$-$4.3\times10^{-3}$ through block 30. The absolute 159x growth is therefore **not** a 159x growth of the angular perturbation: after block 19 the residual stream is mostly growing in magnitude, and the perturbation grows with it.

**Caveat.** The last full-attention block (31) is not in the saved states. Its post-final-norm response is available only for the 24 `dh_last` directions and is $0.643$ per unit radius (median over directions of the per-direction time median), but the final RMSNorm rescales the vector and makes that number **not directly comparable** to the pre-norm block profile. No claim is made about block 31's contribution.

---

## 4. Figure `q35_margins_rstar.png` — a wide flux tube, two controlling steps

![margins and rstar](figures/q35_margins_rstar.png)

**Setup.** A1 run only ($r=0$, margins; $r=0.1$, responses), 1000 directions, 300 tokens. The signed threshold is used throughout: $r_c^-(t,u)=m_t r/\max(-\delta m_t,0)$, with $+\infty$ when the perturbation widens the gap.

**Axes.** (a) $x$: generated step $t$; $y$: $m_t$ on a log scale; the five smallest are marked and the word pair at each is listed. (b) $x$: $r_*$ on a log scale; $y$: fraction of directions with $r_*\le r$; the dashed red line is the run radius, the dotted lines are the 14B and 1.5B medians. (c) Horizontal bars: the ten steps that control the most directions; pale = all 1000, red = the subset that flips at $r=0.1$. (d) $x$: radius on a log scale; $y$: fraction of directions predicted to flip; the black curve is the signed rule over all 1000, the orange points are the free-run observations at 0.05/0.1/0.3 with binomial error bars.

**Margins.** Median 2.461 nats, mean 4.054, p10 0.310, p1 0.0286; 12/300 steps below 0.1 and **1/300 below 0.01**. The five smallest:

| $t$ | 267 | 160 | 69 | 40 | 1 |
|---|---:|---:|---:|---:|---:|
| $m_t$ | **7.27e-3** | 1.74e-2 | 2.05e-2 | 2.87e-2 | 5.55e-2 |
| top-1 vs top-2 | `Ex` / `Specific` | ` of` / ` that` | `Distance` / `The` | `###` / `The` | `This` / `<think>` |

The thinnest gap is a **word-choice tie inside a bullet list**: the baseline writes `**Exhaust Velocity...`, while the runner-up opens `**Specific Impulse...`. Both are grammatically and semantically plausible, which is exactly why a 0.007-nat gap is dangerous.

**Flux-tube radius.** $r_*$ quantiles: p10 0.232, p25 0.394, p50 **0.707**, p75 1.254, p90 1.956, p95 2.516; 350/1000 directions need $r>1$. At $r=0.1$ only **12 directions (1.2%)** cross; at $r=0.3$, 16.0%. The 14B's p50 was 0.116 (6.1x smaller) and its flip fraction at 0.1 was 45.3% (38x larger). The 1.5B's p50 was 1.56 (2.2x larger) and it flipped 0/20 at 0.1. The Qwen3.5-9B flux tube is therefore intermediate but much closer to the 1.5B.

**Danger ladder.** Only 16 distinct steps control any of the 1000 directions, and only **2** control a direction that flips at $r=0.1$:

| step | $m_t$ | all directions | flip at $r=0.1$ |
|---:|---:|---:|---:|
| 267 | 7.27e-3 | 315 | 11 |
| 69 | 2.05e-2 | 184 | 1 |
| 40 | 2.87e-2 | 91 | 0 |
| 1 | 5.55e-2 | 91 | 0 |
| 160 | 1.74e-2 | 81 | 0 |
| 123 | 7.80e-2 | 78 | 0 |
| 154 | 7.91e-2 | 45 | 0 |
| 230 | 1.59e-1 | 45 | 0 |
| 95 | 7.73e-2 | 24 | 0 |
| 39 | 8.62e-2 | 17 | 0 |

The distribution is not a single knife edge: it is a **small set of moderately thin decisions** (0.007-0.16 nats) spread through the continuation. That is qualitatively different from the 14B, where 503 directions were controlled by $t=21$ alone at $9.7\times10^{-5}$.

---

## 5. Figure `q35_free_validation.png` — the signed rule predicts all 12 flips at $\rho=0.1$

![free validation](figures/q35_free_validation.png)

**Setup.** The same 1000 directions and prompt perturbation as A1, now generated greedily with the model's own KV cache for 300 tokens at $\rho=0.05,0.1,0.3$. `obs` is the first generated token that differs from the baseline; `pred` is recomputed here with the **signed** rule from A1's $m_t$ and $\delta m_t$ (the npz's stored `pred` is the sign-blind `m r/|\delta m|` variant and is not used for the scorecard). Both variants are reported.

**Axes.** (a) Stacked bars of the four informative classes (exact, wrong step, false alarm, miss) per radius; the tick labels give observed/predicted flips and the number of both-none directions. (b) Observed vs signed-predicted flip percentage, with the sign-blind prediction as purple crosses. (c-e) Survival $S(t)=P(\text{first flip}>t)$ per radius; black = observed, orange dashed = signed prediction; dotted verticals at $t=69$ and $t=267$.

| | $\rho=0.05$ | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|---:|
| observed flips / 1000 | 0 | 12 | 157 |
| predicted, signed | 0 | 12 | 160 |
| predicted, sign-blind | 2 | 20 | 305 |
| exact | 0 | **12** | 104 |
| wrong step | 0 | 0 | 27 |
| false alarm | 0 | 0 | 29 |
| miss | 0 | 0 | 26 |
| both none | 1000 | 988 | 814 |
| classification accuracy | 100% (trivial) | **100%** | **91.8%** |
| observed mean / median flip step | — | 250.5 / 267 | 187.1 / 267 |
| $A_{\rm after}$ (post-flip agreement) | — | 0.0286 | 0.0201 |
| $A_{\rm late}$ ($t>{\rm obs}+45$) | — | 0.000 | 0.0055 |
| longest identical post-flip run | — | 1 | 1 |
| resynchronised (run $\ge20$) | — | 0/12 | 0/157 |

- **At $\rho=0.1$ the rule is exact for every event and every step.** The 12 flips are 11 at $t=267$ (`Ex`$\to$`Specific`) and 1 at $t=69$ (`Distance`$\to$`The`), exactly the two controlling steps of the A1 ladder. The survival curves coincide to the printed precision (max $|\Delta S|=0$).
- **At $\rho=0.3$ the event count is right (160 predicted vs 157 observed) but individual directions churn.** 104 directions are exact, 27 flip at a different one of the weak steps, 29 predicted flips do not happen, and 26 observed flips were not predicted. All 157 flips are still top-2 swaps: 267 (`Ex`$\to$`Specific`) 89, 69 (`Distance`$\to$`The`) 30, 123 (` Chemical`$\to$` Current`) 21, 40 (`###`$\to$`The`) 8, 1 (`This`$\to$`<think>`) 4, 160 (` of`$\to$` that`) 2, 41 (` `$\to$` Executive`) 2, 230 (` Near`$\to$` Proposed`) 1. The survival curve is reproduced to 0.008 at worst (at $t=39$).
- **The sign-blind rule is rejected.** It predicts 2/20/305 flips at the three radii. At $\rho=0.05$ that is already 2 false alarms out of 2 predictions; at $\rho=0.3$ it overpredicts by 14.8 points (305 vs 157). The sign of $\delta m$ matters even in a model with no razor tie.
- **No re-synchronisation.** The longest identical post-flip run in any of the 169 flipped rows is **1 token**, and 0 rows have a 20-token identical run or suffix. The response is not a transient perturbation that the model shakes off; once the first token differs, the continuation stays on a different trajectory. The late agreement of 0.55% at $\rho=0.3$ is the echoed prompt sentence plus coincidental punctuation.

The $\rho=0.05$ row is trivially perfect: with zero observed and zero signed-predicted flips, accuracy is 100% by construction and carries no step information. It is still a useful null: the signed rule does not hallucinate a flip where the model is stable.

---

## 6. Figure `q35_gain_structure.png` — 15x gain spread, orthogonal responses, 90.1° to the state

![gain structure](figures/q35_gain_structure.png)

**Setup.** A1 run. Gain is computed for all 1000 directions at the last saved block state (block 30): $G(u)=\mathrm{median}_t\|\delta h_t^{\rm last}(u)\|/r$. The pairwise cosines and the angle use the 24 stored `dh_last` vectors (final-norm output, 300 steps x 4096) and the stored `h0_last`; each direction's representative vector is its response at the step where its norm is closest to that direction's median norm (same convention as the 14B gain figure).

**Axes.** (a) $x$: $G$ on a log scale; left $y$: directions per bin, right $y$: ECDF; the red line is the median and the band the IQR. (b) $x$: $\cos(\delta h_i,\delta h_j)$ between two response vectors; $y$: pairs per bin. (c) $x$: $\cos(\delta h,h_0)$ at the final-norm state; $y$: direction $\times$ step pairs; the dashed orange line is the 1.5B's 87°, the red line this model's median.

| quantity | value |
|---|---:|
| gain p5 / p25 / p50 / p75 / p95 | 0.477 / 0.608 / **0.822** / 1.167 / 2.096 |
| gain min / max / max-min ratio | 0.361 / 5.449 / **15.1x** |
| pairwise mean / median $\cos$ | $-0.0047$ / $-0.0037$ |
| pairs with $|\cos|>0.1$ | 6.9% |
| first PC share (isotropic baseline) | 6.9% (4.2%) |
| temporal cosine, lag 1 / 10 / 50 | 0.370 / 0.067 / 0.037 |
| angle median / mean | **90.08°** / 90.25° |
| angle IQR / p1-p99 | 89.51-90.91° / 78.23-102.41° |
| fraction $>90°$ | 54.7% |
| fraction within $87^\circ\pm5^\circ$ | 80.2% |
| median $\|\delta h\|/\|h_0\|$ | $4.9\times10^{-4}$ |
| stored $\cos(h',h_0)$ min / max | 0.99975 / 1.0000000 |

- **Gain is broadly log-normal.** The 15.1x spread is similar to the 14B's 13.7x and the 1.5B's 11.8x; the median 0.822 sits between the 14B's 0.583 and the 1.5B's 1.333.
- **Responses are nearly orthogonal across directions.** The mean pairwise cosine is $-0.005$, only 6.9% of pairs exceed $|0.1|$, and the first PC carries 6.9% of the variance against 4.2% for an isotropic 24-vector cloud. There is no single shared response mode. The temporal cosine is 0.37 at lag 1 and 0.07 at lag 10: the response direction decorrelates within about ten generated steps, much faster than the norm decays.
- **The angle is the new result.** At the final-norm state the response is **perpendicular to the baseline state**: median 90.08°, IQR 89.51-90.91°, mean cosine $-0.0044$, and 54.7% of pairs beyond 90°. The per-step median is stable at 89.5-92.5° for 95% of steps; the two exceptions are $t=0$ (80.7°, the first generated position, where the response is smallest and least settled) and $t=234$ (97.3°). This **confirms and sharpens the 1.5B's ~87°**: at 9B the response is not merely near-perpendicular; its component along the state is zero to the precision of this measurement.
- **Why the stored `cos` field cannot be used for this.** `a1.npz` stores $\cos(h',h_0)$, which is 1 to fp32 precision (minimum 0.99975) because $\|\delta h\|/\|h_0\|\approx5\times10^{-4}$. The angle requires the actual $h_0$ vector, which exists only for the final-norm state and only 24 directions; that is the measurement above. The same limitation blocked the 14B analysis (its `a1.npz` stored no baseline state vectors), so the cross-model angle comparison has only the 1.5B's literature value and this run.

---

## 7. Cross-model comparison

All rows use the same prompt, greedy decoding, FP32, and $r=0.1$ unless stated. The 1.5B and 14B values are from Parts 1-3 and the cross-scale/floor sections; the Qwen3.5-9B values are from this section.

| quantity | 1.5B | DS-14B | **Qwen3.5-9B** |
|---|---:|---:|---:|
| blocks / width | 28 / 1536 | 48 / 5120 | **32 / 4096** |
| attention | full | full | **3 linear : 1 full** |
| $K$ directions | 20 | 1000 | **1000** |
| margin min (step) | 2.02e-2 (123) | 9.73e-5 (21) | **7.27e-3 (267)** |
| margin median | 4.78 | 3.10 | **2.46** |
| $r_*$ p50 / p90 | 1.56 / 4.12 | 0.116 / 2.59 | **0.707 / 1.96** |
| fp32 floor (last saved) | 1.0e-3 | 2.7e-3 | **6.1e-4** |
| clean radius window | [1e-3, 0.1] | [0.03, 0.3] | **[0.01, 0.3]** |
| gain $G$ p50 (range) | 1.33 (11.8x) | 0.583 (13.7x) | **0.822 (15.1x)** |
| flip fraction at $r=0.1$ | 0/20 | 45.3% | **1.2%** |
| free flip fraction 0.05 / 0.1 / 0.3 | — / 0 / — | 32.1% / 45.3% / 64.3% | **0% / 1.2% / 15.7%** |
| response-vs-state angle | ~87° (claimed) | not recoverable | **90.08°** |

Reading the table:

- **Margin thickness.** Qwen3.5-9B is **75x thicker than the 14B** and 2.8x thinner than the 1.5B. It therefore sits much closer to the small model in decision fragility, despite being 6x larger.
- **Threshold radius.** $r_*$ p50 is 6.1x wider than the 14B's and 2.2x narrower than the 1.5B's. The ordering of flippability is 14B $\gg$ Qwen3.5-9B $>$ 1.5B, and at $r=0.1$ the Qwen is 38x harder to flip than the 14B.
- **Floor.** The Qwen floor is the **lowest of the three**, 4.4x below the 14B's and 1.6x below the 1.5B's. The naive "deeper model, higher floor" rule does not hold across architectures; the 14B's 48 full-attention blocks are worse than the Qwen's 32 hybrid blocks.
- **Gain.** The Qwen gain is intermediate. Its 15.1x spread is the widest of the three, so the median gain is a weak summary: direction choice matters at least as much as model size.
- **Angle.** The perpendicular-response picture survives the architecture change from full attention (1.5B) to hybrid linear/full attention (Qwen3.5-9B). That is evidence that the effect is a property of the residual stream and the final norm, not of the attention mechanism.

---

## 8. Prediction scorecard

The scorecard is the free-run validation in §5. In one line: **signed top-2 rule, 0/12/160 predicted vs 0/12/157 observed; 100% exact at $\rho=0.1$, 91.8% classification at $\rho=0.3$; sign-blind 2/20/305 is rejected.** The two quantities that make the rule work are the **sign** of $\delta m_t$ (a margin can be widened, and then no finite radius flips it) and the **first crossing in time** (a step with a small instantaneous threshold can be crossed later than a step with a larger one). The layer profile adds a third, structural fact: the response is not concentrated in one attention type, so a single "full-attention layer" correction would not improve the prediction.

---

## 9. Caveats and what this does not show

1. **Block 31 is missing from the layer profile.** The saved state array stops at state 31 = output of block 30. The last full-attention block's output (state 32) was not stored; `dh_last` gives its post-final-norm response for 24 directions, but the RMSNorm makes it not comparable to the pre-norm profile. The "seven full-attention blocks" in the permutation test are therefore 7 of 8.
2. **The full-attention excess is not significant.** With seven labels and a strong depth trend, $p=0.18$ (0.15 detrended) is weak evidence. The honest statement is "a ~5% per-block excess, visible at four of seven blocks, not established". A cleaner test would need the same profile in a full-attention model of the same depth and width, which does not exist in this campaign.
3. **The calibration median in the brief is not the file's median.** The file gives 2.634 (mean 4.143), not ~3.99; the table and figures use the file. If the 3.99 came from a different aggregation in the run log, that aggregation is not in `calib.npz`.
4. **The free run at $\rho=0.05$ is a null with no power.** Zero flips and zero signed predictions make the accuracy 100% by construction. The informative radii are 0.1 (12 flips, all exact) and 0.3 (157 flips, 91.8% classification).
5. **The angle is a 24-direction, one-state measurement.** It uses the final-norm state only. The per-step median is stable, but the tails (p1 78°, p99 102°) and the two outlier steps show that a single direction's angle can be noisy when its response is small.
6. **No sliding window.** The architecture has none; the `window` field is 0 and window experiments are undefined. The comparison with the 14B's true-window-100 run is therefore a comparison of architectures, not of window settings.
7. **The linear-attention compression hypothesis is not supported at the residual-stream level.** This run measures the residual stream, not the Gated DeltaNet recurrent state itself. A fixed-size recurrent state can lose old information internally while the residual stream still carries a growing perturbation; the measurement here says the latter dominates for 300 tokens. It does not say the recurrent state is lossless.
