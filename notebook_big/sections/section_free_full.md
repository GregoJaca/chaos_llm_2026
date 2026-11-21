# Part free-run. Validating the signed flip rule against free-running generation (14B, full attention)

## Summary

At free-run radii $\rho=0.05,0.1,0.3$, the fraction of the 1000 directions that actually change the greedy continuation is 32.1%, 45.3% and 64.3%, against 31.6%, 46.4% and 67.1% predicted by the signed teacher-forced rule; at the calibrated radius $\rho=0.1$ there are 449 exact step matches, 1 wrong step, 14 false alarms and 3 misses out of 1000. Every flip happens at one of two razor-thin ties, $t=19$ or $t=21$ (plus 33 at $t=189$ and one at $t=273$ only at $\rho=0.3$), where the competing tokens are `" reference"` vs `" research"` and `"Alright"` vs `"Okay"`. The stored 73.0/61.9/47.5% "post-flip agreement" is not a post-flip number: it averages over all directions, including the non-flipped ones that contribute exactly 1.0; among flipped directions the whole-run agreement is 16.0/15.9/18.4%, and strictly after the first flip it is 9.7/9.7/9.6%. After the shared 18-token restatement of the prompt the agreement falls to 1.7%, no run ever re-synchronises to the baseline, and the large apparent agreement is local boilerplate, not re-convergence. The flipped runs do re-converge to each other: all 33 saved raw continuations contain only three distinct texts, and flips at the two gates differ by a single token.

---

## 1. The run, the quantities, and what is being validated

### 1.1 Setup

| item | value |
|---|---|
| model | `DeepSeek-R1-Distill-Qwen-14B` (48 blocks, hidden size 5120) |
| precision | FP32 |
| attention | **full attention** — the intended window-100 setting was silently inactive (correction note in the notebook); a true-window rerun is being generated |
| prompt | `interstellar_travel`, $P=65$ tokens, identical to the A1 run |
| baseline | $T=300$ greedy tokens, no sampling, no EOS stop |
| A1 (calibration) | 1000 unit directions at $r=0.1$, added to the prompt embeddings only; every perturbed pass teacher-forced on the baseline `prompt_ids`+`gen_ids`; 50 disjoint blocks of 20 non-zero coordinates per direction |
| free run | the **same** 1000 directions and the same perturbation construction, at $\rho\in\{0.05,0.1,0.3\}$; each row generates its own 300 greedy tokens with a KV cache |
| reference | an in-batch $r=0$ row in every chunk; it is not stored in `gen_ids`, but the non-flipped saved rows are bit-identical to A1's `gen_ids` |
| baseline identity | A1 `top1[t] == gen_ids[t]` for all $t$ (checked); the baseline is self-consistent |
| files | `results_big/ds14/a1_inter_full/a1.npz`, `results_big/ds14/free_inter_full/free_run.npz`, `results_big/ds14/free_inter_full/summary.json` |

Two data facts have to be stated before any number. (i) The summary's `run` field says `results_big/ds14/a1_inter_w100`, but the correction note applies: that run used full attention. The free rows confirm it — every non-flipped row equals the full-attention A1 `gen_ids` exactly. (ii) The writer of `free_run.npz` overwrote its `all_gen` dictionary once per generation chunk, so `gen_ids` contains only the **last** chunk: directions 989–999, 11 of 1000, at each radius. The mapping is not assumed: the first mismatch of each row reproduces `obs[:,989:1000]` exactly for all three radii, and non-flipped rows are bit-identical to A1. Direct token-level post-flip analysis therefore rests on 19 flipped rows (4 at $\rho=0.05$, 7 at $\rho=0.1$, 8 at $\rho=0.3$); the population statements below are recovered exactly from the stored whole-run agreement, as shown in §5.

### 1.2 Quantities

Notation: $u$ indexes a direction, $t\in\{0,\dots,299\}$ a generated step, $m_t$ the top-1 minus top-2 logit gap of the unperturbed A1 run, $\delta m_t(u)$ its change under direction $u$ at the A1 radius $r=0.1$, and $\rho$ the free-run radius.

| symbol | explicit expression | meaning | why this quantity |
|---|---|---|---|
| $m_t$ | $\ell^{(1)}_t-\ell^{(2)}_t$ (nats) | gap between the two best tokens of the baseline | a flip needs this gap closed |
| $\delta m_t(u)$ | $m_t(u)-m_t$ (nats), stored `dm` | first-order change of that gap at $r=0.1$ | response of the decision variable, measured without letting the text diverge |
| $r_c^-(t,u)$ | $m_t\,r/\max(-\delta m_t(u),0)$; $+\infty$ if $\delta m_t\ge0$ | radius at which step $t$ would flip | solve $-\delta m_t\ge m_t$ for the radius; only a **negative** $\delta m$ can close a gap |
| $r_*(u)$ | $\min_t r_c^-(t,u)$ | flux-tube radius | one number per direction: the largest radius that leaves every baseline argmax intact |
| $t_*(u)$ | $\arg\min_t r_c^-(t,u)$ | controlling tie | which near-tie is the global weak point |
| $T_{\rm pred}(\rho,u)$ | $\min\{t:\rho\ge r_c^-(t,u)\}$, $-1$ if no $t$ | first predicted flip **in time** | the prediction that the free run tests; $\rho\ge r_*(u)$ is equivalent to $T_{\rm pred}\ge0$ |
| ${\rm obs}(\rho,u)$ | first $t$ with free token $\ne$ baseline token, $-1$ if none | observed first flip | the experimental outcome |
| $A_{\rm after}(\rho,u)$ | $\frac{1}{T-{\rm obs}}\sum_{t>{\rm obs}}\mathbf 1[g_t=\hat g_t]$ | token agreement strictly after the first flip | does the run come back to the baseline text? |
| $A_{\rm late}(\rho,u)$ | the same sum for $t>{\rm obs}+45$ | agreement past the echoed task sentence | isolates genuine re-convergence from shared prompt restatement |
| $\Delta(u)$ | ${\rm obs}(\rho,u)-t_*(u)$ | latency relative to the controlling tie | "immediate" if 0, "delayed" if $>0$; negative is possible because the first crossing need not be the global minimum |

**Signed versus sign-blind.** The npz's stored `r_c` is built with $|\delta m_t|$, not $-\delta m_t$. Recomputation reproduces it to $1.5\times10^{-8}$, so the stored array is exactly $m_t r/|\delta m_t|$. A direction with $\delta m_t>0$ widens the gap and can never flip a top-1/top-2 decision, yet the sign-blind rule calls it a flip at a finite radius. At $\rho=0.1$ that rule predicts 810 flips against 464 for the signed rule; 360 of the 810 are directions that never mismatch in the free run (44.4% false-alarm rate). The signed rule's false-alarm rate is 3.0%. Every prediction below is the signed one.

**First crossing versus argmin.** $T_{\rm pred}$ is the first time the threshold is crossed, while $t_*$ is the time of the smallest threshold. These differ when an earlier step has a larger-but-still-crossed threshold: at $\rho=0.1$, 31 directions flip at $t=19$ while $t_*=21$. Those runs are exact predictions, not errors; the distinction is why $\Delta$ can be $-2$ without any failure of the rule.

### 1.3 The result in one table

| quantity | $\rho=0.05$ | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|---:|
| observed flips / 1000 | 321 | 453 | 643 |
| predicted flips, signed / 1000 | 316 | 464 | 671 |
| predicted flips, sign-blind / 1000 | 599 | 810 | 969 |
| exact (obs $=T_{\rm pred}$) | 300 | 449 | 580 |
| wrong step | 1 | 1 | 37 |
| false alarm (pred flip, obs none) | 15 | 14 | 54 |
| miss (pred none, obs flip) | 20 | 3 | 26 |
| both none | 664 | 533 | 303 |
| classification accuracy | 96.4% | 98.2% | 88.3% |
| signed false-alarm rate | 4.75% | 3.02% | 8.05% |
| observed mean / median flip step | 20.9 / 21 | 20.7 / 21 | 29.2 / 21 |
| flips at $t=19$ / $t=21$ / $t=189$ / $t=273$ | 11 / 310 / 0 / 0 | 74 / 379 / 0 / 0 | 267 / 342 / 33 / 1 |
| post-flip agreement, population | 0.0969 | 0.0972 | 0.0962 |
| agreement after $t>{\rm obs}+45$ | 0.0172 | 0.0171 | 0.0171 |

The signed rule predicts the flip fraction to 0.5 percentage points at $\rho=0.05$ and 1.1 points at $\rho=0.1$; at three times the calibration radius it over-predicts by 2.8 points. The apparent rise of the mean flip step at $\rho=0.3$ is not a slowdown of the two main gates: the median is still 21, and the mean is pulled up by 34 late flips (33 at $t=189$, one at $t=273$).

---

## 2. Figure `ds14free_scorecard.png` — the signed rule is nearly exact at and below the calibrated radius

![scorecard](figures/ds14free_scorecard.png)

**Setup.** All 1000 directions at each of the three radii. Each direction is assigned to exactly one class: **exact** if ${\rm obs}=T_{\rm pred}\ge0$; **wrong step** if both are $\ge0$ but differ; **false alarm** if $T_{\rm pred}\ge0$ and ${\rm obs}=-1$; **miss** if $T_{\rm pred}=-1$ and ${\rm obs}\ge0$; **both none** if $T_{\rm pred}={\rm obs}=-1$. The prediction uses the signed threshold $r_c^-$ recomputed from `dm`, and $T_{\rm pred}$ is the first crossing in time.

**Axes.** $x$: free-run radius $\rho$; $y$: number of directions, linear 0–1000. Bars are stacked in the order exact, wrong step, false alarm, miss, both none. Above each bar: observed and predicted flip counts. Counts below 30 are written as a callout to the right of the bar rather than inside a sliver.

**Numbers.**

| class | $\rho=0.05$ | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|---:|
| exact | 300 | 449 | 580 |
| wrong step | 1 | 1 | 37 |
| false alarm | 15 | 14 | 54 |
| miss | 20 | 3 | 26 |
| both none | 664 | 533 | 303 |
| exact / predicted flips | 94.9% | 96.8% | 86.4% |
| exact / observed flips | 93.5% | 99.1% | 90.2% |

**What is in the figure.**
- At $\rho=0.1$ — the radius at which $\delta m$ was measured — the scorecard is almost diagonal: 449 exact, 1 wrong step, 14 false alarms, 3 misses. The rule gets the **event** right for 982/1000 directions and the **step** right for 449/453 observed flips.
- At $\rho=0.05$, below the calibration radius, the pattern is the same with a slightly larger miss count: 300 exact, 1 wrong step, 15 false alarms, 20 misses. The misses are not exotic third-token flips. Their smallest threshold radius lies in $[0.0509,0.147]$ with median 0.0567 (13% above $\rho$) and 90th percentile 0.0738 (48% above); only the extreme tail reaches 2–3x $\rho$. The false alarms have smallest threshold in $[0.0405,0.0492]$, just below $\rho$. The flip/no-flip boundary is sharp, and the misses are a small threshold-calibration error, not a qualitative failure.
- At $\rho=0.3$, three times the radius at which the response was measured, linear extrapolation degrades: 580 exact, 37 wrong step, 54 false alarms, 26 misses. The false alarms again sit just below threshold (smallest $r_c$ median 0.159, maximum 0.297), while the misses reach as high as $r_c=13.2$: at this radius some directions flip without their top-1/top-2 gap ever being predicted to close, which is the signature of a genuinely nonlinear response or of a third token overtaking.
- The sign-blind rule is not a viable alternative at any radius: 599/810/969 predicted flips and 49.7/44.4/34.4% false-alarm rates, against 4.8/3.0/8.1% for the signed rule (numbers in the JSON and §1.2).

**What it shows.** The signed teacher-forced rule is a calibrated predictor of whether a random prompt perturbation destroys the greedy continuation, and at the measurement radius it also predicts the exact step for 99% of the flips. The degradation at $\rho=0.3$ is confined to the step/threshold assignments; even there, 88.3% of directions are classified correctly.

**What it does not show.** It does not validate the rule at radii above 0.3 or below 0.05, nor for a different prompt or a different model. It does not show that a "miss" is a top-1/top-2 flip at all: the free run records only tokens, not the logits after divergence, so the missing third-token hypothesis cannot be tested here. And because the prediction and the free run share the same `dm` measurement, the agreement is not an independent estimate of the linear response — it is a test of whether that response extrapolates to a different radius and to free generation.

---

## 3. Figure `ds14free_survival.png` — survival curves agree to 0.035

![survival](figures/ds14free_survival.png)

**Setup.** $S(t)=P(\text{first flip}>t)$, with a never-flipped direction treated as first flip $=+\infty$. Observed: from `obs`; predicted: from the signed $T_{\rm pred}(\rho,u)$. Both are step functions on $t=0,\dots,299$, one panel per radius. Vertical dotted lines mark $t=19$ and $t=21$.

**Axes.** $x$: generated step $t$, linear 0–300; $y$: survival probability, linear 0–1. Black solid: observed; red dashed: predicted signed.

**Numbers.**

| | $\rho=0.05$ | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|---:|
| $S(19)$ observed / predicted | 0.989 / 0.988 | 0.926 / 0.925 | 0.733 / 0.720 |
| $S(21)$ observed / predicted | 0.679 / 0.684 | 0.547 / 0.536 | 0.391 / 0.356 |
| $S(189)$ observed / predicted | 0.679 / 0.684 | 0.547 / 0.536 | 0.358 / 0.329 |
| max $|S_{\rm obs}-S_{\rm pred}|$ (step) | 0.005 ($t=21$) | 0.011 ($t=21$) | 0.035 ($t=21$) |

**What is in the figure.**
- Up to $t=19$ both curves are at 1.0: no direction flips before the first near-tie. The first drop is a single token at $t=19$, then the much larger drop at $t=21$: at $\rho=0.1$, 7.4% of directions are already gone at 19 and another 37.9% at 21, leaving 54.7% intact.
- The predicted curve is slightly below the observed one at $\rho=0.3$ after $t=21$ (0.356 vs 0.391 at 21, 0.329 vs 0.358 at 189), which is the same 26 misses seen in the scorecard (the prediction flips more directions than are observed). At $\rho=0.05$ and $0.1$ the two curves are within 0.011 everywhere.
- The late shoulder at $t=189$ exists only at $\rho=0.3$: 3.3% of directions flip there and one at $t=273$. The prediction puts 2.7% at 189 and none at 273.

**What it shows.** The entire distribution of first-flip times is reproduced, not just its mean: the two near-ties at 19 and 21, the absence of any earlier event, and the small late tail at the largest radius. A rule calibrated from teacher-forced gaps at $r=0.1$ predicts the survival curve of free-running generation.

**What it does not show.** Survival only records the first event; it says nothing about what happens after a flip. A curve that matches at $t\le21$ is compatible with completely different texts afterwards, which is exactly what §5 finds. It also does not resolve the mechanism of the 26 misses at $\rho=0.3$: they could be nonlinear threshold shifts or third-token flips.

---

## 4. Figure `ds14free_latency.png` — the flips are immediate, and the step is predicted

![latency](figures/ds14free_latency.png)

**Setup.** Top row: histogram of observed first-flip steps, using only the steps that occur (19, 21, 189, 273); the never-flipped directions are annotated, not plotted on the log axis. Bottom row: bubble matrix of predicted step $T_{\rm pred}$ against observed step, all 1000 directions, including $-1$ ("never"). Bubble area scales as $\sqrt{n}$ and colour as $\log_{10} n$; labels give the count. The diagonal is the perfect prediction.

**Axes.** Top: $x$: observed step (categorical, only observed values); $y$: directions, log. The shaded band marks the two razor-thin ties. Bottom: $x$: predicted step; $y$: observed step; both categorical with ticks $\{-1,19,21,189,273\}$.

**Numbers.**

| | $\rho=0.05$ | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|---:|
| immediate, ${\rm obs}=t_*$ | 315 | 422 | 496 |
| earlier than $t_*$, ${\rm obs}<t_*$ | 6 | 31 | 132 |
| delayed, ${\rm obs}>t_*$ | 0 | 0 | 15 |
| ${\rm obs}-T_{\rm pred}=0$ | 300 | 449 | 580 |
| ${\rm obs}-T_{\rm pred}=\pm2$ | 1 | 1 | 24 |
| larger step errors | 0 | 0 | 13 |
| mean / median observed step | 20.9 / 21 | 20.7 / 21 | 29.2 / 21 |

**What is in the figure.**
- At $\rho=0.05$ and $0.1$ every flip is at 19 or 21; there are no delayed flips at all. The six (resp. 31) "early" runs are not early relative to the first crossing: they are directions whose global minimum is at 21 but whose $r_c(19)$ is already below $\rho$, so $T_{\rm pred}=19$. They are counted as exact in the scorecard.
- At $\rho=0.3$, 119 of the 132 early runs are the same $\pm2$ effect (19 vs 21). The remaining 13 jump between the 21/19 pair and 189/273: 9 flip at 21 although the global minimum is at 189 ($\Delta=-168$), 3 at 21 with $\Delta=-170$, one at 19 with $\Delta=-252$. The 15 delayed runs are the mirror image: 7 flip at 189 although the global minimum is at 21 ($\Delta=+168$), one at 189 with $\Delta=+170$, 5 at 21 with $\Delta=+2$, and one each with $\Delta=+11$ and $\Delta=+21$.
- The mean 29.2 at $\rho=0.3$ is entirely composition: $(267\cdot19+342\cdot21+33\cdot189+1\cdot273)/643=29.18$, while the median stays 21. The late flips are 5.3% of the outcomes but add $[33(189-21)+1(273-21)]/643=9.0$ points to the mean.
- The bubble matrix shows that the 580 exact directions are the two large diagonal bubbles at 21 (310) and 19 (248), plus 22 at 189. Off-diagonal mass is concentrated in the 19↔21 block and in the 21↔189 block; there is no diffuse scatter, because only four steps ever occur.

**What it shows.** Latency is not a free parameter: the rule predicts not only that a direction flips but at which of the two ties, and the "delayed" phenomenon is essentially absent at and below the calibration radius. At $\rho=0.3$ the rule still identifies the right pair of gates; the residual error is which member of the pair, or whether the 189 gate fires first.

**What it does not show.** It does not show that a flip at 19 is a different physical event from a flip at 21: both lead to the same alternative continuation (§6). The latency classes use $t_*=\arg\min r_c$, so the negative entries are a definitional artefact of first-crossing versus argmin, not evidence of a failure; measured against $T_{\rm pred}$ the same runs are exact. And the late 189/273 events are observed only at one radius, so their rate is not a calibrated function of $\rho$.

---

## 5. Figure `ds14free_resync.png` — the 73% "post-flip agreement" is an average over non-flipped runs

![resync](figures/ds14free_resync.png)

**Setup.** The stored `summary.json` field `mean_postflip_agreement` is, despite its name, the mean of `agreement[ri,u]` over **all** 1000 directions and over the **whole** 300-token run. The free-run writer computes `agreement` as the fraction of the 300 positions where the perturbed generation equals its in-batch $r=0$ reference, so a direction with `obs=-1` contributes exactly 1.0. The figure decomposes that mean. Top row: ECDF of the directly measured $A_{\rm after}(u)$ for the 11 saved directions, separately for flips at 19 and 21; vertical lines mark the all-direction inference, the 1.5B literature value quoted in the brief (2.7%), and $A_{\rm late}$ ($t>{\rm obs}+45$). Bottom row: the post-flip token positions split into matches inside runs of length $\ge5$, isolated matches, and mismatches. The direct measurement uses 19 flipped rows; the population value is recovered algebraically below.

**Axes.** Top: $x$: $A_{\rm after}(u)$, linear 0–1; $y$: ECDF over flipped directions. Bottom: $x$: fraction of post-flip token positions; $y$: none (single stacked bar).

**The population algebra.** Let $f$ be the observed flip fraction, $\bar A$ the stored whole-run mean over all directions, and $\bar o$ the mean first-flip step among flipped directions. Non-flipped directions have $A=1$ exactly (`obs=-1` means no mismatch anywhere). Hence the mean whole-run agreement among flipped directions is
$$\bar A_{\rm flip}=\frac{\bar A-(1-f)}{f},$$
and, because the prefix $t<{\rm obs}$ matches and the position $t={\rm obs}$ does not, the mean agreement strictly after the flip is
$$\bar A_{\rm after}=\frac{\bar A_{\rm flip}T-\bar o}{T-\bar o}.$$
This uses only stored quantities and is exact under the definition of `obs`; it reproduces the directly measured sample to the third decimal.

| | $\rho=0.05$ | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|---:|
| stored `mean_postflip_agreement` (all dirs, whole run) | 0.7303 | 0.6192 | 0.4754 |
| implied mean whole-run, **among flipped** | 0.1599 | 0.1595 | 0.1841 |
| implied mean **after** the first flip | 0.0969 | 0.0972 | 0.0962 |
| directly measured $A_{\rm after}$, saved rows | 0.0971 | 0.0979 | 0.0986 |
| directly measured $A_{\rm late}$ ($t>{\rm obs}+45$) | 0.0172 | 0.0171 | 0.0171 |
| re-synchronised runs (sustained suffix $\ge20$) | 0 / 4 | 0 / 7 | 0 / 8 |

**What is in the figure.**
- The headline 73.0/61.9/47.5% is a mixture, not a post-flip measurement. It falls with $\rho$ only because the flip fraction rises: 0.7303 = 0.679·1.0 + 0.321·0.1599, and similarly for the other radii. The agreement among flipped directions is remarkably constant, 0.159–0.184, and the strictly post-flip value is constant at 0.096–0.097.
- The small rise of the whole-run value among flipped directions at $\rho=0.3$ (0.184 vs 0.160) is not better agreement: 34 of the 643 flips happen at 189/273, so those runs have less time to diverge. The post-flip value stays at 0.096.
- The direct token analysis shows what the matching tokens are. Of the 108 matching post-flip positions across the 4 saved $\rho=0.05$ flips, 92 (85%) lie in just two contiguous runs: `", so I need to"` (5 tokens immediately after the gate) and an 18-token run at $t=28$–$45$, `" a comprehensive technical review of current and proposed propulsion systems for interstellar travel. The user has"`. The second is a verbatim restatement of the prompt sentence. Pooled over all 19 flipped rows, 437 of the 519 matching tokens (84.2%) are inside runs of length $\ge5$; outside those runs only 82 of the remaining 4,857 positions match (1.7%).
- After $t={\rm obs}+45$ the agreement is 1.7%, and it never returns. The longest identical run anywhere after a flip is 18 tokens (the prompt echo); the identical-suffix length is 0 for every flipped row; the last token differs in every case. Under the stated definition — 20 consecutive identical tokens, or an identical suffix of 20 — the re-synchronisation count is 0/19.
- The flip-19 and flip-21 variants differ at exactly one token (the $t=19$ gate), agreement 0.9967. All rows that flip at the same gate are **bit-identical** over all 300 tokens. Across the 33 saved rows (11 directions × 3 radii) there are only **three** distinct continuations: the baseline, the alternative entered at $t=21$, and the alternative entered at $t=19$ (one token different from the second).

**What it shows.** The apparent 73% is not a cross-scale anomaly; it is a mixture artefact. Once the mixture is removed, the 14B behaves like the 1.5B literature value: token agreement after a flip is ~9.7% if the shared prompt restatement is counted, and ~1.7% if it is not, against 2.7% quoted for the 1.5B with no re-synchronisation. The correct cross-scale statement is therefore *no major difference*, not a 27-fold one. What does differ is the **ensemble** behaviour: perturbed runs re-converge to one another almost perfectly (pairwise agreement 0.9967–1.0), because they all enter the same alternative basin, while none re-converges to the baseline.

**What it does not show.** The direct ECDF rests on 19 flipped rows, all from the saved final chunk (directions 989–999); the population curve is inferred, not measured token-by-token. It cannot separate "synonym-like gate token" from "same semantic plan" as causes: `"Alright"`→`"Okay"` is a discourse-marker substitution, and `" reference"`→`" research"` is a plausible noun substitution, but the downstream essays are different outlines, not paraphrases of the same sentences. Token agreement is not semantic agreement — the two texts are both on-topic reviews, so a human would call them similar even though 90% of tokens differ. And the 2.7% 1.5B number is quoted from the brief, not recomputed on the same protocol here; the comparison is order-of-magnitude, not a controlled cross-model contrast.

---

## 6. Figure `ds14free_flipword.png` — what the two competing tokens are

![flipword](figures/ds14free_flipword.png)

**Setup.** The unperturbed A1 top-1 and top-2 tokens at the two gates, decoded with `assets/tok_ds14/tokenizer.json` (the `tokenizers` library), with their token ids and the logit margin $m_t$. Below each gate: the number of directions whose **first** mismatch is at that step, per radius, and which token the 11-direction saved sample took.

**Axes.** None (text cards); the numbers are counts of directions.

**Numbers.**

| gate | top-1 (baseline) | top-2 (competitor) | $m_t$ (nats) | observed first flips at the gate, $\rho=0.05/0.1/0.3$ | saved flips taking top-2 |
|---|---|---|---:|---|---:|
| $t=19$ | `" reference"` (id 5785) | `" research"` (id 3412) | $1.144\times10^{-3}$ | 11 / 74 / 267 | 6 / 6 |
| $t=21$ | `"Alright"` (id 71486) | `"Okay"` (id 32313) | $9.727\times10^{-5}$ | 310 / 379 / 342 | 13 / 13 |

(The saved-flips column is six of six and thirteen of thirteen across all radii; no saved flip took top-1.)

**What is in the figure.**
- The two gates are ordinary words in an ordinary sentence. The baseline reads `"...or professional reference.\n\nAlright, so I need to provide a comprehensive technical review..."`. A $t=19$ flip gives `"...or professional research.\n\nOkay, so I need to write a comprehensive technical review..."`; a $t=21$ flip gives `"...professional reference.\n\nOkay, so I need to write..."`. Both alternatives then continue with the same essay, which is why the flip-19 and flip-21 texts differ by one token.
- $t=21$ is the thinner tie by a factor 11.8 ($9.7\times10^{-5}$ vs $1.14\times10^{-3}$ nats), and it is the dominant gate at the two smaller radii (310/321 flips at $\rho=0.05$, 379/453 at $\rho=0.1$). At $\rho=0.3$ the 19 gate gains ground (267 vs 342, still behind 21 but far closer than the 74 vs 379 at $\rho=0.1$).
- Both gate words are near-synonyms in context (`Alright`/`Okay`; `reference`/`research` after "professional"), so the first few tokens look almost unchanged: the first 5 post-flip tokens match 100% (flip at 21) or 80% (flip at 19), the first 10 match 90%/80%, the first 20 match 95%/90%. That local agreement is the synonym-like part (these window fractions are for the representative saved rows, dir 989 at $\rho=0.05$ and dir 995 at $\rho=0.1$). It does not persist: the first 50 match only 46%/48%, and after $t={\rm obs}+45$ only 1.7% of the remaining 254 positions match.
- At $\rho=0.3$ two further gates appear: 33 flips at $t=189$ (`" by"` vs `" about"`, $m=2.22\times10^{-3}$ nats) and one at $t=273$ (`","` vs `" today"`, $m=1.92\times10^{-2}$ nats). They are 5.3% of the flips and do not change the two-gate picture at the calibrated radius.

**What it shows.** The answer to "what are the two competing tokens": at $t=21$, `"Alright"` (baseline, id 71486) and `"Okay"` (id 32313, taken by all 13 saved runs whose first flip is at 21); at $t=19$, `" reference"` (baseline, id 5785) and `" research"` (id 3412, taken by all 6 saved runs whose first flip is at 19). Both are single-token, locally interchangeable substitutions at the two smallest logit gaps in the run. They are not random vocabulary: they are the two best candidates of a nearly tied decision.

**What it does not show.** The token identity is directly verified only for the 19 saved flips; for the other 1000-direction runs the evidence is that their first mismatch step is 19 or 21 and that the only tracked competitor at those steps is top-2. A third token overtaking top-1 without top-2 winning cannot be excluded from `obs` alone, but it is not observed in any saved row. And a one-token substitution at the gate does not imply a one-token change in meaning: §5 shows the essays diverge from token 24 onward.

---

## 7. Facts that qualify the whole validation

1. **Full attention, not window 100.** The 14B A1 run and this free run used full attention because the window setting was silently inactive (transformers gates it on a `layer_types` field that the stock config does not set). Non-flipped saved rows are bit-identical to the full-attention A1 `gen_ids`, so the validation is internally consistent, but it is not a window-100 validation. A true-window rerun is being generated; the window can change the margin spectrum, so the $t=19/21$ gate structure may move in that run.
2. **One prompt, one model, 300 tokens.** The gates are properties of this continuation. A different prompt, or a longer horizon, can expose other near-ties; the late 189/273 gates at $\rho=0.3$ already show that the two-gate picture is radius-dependent.
3. **`gen_ids` contains only the final chunk.** The writer overwrote its dictionary per chunk, so raw tokens exist for directions 989–999 only. The mapping was verified against `obs`; the direct post-flip ECDF is 19 rows, and the population post-flip value is an algebraic inference from the stored whole-run means, not a token-level census.
4. **The summary field is misnamed.** `mean_postflip_agreement` is the whole-run agreement averaged over all directions, with non-flipped directions at exactly 1.0. It must not be quoted as "agreement after the flip". The corrected values are 0.0969/0.0972/0.0962 (post-flip) and 0.0172/0.0171/0.0171 (after the prompt echo).
5. **Radii 0.05 and 0.3 are extrapolations.** The response was measured at $r=0.1$ only. The rule is nearly exact at 0.05 and 0.1 and degrades at 0.3 (88.3% classification, 37 wrong steps, 13 large step jumps). The degradation is a statement about linear extrapolation, not about the direction of the effect.
6. **A flip is a token mismatch, not a semantic failure.** All continuations here remain coherent, on-topic technical reviews. "32% of directions flip" means 32% produce a different token stream, not that 32% produce a wrong or degraded answer.
7. **Latency uses $t_*=\arg\min r_c$, not first crossing.** Negative latencies at 19 vs 21 are definitional; measured against $T_{\rm pred}$ they are exact. Do not read $\Delta<0$ as a rule failure.
8. **No post-flip logits.** Once the text diverges, the teacher-forced `dm` no longer describes the state. The analysis can say where and when a run left the baseline and how far it went, but it cannot compute a causal threshold radius for the new text.

---

## 8. What follows directly

- The signed rule, not the sign-blind one, is the correct low-order description: it predicts the flip fraction to 0.5–1.1 percentage points at $\rho\le0.1$ and classifies 96.4–98.2% of directions correctly, while the $|\delta m|$ rule predicts 599/810/969 flips and fails on 49.7/44.4/34.4% of them.
- The first failure of a free-running 14B continuation is a **discrete gate choice**, not a gradual degradation: 94.7% of all flips at $\rho=0.3$, and 100% at the smaller radii, occur at just two tokens, `"Alright"`→`"Okay"` and `" reference"`→`" research"`.
- The correct post-flip agreement is ~9.7% (or ~1.7% after removing the verbatim prompt restatement), not 73/62/48%. This is the same order as the 2.7% quoted for the 1.5B; there is no large cross-scale difference in post-flip re-synchronisation.
- What *is* different from a naive "chaos" picture is the ensemble: perturbed runs do not fan out into many texts. They collapse into one alternative continuation, bit-identically, and that continuation is not the baseline. The sensitivity is a switch between a small number of discrete basins.
- The remaining open measurement is the true-window 14B free run. If the window moves the $t=19/21$ gates or changes the post-flip basin structure, the numbers in §1.3 and §5 must be recomputed on that run; the method (signed $r_c$, first-crossing $T_{\rm pred}$, and the $A_{\rm after}$ algebra) carries over unchanged.

## 9. Reproduce

```
python scripts_spark/figs_free_full.py
```

Writes `notebook_big/figures/ds14free_{scorecard,survival,latency,resync,flipword}.png` (dpi 170) and `notebook_big/figures/ds14free_numbers.json`, which contains the full scorecard, the sign-blind comparison, the latency distributions, the survival curves, the per-run post-flip records (including the decoded longest shared run), the population inference, and the decoded gate tokens with their ids. The script reads the two npz files and the summary in place; it does not modify them. The tokenizer is `assets/tok_ds14/tokenizer.json` via the `tokenizers` package.
