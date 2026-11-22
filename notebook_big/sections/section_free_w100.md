# Addendum. Free-run validation under the true window 100 (14B)

## Summary

Under a verified sliding window of 100, the free run at $\rho=0.1$ flips **483/1000** directions (48.3%) against **493** predicted by the signed rule (49.3%): 478 exact steps, 2 wrong steps, 13 false alarms, 3 misses, 98.2% classified correctly. At $\rho=0.3$ it flips 703/1000 (70.3%) against 723 predicted (72.3%), with 625 exact, 41 wrong steps, 57 false alarms, 37 misses (86.5%). The two early ties are **unchanged by the window**: the margins at $t=19$ and $t=21$ are bit-identical to the full-attention run (the prompt is 65 tokens, so the window cannot drop a key before generated step $t=36$), and the flip counts at 19/21 are literally the same numbers, 74/379 at $\rho=0.1$ and 267/342 at $\rho=0.3$. The window adds one new razor tie at $t=297$ — `" These"` vs `" Each"`, $m=4.8\times10^{-3}$ nats — which is the *entire* excess at $\rho=0.1$ (483 = 453 + 30); at $\rho=0.3$ it reorganises the late gates (271 and 297 replace the full-attention 189 and 273). With `gen_ids` now complete for all 1000 directions at both radii, the post-flip analysis is a census of all 1186 flipped runs: every one took top-2 at its gate, post-flip agreement for the long-horizon flips is 10.5%/10.1% (2.6%/2.0% after the prompt echo), and **0/1186 re-synchronise** to the baseline. The longest shared run anywhere is 17 tokens, the verbatim prompt echo.

---

## 1. Run, data, quantities

| item | value |
|---|---|
| model | `DeepSeek-R1-Distill-Qwen-14B`, FP32 |
| attention | **true sliding window 100** (actuation verified; no key is dropped for $t\le35$ by construction) |
| prompt | `interstellar_travel`, $P=65$ tokens, $T=300$ generated tokens |
| A1 calibration | 1000 directions at $r=0.1$, teacher-forced on the baseline |
| free run | same 1000 directions and perturbation, $\rho\in\{0.1,0.3\}$, greedy, own KV cache |
| files | `results_big/ds14/a1_inter_w100/a1.npz`, `results_big/ds14/free_inter_w100/{free_run.npz,summary.json}` |

**Data integrity.** This run fixes the accumulation bug: `gen_ids` is `2 x 1000 x 300`. All 1000 rows at each radius reproduce `obs` exactly (`first_mismatch == obs`), the 517 (resp. 297) non-flipped rows are bit-identical to the A1 `gen_ids`, and the stored whole-run `agreement` is recovered from the raw rows to $3\times10^{-8}$. The A1 baseline is self-consistent (`top1[t] == gen_ids[t]` for all $t$).

Notation as in the full-attention section: $m_t$ is the top1–top2 gap of the unperturbed A1 run, $\delta m_t(u)$ its change at $r=0.1$, $r_c^-(t,u)=m_t r/\max(-\delta m_t(u),0)$, $T_{\rm pred}(\rho,u)=\min\{t:\rho\ge r_c^-(t,u)\}$, and ${\rm obs}$ is the first generated token that differs from the baseline. The npz's `r_c` is the sign-blind $m_t r/|\delta m_t|$; every prediction below uses the signed $r_c^-$. $A_{\rm after}$ is the token agreement strictly after ${\rm obs}$, $A_{\rm late}$ the same for $t>{\rm obs}+45$.

| quantity | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|
| observed flips / 1000 | 483 | 703 |
| predicted flips, signed | 493 | 723 |
| predicted flips, sign-blind | 834 | 976 |
| exact (${\rm obs}=T_{\rm pred}$) | 478 | 625 |
| wrong step | 2 | 41 |
| false alarm | 13 | 57 |
| miss | 3 | 37 |
| both none | 504 | 240 |
| classification accuracy | 98.2% | 86.5% |
| signed false-alarm rate | 2.64% | 7.88% |
| observed mean / median flip step | 37.8 / 21 | 56.9 / 21 |
| flips at $t=19/21/271/297$ | 74 / 379 / 0 / 30 | 267 / 342 / 6 / 88 |
| $A_{\rm after}$, flips at 19/21 | 0.105 | 0.101 |
| $A_{\rm late}$ ($t>{\rm obs}+45$) | 0.026 | 0.020 |

The signed rule tracks the flip fraction to 1.0 point at $\rho=0.1$ and 2.0 points at $\rho=0.3$. The sign-blind rule predicts 834/976 flips with 42.4%/28.7% false alarms and is rejected as before.

---

## 2. Figure `ds14w100free_scorecard.png`

![scorecard](figures/ds14w100free_scorecard.png)

All 1000 directions at each radius, one class per direction (exact, wrong step, false alarm, miss, both none).

| class | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|
| exact | 478 | 625 |
| wrong step | 2 | 41 |
| false alarm | 13 | 57 |
| miss | 3 | 37 |
| both none | 504 | 240 |
| exact / predicted flips | 97.0% | 86.4% |
| exact / observed flips | 99.0% | 88.9% |

- At $\rho=0.1$ the rule gets the **event** right for 982/1000 directions and the **step** right for 478/483 flips. The two wrong steps are one flip 2 steps late and one 276 steps late (predicted 21, flipped at 297).
- At $\rho=0.3$ the residual is concentrated in the late gates: of the 41 wrong steps, 24 are 19↔21 swaps and 17 involve 271/297 (10 predicted 21→flipped 297, 3 predicted 19→297, 2 predicted 271→297, 1 predicted 297→271, 1 predicted 297→21).
- The 13 false alarms at $\rho=0.1$ have smallest threshold radius $r_*\in[0.073,0.0995]$, all just below $\rho$: threshold-calibration errors, not a qualitative failure. The signed false-alarm rate is 2.6% against 42.4% for the sign-blind rule.

---

## 3. Figure `ds14w100free_survival.png`

![survival](figures/ds14w100free_survival.png)

$S(t)=P(\text{first flip}>t)$, never-flipped treated as $+\infty$; observed vs signed prediction, one panel per radius.

| | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|
| $S(19)$ obs / pred | 0.926 / 0.925 | 0.733 / 0.720 |
| $S(21)$ obs / pred | 0.547 / 0.536 | 0.391 / 0.356 |
| $S(271)$ obs / pred | 0.547 / 0.536 | 0.385 / 0.349 |
| $S(297)$ obs / pred | 0.517 / 0.507 | 0.297 / 0.277 |
| max $|S_{\rm obs}-S_{\rm pred}|$ | 0.011 ($t=21$) | 0.036 ($t=21$) |

The distribution of first-flip times is reproduced, including the new drop at $t=297$. At $\rho=0.1$ there is no event between 21 and 297: the curve is flat at 0.547 for 276 steps, then loses the 30 directions that flip at 297. At $\rho=0.3$ the predicted curve is consistently 0.013–0.036 below the observed one after $t=19$, the same 37 misses seen in the scorecard.

---

## 4. Figure `ds14w100free_latency.png`

![latency](figures/ds14w100free_latency.png)

Top: histogram of observed first-flip steps. Bottom: predicted vs observed step, all 1000 directions.

| | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|
| immediate (${\rm obs}=t_*$) | 448 | 527 |
| earlier than $t_*$ | 35 | 164 |
| delayed (${\rm obs}>t_*$) | 0 | 12 |
| ${\rm obs}-T_{\rm pred}=0$ | 478 | 625 |
| $\pm2$ | 1 | 24 |
| larger step error | 1 | 17 |

- At $\rho=0.1$ every flip is at 19, 21 or 297; the 35 "early" runs are the usual first-crossing effect (global minimum at 21, but $r_c(19)$ already below $\rho$) and are counted as exact against $T_{\rm pred}$.
- The mean 37.8 at $\rho=0.1$ is pure composition: $(74\cdot19+379\cdot21+30\cdot297)/483$, while the median stays 21. At $\rho=0.3$ the 94 late flips (6 at 271, 88 at 297) add $[6(271-21)+88(297-21)]/703=36.7$ points to the mean; the median is still 21.

---

## 5. Figure `ds14w100free_resync.png` — full census

![resync](figures/ds14w100free_resync.png)

With every row decoded, this is no longer an inference. Over the 1186 flipped rows:

| | $\rho=0.1$ | $\rho=0.3$ |
|---|---:|---:|
| flipped rows | 483 | 703 |
| $A_{\rm after}$, flips at 19/21 | 0.1054 | 0.1010 |
| $A_{\rm after}$, flips at 19 only | 0.0925 | 0.0921 |
| $A_{\rm after}$, flips at 21 only | 0.1079 | 0.1079 |
| $A_{\rm late}$ ($t>{\rm obs}+45$) | 0.0265 | 0.0202 |
| matching positions inside runs $\ge5$ | 75.0% | 78.1% |
| re-synchronised (run $\ge20$ or suffix $\ge20$) | 0 / 483 | 0 / 703 |
| longest identical run | 17 | 17 |
| distinct continuations (flipped) | 4 | 9 |
| pairwise agreement, flipped (sample) | 0.773 | 0.493 |

- **No re-synchronisation.** The longest shared run anywhere is 17 tokens, $t=28$–$44$, `" a comprehensive technical review of current and proposed propulsion systems for interstellar travel. The user"`, a verbatim echo of the prompt; the identical-suffix length is 0 for every flipped row. Under the stated definition the count is 0/1186.
- **The ensemble collapse is partial.** At $\rho=0.1$ there are 4 flipped continuations: the 21 gate has exactly **one** bit-identical 300-token text (379 rows), the 297 gate one (30 rows), but the 19 gate splits into two downstream branches (67 and 7 rows). At $\rho=0.3$ the 21 and 297 gates are still single texts, while the 19 gate splits into five (193/52/14/6/2) and the 271 gate into two (4/2).
- **The stored field is still misnamed.** `mean_postflip_agreement` is 0.6222/0.4876, a mean over all directions and the whole run. Removing the non-flipped directions ($A=1$) and the matched prefix gives 0.1049/0.1003, which matches the measured 0.1054/0.1010.
- The 30 (resp. 88) flips at 297 leave only 3 (resp. 3) tokens, so their $A_{\rm after}=0$ is arithmetic, not evidence; all re-synchronisation statements use the 19/21 flips. The gate-level figures are stable across the two radii (21: 0.1079 at both; 19: 0.092), and the 21-gate alternative is a single text shared by both radii, consistent with the early computation being identical.

---

## 6. Figure `ds14w100free_flipword.png` — the three ties

![flipword](figures/ds14w100free_flipword.png)

| gate | top-1 (baseline) | top-2 (competitor) | $m_t$ (nats) | observed first flips $\rho=0.1/0.3$ |
|---|---|---|---:|---:|
| $t=19$ | `" reference"` (5785) | `" research"` (3412) | $1.144\times10^{-3}$ | 74 / 267 |
| $t=21$ | `"Alright"` (71486) | `"Okay"` (32313) | $9.727\times10^{-5}$ | 379 / 342 |
| $t=297$ | `" These"` (4220) | `" Each"` (8886) | $4.791\times10^{-3}$ | 30 / 88 |
| *($t=271$)* | *`" key"` (1376)* | *`" target"` (2169)* | *$5.221\times10^{-2}$* | *0 / 6* |

**What the new tie competes between.** The baseline text is

> `"... additional elements like appendices or indexes.. These sections are"`

and a $t=297$ flip reads

> `"... additional elements like appendices or indexes.. Each of these"`

So `" These"` $\to$ `" Each"` is a sentence-initial determiner swap between two grammatical continuations; only three tokens ($t=298,299$) follow it inside the 300-token horizon, so its downstream text cannot be compared here. The gate is the third-smallest gap in the window run (after 21 and 19) and controls 29 directions at $\rho=0.1$: 174 directions have $t_*=297$ and 34 have $r_*\le0.1$, but for 29 of them 297 is the first crossing, which is what $T_{\rm pred}$ reports. All 29 predicted crossings are exact; one further direction flips at 297 while predicted at 21.

All 1186 flipped rows took **top-2** at their gate; none took top-1 or a third token. The 271 gate is not razor-thin ($5.2\times10^{-2}$ nats) and only fires at $\rho=0.3$ (6 rows, two downstream branches).

---

## 7. Full attention vs true window

| | full $0.1$ | **w100 $0.1$** | full $0.3$ | **w100 $0.3$** |
|---|---:|---:|---:|---:|
| observed flips / 1000 | 453 | **483** | 643 | **703** |
| predicted flips (signed) | 464 | **493** | 671 | **723** |
| exact | 449 | **478** | 580 | **625** |
| wrong step | 1 | **2** | 37 | **41** |
| false alarm | 14 | **13** | 54 | **57** |
| miss | 3 | **3** | 26 | **37** |
| both none | 533 | **504** | 303 | **240** |
| classification accuracy | 98.2% | **98.2%** | 88.3% | **86.5%** |
| agreement after flip | 0.097 | **0.105** | 0.096 | **0.101** |
| mean flip step | 20.7 | **37.8** | 29.2 | **56.9** |
| flips at 19 / 21 / late gates | 74/379/0 | **74/379/30** | 267/342/34 | **267/342/94** |

- **The early gates do not move.** Because $P=65$, the window keeps all keys for $t\le35$; the first prediction it can change is $t=36$ (the A1 margins are bit-identical through $t=35$ and first differ at 36). Consequently the 19/21 margins and every early flip count are identical.
- **The window changes only the late landscape.** At $\rho=0.1$ it adds the 297 tie, and the extra 30 flips are exactly the 3.0-point increase (483 vs 453). At $\rho=0.3$ it removes the full-attention 189/273 gates (33+1 flips) and installs 271/297 (6+88); the late-gate margins move from $m_{189}=2.2\times10^{-3}$ to $2.85\times10^{-1}$ and from $m_{297}=0.545$ to $4.8\times10^{-3}$.
- **Fragility increases, predictability does not degrade at $\rho=0.1$.** Flip fractions rise by 3.0 points (0.1) and 6.0 points (0.3); classification accuracy is unchanged at 0.1 and falls 1.8 points at 0.3, where the moving late gates are the whole residual.
- **Post-flip behaviour is unchanged.** ~10% token agreement, ~2% after the prompt echo, zero re-synchronisation, and (within this run) a single bit-identical alternative text at each of the 21 and 297 gates.

---

## 8. Facts that qualify this addendum

1. **One prompt, one model, 300 tokens, two radii.** The 297 tie is a property of this continuation; the window-100 late-gate landscape is not universal.
2. **The early-gate identity is structural, not empirical luck.** With $P=65$ and window 100, no key is dropped for $t\le35$; the 19/21 agreement is a check of the harness, not a test of the window.
3. **$\rho=0.3$ is a 3x extrapolation** of a response measured at $r=0.1$; its 37 misses and 41 wrong steps are the linear-rule limit.
4. **297/271 flips have no downstream horizon** (3 and ≤28 tokens); their $A_{\rm after}$ is not comparable and is excluded from the re-synchronisation verdict.
5. **The full-attention $A_{\rm after}$ is inferred** from stored whole-run means and 11 saved rows; the window value is measured on all 1186 rows. The comparison is therefore order-of-magnitude at the third decimal.
6. **A flip is a token mismatch, not a semantic failure.** All continuations remain coherent; the 297 alternative (`"Each of these"`) is locally grammatical.

---

## 9. Reproduce

```
python scripts_spark/figs_free_w100.py
```

Writes `notebook_big/figures/ds14w100free_{scorecard,survival,latency,resync,flipword}.png` and `ds14w100free_numbers.json` (full scorecards, sign-blind comparison, gate table, latency, survival curves, census-level post-flip records, variant counts, and the recomputed full-attention reference). It also asserts the `gen_ids` census, the baseline identity, and the reproduction of the summary fields.
