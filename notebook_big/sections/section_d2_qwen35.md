# Part D2 (Qwen3.5-9B). No scaling region in twelve 1000-token trajectories; one late period-7 partial repeat

## Summary

The last-layer states of `Qwen3.5-9B` (FP32, greedy, no sliding window, 1000 generated tokens, $d=4096$) form a point cloud on which the validated correlation-dimension pipeline finds **no scaling region in any of the 12 trajectories**, at any of the three Theiler windows. Under the fixed rule that was applied to the 1.5B and the DS-14B, the usable range between the depletion floor $C=10/N$ and saturation $C=0.1$ is **0.065-0.082 decades** (0.157 for one run with a late repeat), sampled by **2-3 of the 90 grid points** (6 for that run); the rule requires at least 5 points over at least 0.40 decades. The local slope over that range runs **9.0 -> 18.8** (2.7 -> 9.8 on the repeat-contaminated run) and its maximum anywhere is **14.8-31.7**, against the finite-$N$ ceiling $2\log_{10}N=6.0$. The edge-to-edge drift is 2.6-7.3 units in the 11 clean runs, above the flatness tolerance (0.34-0.94 at these slopes). There is therefore no $\hat D_2$ to quote. On a 900-point grid the chord over the admissible range is **6.64-15.33** (median 14.54); the fixed 90-point grid defines it for only 5 of 12 runs (7.11, 11.69, 13.89, 15.47, 15.62) because it lands 2-3 points inside the band. The old max-$R^2$ estimator returns **12.58-22.36** over windows **0.013-0.066 decades** wide.

The point-cloud estimates disagree with each other more than in either previous model. TwoNN gives **6.05-8.09** (median 7.51), the Levina-Bickel MLE at $k=10$ gives **8.90-12.89** (median 12.00), the chord gives 6.64-15.33, and the T4-style estimator gives 12.58-22.36: within one trajectory the spread reaches a factor **3.1** (T4/TwoNN, `smartphones` 22.36/7.12). Neither TwoNN nor the MLE is converged in $N$: on `great_emu_war` TwoNN falls 6.76 -> 6.68 -> 6.05 as $N$ goes 500 -> 750 -> 1000, and its MLE falls 10.8 -> 10.1 -> 8.9 because a repeated tail enters at $t\approx800$; on `robot_friendship` the same numbers move 7.73 -> 7.84 -> 8.09 and 12.4 -> 12.7 -> 12.9. The correct statement is a range, not a dimension: **the 9B last-layer cloud is locally 6-13 dimensional by the nearest-neighbour estimators and 12-22 by the global chord, depending on method, neighbourhood and $N$, with no estimator stable in $N$.**

**Almost no trajectory repeats.** Eleven of 12 have a lag-dip prominence of **0.981-0.995** (a 1.5B loop is $<0.6$; the 1.5B's full loops were 0.04-0.22) and 0.0% of positions return within $0.1\,d_{typ}$ at the best lag. The exception is `great_emu_war`: prominence **0.849**, a **period-7** dip in the last ~200 states (positions 800-990) with **13.8% coverage**, the only run whose nearest non-adjacent neighbour drops to **0.059 $d_{typ}$** (all others $\ge0.21$). That is a late, local, 7-token repeat — the fixed rule's partial cut is $<0.8$, so the pipeline calls it healthy, but the coverage and nearest-neighbour evidence make it **one partial out of 12**. The count is therefore **0 full + 1 partial / 12**, against 9 full + 2 partial of 20 on the 1.5B (6 full + 2 partial of the 16 runs at the same 1000-token horizon) and 0 full + 0 partial of 13 on the DS-14B. The GPU-side `summary.json` agrees with this reading: median `rel_nearest` 0.68-1.12 with p05 0.11-0.76, and `great_emu_war` is the only run whose p05 falls to 0.113.

Two controls make the interpretation explicit. **N-dependence:** at $N=500/750/1000$ the valid window is still empty at every $N$; the admissible span moves 0.05 -> 0.16 decades, and the `great_emu_war` chord drops 11.95 -> 10.95 -> 6.64 only because the repeated tail enters at $N=1000$, not because of finite-$N$ bias. **Time shuffle:** permuting the distance-matrix rows and columns identically leaves the point cloud unchanged, and at $W=0$ the correlation integral and TwoNN are **bit-identical**; only the Theiler band moves, changing the chord by +3.8 on `interstellar` and **+9.4** on `great_emu_war` at $W=500$, the latter precisely because the exclusion removes the lag-7/56 repeats.

The answer to the architecture question is therefore **no on both counts**. The hybrid Gated-DeltaNet/full-attention stack does **not** produce a valid scaling region where the dense 1.5B and 14B had none, and it does **not** lower the last-layer state-cloud dimension: the 9B's TwoNN (6.05-8.09, median 7.51) lies between the 1.5B's 5.44-6.23 and the 14B's median 8.19, its MLE $k=10$ (8.9-12.9) is below the 14B's 11.0-15.9 and just above the 1.5B's healthy 7.9-10.1, and its global estimates are the highest of the three — the chord median (14.5) is above the 14B's 8.1-10.3 band and the T4 (12.6-22.4) is above both — because its pairwise-distance distribution is the narrowest (CV 0.057-0.079 against 0.110-0.135 for the 14B). Nor does it make the model loop-prone: one late partial repeat in 12 runs is statistically indistinguishable from the 14B's zero and far below the 1.5B's 9/20.

---

## 1. Data, definitions, and the fixed decision rule

### 1.1 The 12 trajectories

| item | value |
|---|---|
| model | `Qwen3.5-9B`, 32 blocks, hidden size 4096 |
| architecture | hybrid: per period of four, three **Gated DeltaNet** (linear-attention) blocks then one full-attention block; full-attention blocks 3, 7, 11, ..., 31 |
| attention window | none (metadata `window=0`), so no clipping inside the 1000-token horizon |
| precision / decoding | FP32, greedy |
| states | last layer (`h_last`), one $[1000,4096]$ cloud per trajectory; `h_last` is the final-norm state (the lm_head input after block 31), not a raw block output: in the 14B files the raw block-47 state has norm ~985 while `h_last` has norm ~170-200, and the same slot is used here. The last block (31) is a full-attention block, so `h_last` is not a Gated-DeltaNet recurrent state |
| input | `results_big/qwen35/d2/dist_<name>_h_last.npy`, fp32 Euclidean $[1000,1000]$, plus `summary.json` |
| horizon | 1000 tokens for all 12 |
| prompts | the 12 lines of `prompts_d2.txt`, one trajectory each (no duplicate topic, unlike the 14B's extra interstellar run) |
| state norms | median 133.5-135.6, late/early norm ratio 0.973-1.003 (no drift or blow-up) |

The distance matrices are float32, symmetric to $3\times10^{-5}$, with diagonal entries of 0.11-0.13 and smallest off-diagonal distance 36.7-50.5 (2.24 for `great_emu_war`, the near-repeat). Every $\epsilon$ in the admissible range is $\ge0.61\,d_{typ}\approx100$, so the diagonal round-off cannot enter any count that matters here.

Only `h_last` was analysed. The collector also stored block outputs 8, 16, 24 and 31 in the trajectory files, but those were not pulled to the laptop and are not part of this section.

### 1.2 Definitions

Let $d_{ij}=\lVert x_i-x_j\rVert$, $N=1000$ points, Theiler window $W$. The definitions are identical to the DS-14B section; the two additions are the fine-grid chord and the distance-distribution CV.

| quantity | expression | note |
|---|---|---|
| typical scale | $d_{typ}=\mathrm{median}_i\left(\frac{1}{N}\sum_j d_{ij}^2\right)^{1/2}$ | median over points of the rms distance to the cloud (512-point subsample, seed 0); 165.9-178.4 here |
| correlation integral | $C(\epsilon)=\dfrac{\#\{(i,j):\ \lvert i-j\rvert>W,\ d_{ij}<\epsilon\}}{\#\{(i,j):\ \lvert i-j\rvert>W\}}$ | denominator $N(N-1)-2\sum_{k\le W}(N-k)$; at $W=0$ exactly $\frac{2}{N(N-1)}\#\{i<j:d_{ij}<\epsilon\}$ |
| grid | 90 log-spaced $\epsilon/d_{typ}\in[10^{-2},2.5]$ | one sort of the upper triangle, then `searchsorted`; $W$-band counts subtracted from the diagonals |
| local slope | $\mathrm{d}\log C/\mathrm{d}\log\epsilon$ | central differences, second order at the edges |
| TwoNN | $\mu_i=r_2^{(i)}/r_1^{(i)}$, $\hat d=1/\langle\ln\mu\rangle$ | Pareto MLE for $P(\mu>M)=M^{-d}$; the CDF-OLS variant gives 6.6-8.5 and the "$\log 2/\langle\log_2\mu\rangle$" variant returns $0.48d$ and is not used |
| Levina-Bickel MLE | $m_k(i)=\frac{1}{k-1}\sum_{j<k}\ln\frac{d_k(i)}{d_j(i)}$, $\hat d_k=\langle 1/m_k\rangle$ | $k=5,10,20$ |
| repetition screen | fraction of points with a nearest neighbour at $\lvert i-j\rvert\ge50$ below $10^{-3}d_{typ}$ | median and p05 of that relative distance are also reported |
| lag-distance loop screen | $p(k)=\overline{\lVert x_t-x_{t+k}\rVert}/d_{typ}$, prominence $=p(k)/\mathrm{median}\,p(k\pm1,2,3)$ | a drifting loop has a sharp dip; coverage is the fraction of positions with $d_{t,t+p}<0.1d_{typ}$ |
| admissible range | $\epsilon\in[\epsilon_{10/N},\epsilon_{0.1}]$, the interpolation points where $C=10/N$ and $C=0.1$ | at least $\sim5N$ pairs inside the ball at the left edge; below saturation at the right |
| chord $D_2$ | OLS slope of $\log C$ on $\log\epsilon$ over the admissible range | quoted on a **900-point** grid; the 90-point pipeline grid lands 2-3 points inside the band and leaves the chord undefined for 7/12 |
| T4-style estimator | the reference `fit_plateau`: maximum $R^2$ over every sub-interval with $\ge6$ grid points, on percentiles 2-60 of a 1000-point subsample, Theiler 20 | reproduces `results/d2/` exactly on the 1.5B; not a plateau test |
| distance CV | $\mathrm{std}(d_{ij})/\mathrm{mean}(d_{ij})$ over all pairs | width of the pairwise-distance distribution; sets the largest achievable $\log C$ slope near the mode |

### 1.3 The valid-window rule, unchanged from the 1.5B validation

A window is valid only if all three hold simultaneously.

1. **Flatness.** The longest run of $\epsilon$ in which the local slope stays within $\mathrm{tol}=\max(0.15,\,0.06\lvert\hat D_2\rvert)$ of its own median, on a 3-point median-filtered slope, and the run spans at least **0.40 decades**. Five grid points minimum.
2. **Depletion floor.** $\epsilon\ge\epsilon_{10/N}$ (so at least $\sim5N$ pairs are inside the ball).
3. **Below saturation.** $C(\epsilon)\le0.1$.

The width is reported whether or not it is valid. There is no post-hoc choice of sub-interval: the rule, the 90-point grid, the 0.40-decade and 6% thresholds, and the $10/N$ / $0.1$ edges are the ones in `scripts_spark/d2_local.py`, fixed on the 1.5B before these matrices were read. The Theiler windows are $W=0,50,500$; at $N=1000$, $W=500$ keeps only pairs with $\lvert i-j\rvert>500$. All 12 runs fail the rule at all three windows with a window width of exactly 0.00 decades.

### 1.4 All per-trajectory numbers

$W=0$, $N=1000$ for every row. $\epsilon$ in units of $d_{typ}$; the span is $\log_{10}(\epsilon_{0.1}/\epsilon_{10/N})$; the chord is the fine-grid OLS; "points" counts 90-point grid points inside the band.

| trajectory | $\epsilon_{10/N}$ | $\epsilon_{0.1}$ | span (dec) | slope at edges | points | $D_2$ chord | T4-style (fit dec) | valid window |
|---|---:|---:|---:|---|---:|---:|---|---:|
| `childhood` | 0.777 | 0.925 | 0.076 | 10.3 -> 16.9 | 2 | 13.08 | 20.13 (0.014) | none (0.00 dec) |
| `cybersecurity` | 0.781 | 0.916 | 0.069 | 11.5 -> 16.6 | 2 | 14.59 | 18.77 (0.024) | none (0.00 dec) |
| `future_of_work` | 0.786 | 0.920 | 0.069 | 11.6 -> 17.3 | 2 | 14.57 | 19.46 (0.033) | none (0.00 dec) |
| `great_emu_war` | 0.609 | 0.873 | 0.157 | 2.7 -> 9.8 | 6 | 6.64 | 12.58 (0.020) | none (0.00 dec) |
| `healthy_habits` | 0.800 | 0.933 | 0.066 | 11.5 -> 18.8 | 3 | 15.06 | 22.11 (0.013) | none (0.00 dec) |
| `interstellar` | 0.754 | 0.912 | 0.082 | 9.0 -> 15.1 | 3 | 12.23 | 17.24 (0.019) | none (0.00 dec) |
| `personal_finance` | 0.776 | 0.913 | 0.071 | 11.5 -> 16.1 | 2 | 14.08 | 18.32 (0.027) | none (0.00 dec) |
| `quantum_consciousness` | 0.783 | 0.917 | 0.068 | 12.0 -> 16.5 | 2 | 14.71 | 18.30 (0.022) | none (0.00 dec) |
| `remote_work` | 0.794 | 0.925 | 0.067 | 11.7 -> 18.1 | 2 | 15.12 | 20.49 (0.024) | none (0.00 dec) |
| `robot_friendship` | 0.778 | 0.912 | 0.069 | 12.1 -> 16.0 | 2 | 14.52 | 17.52 (0.040) | none (0.00 dec) |
| `smartphones` | 0.803 | 0.933 | 0.065 | 12.0 -> 18.7 | 3 | 15.33 | 22.36 (0.013) | none (0.00 dec) |
| `transformers` | 0.760 | 0.897 | 0.072 | 11.9 -> 14.5 | 3 | 14.08 | 16.04 (0.066) | none (0.00 dec) |

| trajectory | TwoNN | MLE 5 / 10 / 20 | prominence (coverage) | nearest-twin median / p05 / frac $<10^{-3}$ | GPU `rel_nearest` (p05) | norm median |
|---|---:|---|---:|---|---:|---:|
| `childhood` | 7.61 | 11.7 / 11.7 / 13.0 | 0.992 (0%) | 0.735 / 0.453 / 0.000 | 1.056 (0.641) | 135.6 |
| `cybersecurity` | 7.58 | 11.6 / 11.8 / 13.4 | 0.995 (0%) | 0.759 / 0.531 / 0.000 | 1.094 (0.754) | 135.4 |
| `future_of_work` | 7.56 | 12.7 / 12.3 / 13.5 | 0.994 (0%) | 0.745 / 0.503 / 0.000 | 1.062 (0.727) | 135.4 |
| `great_emu_war` | 6.05 | 9.3 / 8.9 / 9.7 | **0.849 (14%)** | **0.533 / 0.082** / 0.000 | **0.682 (0.113)** | 134.9 |
| `healthy_habits` | 7.36 | 11.7 / 11.7 / 13.1 | 0.993 (0%) | 0.755 / 0.495 / 0.000 | 1.086 (0.697) | 135.5 |
| `interstellar` | 6.97 | 11.2 / 11.5 / 12.9 | 0.981 (0%) | 0.671 / 0.382 / 0.000 | 0.932 (0.517) | 135.3 |
| `personal_finance` | 7.65 | 11.8 / 11.9 / 13.5 | 0.990 (0%) | 0.753 / 0.461 / 0.000 | 1.074 (0.663) | 135.4 |
| `quantum_consciousness` | 7.12 | 12.4 / 12.4 / 13.8 | 0.991 (0%) | 0.722 / 0.459 / 0.000 | 1.044 (0.619) | 135.6 |
| `remote_work` | 7.84 | 12.7 / 12.7 / 14.0 | 0.993 (0%) | 0.755 / 0.526 / 0.000 | 1.080 (0.739) | 135.5 |
| `robot_friendship` | 8.09 | 12.7 / 12.9 / 14.2 | 0.994 (0%) | 0.720 / 0.518 / 0.000 | 1.050 (0.762) | 135.5 |
| `smartphones` | 7.12 | 12.0 / 12.0 / 13.7 | 0.991 (0%) | 0.772 / 0.491 / 0.000 | 1.120 (0.689) | 135.6 |
| `transformers` | 7.45 | 12.1 / 12.3 / 13.8 | 0.991 (0%) | 0.704 / 0.492 / 0.000 | 0.995 (0.678) | 133.5 |

How to read the two tables together. All clouds put the $C=10/N$ edge at 0.61-0.80 and the $C=0.1$ edge at 0.87-0.93, i.e. the usable interval is a 17-21% change in $\epsilon$, sampled by two or three of 90 grid points. The average slope needed to carry $C$ from $10/N$ to 0.1 across that interval is $\log_{10}(N/100)/\text{span}=1/\text{span}=12.2$-$15.4$, and the observed chord (12.2-15.3 for the 11 clean runs) matches it: the number is set by the width of the distance distribution, not by a fractal dimension. In every run the slope at the two edges differs by 2.6-7.3 units, and the maximum slope anywhere is 14.8-31.7, both above the 0.34-0.94 flatness tolerance and far above the 6.0 finite-$N$ ceiling. The 0.40-decade rule cannot be met by construction.

The distance CV column explains the cross-model position: 0.057-0.079 for these clouds against 0.110-0.135 for the 14B. A narrower distance distribution means a steeper $\log C$ in the 1%-to-10% band, hence a higher chord and T4, even though the nearest-neighbour estimators are lower. This is the main new quantitative fact in this section, and it is why the 9B's global estimates exceed the 14B's while its local ones do not.

---

## 2. Figure `q35d2_overview.png` — four clouds, four non-plateaus

![overview](figures/q35d2_overview.png)

**Setup.** Four of the 12 trajectories, chosen to span the TwoNN range and the repeat outlier: `great_emu_war` (TwoNN 6.05, lowest, and the only partial repeat), `healthy_habits` (7.36, the second-highest T4 chord), `interstellar` (6.97; mid-range MLE), `robot_friendship` (8.09, highest TwoNN). For each, $C(\epsilon)$ is drawn for Theiler $W=0,50,500$ on the common $\epsilon/d_{typ}$ grid; the grey vertical band is the admissible range $[\epsilon_{10/N},\epsilon_{0.1}]$; dotted horizontals in the top row are $10/N$ and $0.1$. The bottom row is the local slope, with the finite-$N$ ceiling $2\log_{10}N=6.0$ (violet dotted) and the fine-grid chord over the admissible range (grey dash-dot). No valid window is shaded because there is none.

**Axes.** Top row: $\epsilon/d_{typ}$ log, $10^{-2}$ to $2.5$; $C(\epsilon)$ log, $10^{-5}$ to $1$. Bottom row: same $x$; $d\log C/d\log\epsilon$ linear, 0 to 22 (the first-nonzero-$C$ spike of `great_emu_war` reaches 31.7 above the axis; it is a two-pair fluctuation). Each panel is one trajectory, $N=1000$.

**Numbers.** $W=0$, in the order of the panels:

| quantity | `great_emu_war` | `healthy_habits` | `interstellar` | `robot_friendship` |
|---|---:|---:|---:|---:|
| admissible span (dec) | 0.157 | 0.066 | 0.082 | 0.069 |
| $\epsilon$ at $C=10/N$ / $C=0.1$ | 0.609 / 0.873 | 0.800 / 0.933 | 0.754 / 0.912 | 0.778 / 0.912 |
| slope at those edges | 2.7 -> 9.8 | 11.5 -> 18.8 | 9.0 -> 15.1 | 12.1 -> 16.0 |
| chord $D_2$ (fine grid) | 6.64 | 15.06 | 12.23 | 14.52 |
| T4 max-$R^2$ | 12.58 | 22.11 | 17.24 | 17.52 |
| TwoNN | 6.05 | 7.36 | 6.97 | 8.09 |
| peak local slope (any $\epsilon$) | 31.7 | 18.8 | 16.8 | 16.8 |
| distance CV | 0.110 | 0.058 | 0.071 | 0.068 |

The three clean panels are nearly identical: $C(\epsilon)$ rises with a slope of 5-6 in a shallow range below 0.5, steepens to 11-18 through the 1%-to-10% band, and saturates at $C=1$ with a slope of zero. The $W=50$ and $W=500$ curves peel off only at the smallest $\epsilon$ (where excluding neighbouring states matters) and coincide through and above the admissible band. No panel has a horizontal segment: the slope is a monotone rise cut off by saturation.

`great_emu_war` is different in one specific way. Its $C(\epsilon)$ curve has an extra shelf between $10^{-4}$ and $10^{-2}$ — the correlation integral grows only as $\epsilon^{0.5-2.2}$ from $\epsilon=0.05$ to 0.5 — because the last 200 states contain near-recurrences every 7 tokens. That is what pulls its chord to 6.64 and its TwoNN to 6.05 while its T4 (12.58) remains in the normal range. The panel marks the period-7 dip; the effect is quantified in section 3.3 and in the shuffle control.

**What it shows.** The absence of a plateau is not idiosyncratic to one prompt: the same geometry (a distance distribution whose 1% and 10% quantiles are 17-21% apart in $\epsilon$) appears in all four panels. The four panels also show why the "D2" numbers move with the Theiler window: the only visible changes from $W=0$ to $W=500$ are below the depletion floor, except in `great_emu_war`, where the exclusion removes the repeated tail and lifts the chord by 9.4.

**What it does not show.** The 2-3-point sampling of the admissible band means the plotted slope inside the grey band is essentially an interpolation between grid points, not evidence of a fractal at finer resolution. The panels are at $N=1000$ only. The similarity of the four prompts is not a test of prompt-independence; all 12 are long expository requests from one prompt list. And `great_emu_war`'s shelf is a repetition effect, not a second scaling regime: on a 900-point grid the shelf is still only 0.15 decades wide and its slope drifts by 7 units across it.

---

## 3. Figure `q35d2_summary.png` — estimates, loop screen, and N dependence

![summary](figures/q35d2_summary.png)

### 3.1 (a) Filled circles are TwoNN; open squares are chords; crosses are T4

**Setup.** All 12 trajectories, sorted by TwoNN. Filled blue circles are TwoNN; open violet squares are the fine-grid chord over the admissible range (the number a reader gets who does not demand a plateau); grey crosses are the T4 max-$R^2$ estimate. No filled symbol of the "valid window $D_2$" kind appears because 0/12 passed. Horizontal bands are reference values: 1.5B TwoNN 5.44-6.23 (grey), 1.5B healthy T4 9.20-11.35 (blue), and 14B TwoNN 4.66-8.90 (light blue). The 14B chord band (8.10-10.27) is omitted because it overlaps the 1.5B T4 band; it is in the text.

**Axes.** $x$: the 12 trajectory names, sorted by TwoNN; $y$: $\hat D$ / TwoNN, 0-25. Numbers beside the circles are TwoNN; numbers beside the crosses are T4. The chord is not labelled point by point (it is between the two).

**Numbers.** TwoNN 6.05-8.09 (median 7.51); chord 6.64-15.33 (median 14.54); T4 12.58-22.36 (median 18.55) over 0.013-0.066 decades; MLE $k=10$ 8.90-12.89 (median 12.00). All 12 TwoNN values sit inside the 14B band; 11 of 12 are above the 1.5B band, the exception being `great_emu_war` (6.05), which is the repeat-contaminated run. All 12 chords except `great_emu_war` (6.64) sit above the 14B chord band, and all 12 T4 values sit above the 1.5B healthy T4 band. So by the global estimators the 9B is the highest-dimensional of the three clouds, while by TwoNN/MLE it is between them.

**What it shows.** Three things. (1) "The dimension of the 9B state cloud" is not a single number even before the scaling question: within one run the estimators span a factor up to 3.1, and the T4 value is always the largest because max-$R^2$ selects the steepest 0.01-0.07-decade sub-interval. (2) The 9B is not a compressed, recurrent-state-like object: its nearest-neighbour dimension is above the 1.5B's and its global distance-concentration slope is above both. (3) Any 9B number quoted as "D2" would be a chord over 0.065-0.082 decades (0.157 for the repeat run) — narrower than the 1.5B's already-narrow 0.04-0.10.

**What it does not show.** The bands are reference *values*, not confidence intervals, and the three models use different $N$, $d$ and prompt sets. TwoNN is not corrected for the finite-$N$ bias visible in panel (c). The chord is a 900-point evaluation of an estimator that the 90-point pipeline leaves undefined for 7/12; the five pipeline chords are 7.11, 11.69, 13.89, 15.47, 15.62, consistent with the fine-grid values to 0.3. And the T4 values are computed on the 1000-point matrix, not on the 2000-state rebuild that reproduced the published 1.5B numbers exactly.

### 3.2 (b) The loop screen: eleven clean, one late period-7 repeat

**Setup.** For each trajectory, the minimum of $p(k)/\mathrm{median}\,p(k\pm1,2,3)$ over lags $k=1\ldots599$, where $p(k)$ is the mean distance at lag $k$ in units of $d_{typ}$. The red band reaches 0.45, the largest prominence of any 1.5B loop; dashed lines are the pipeline's loop cut 0.6 and partial cut 0.8; the grey dotted line is the 1.5B healthy floor 0.91 and the blue dotted line the 14B floor 0.961. Bars are the 12 values; the orange bar is `great_emu_war`.

**Axes.** $x$: trajectories sorted by prominence; $y$: prominence, 0-1.13. A loop is a *low* bar. Coverage (fraction of positions returning within $0.1d_{typ}$ at the best lag) is 0% for every bar except `great_emu_war` (13.8%).

**Numbers.** Prominence 0.849-0.995; the second-lowest is 0.981, and 11 of 12 are $\ge0.981$. `great_emu_war` has its minimum at **lag 7**, prominence 0.849, and 13.8% coverage. Its nearest non-adjacent neighbour is 0.059 $d_{typ}$ at lags 56 and 70, and the pairs with $d(x_t,x_{t+7})<0.1d_{typ}$ cover positions 800-990 (72% of that tail). The other 11 have nearest-twin medians 0.67-0.77 and p05 0.38-0.53, with 0.000 fraction below $10^{-3}$ in all 12. The GPU screen gives the same ordering: `great_emu_war` has `rel_nearest` 0.682 and p05 0.113, all others 0.932-1.120 and p05 0.382-0.762.

**What it shows.** The absence of loops in 11 of 12 is not a threshold artefact: those values sit at the top of the metric's range, 0.38 above the loop cut and 0.02-0.03 above even the 14B floor. The lag-distance curve has no dip at any lag: even lag 1 has mean distance 0.726-0.767 $d_{typ}$, so there is no short-period alternation either. The exception is a *local* repeat: a 7-token cycle in the final fifth of one run, which returns the state to 5.9% of the cloud scale rather than to the 1.5B loops' 1.6%. That is qualitatively weaker than the 1.5B loops (coverage 70-95%), but it is a real near-recurrence and the nearest-neighbour screen sees it.

**What it does not show.** A repetition with period $>600$ tokens cannot produce a prominence dip in a 1000-token run; the screen would miss a slow loop that appears only once. The screen cannot rule out repetition in the *text* that does not return the state to within $0.1d_{typ}$; it measures state recurrence, not token n-grams. And the fixed rule's cut at 0.8 does not flag `great_emu_war` (0.849): that is why this section reports both the rule's count (0 full + 0 partial) and the evidence-based count (0 full + 1 partial), and does not move the threshold after seeing the data.

### 3.3 (c) N dependence: the window stays empty; the repeat, not finite $N$, moves the outlier

**Setup.** Nested prefixes of two trajectories chosen at the TwoNN extremes: `great_emu_war` (lowest) and `robot_friendship` (highest). At each $N\in\{500,750,1000\}$ the same pipeline is rerun with $W=0$: the top panel is TwoNN, the bottom panel the fine-grid chord over the admissible range. The valid window is 0.00 decades at every point.

**Axes.** $x$: $N=500,750,1000$ (nested prefixes). Top $y$: TwoNN, 4.0-10.2, with the 1.5B band 5.44-6.23 and 14B band 4.66-8.90 shaded. Bottom $y$: chord $D_2$, 5.0-17.0.

**Numbers.**

| $N$ | `great_emu_war` TwoNN | `great_emu_war` chord | `great_emu_war` MLE 10 | `robot_friendship` TwoNN | `robot_friendship` chord | `robot_friendship` MLE 10 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 6.76 | 11.95 | 10.8 | 7.73 | 13.96 | 12.4 |
| 750 | 6.68 | 10.95 | 10.1 | 7.84 | 14.56 | 12.7 |
| 1000 | 6.05 | 6.64 | 8.9 | 8.09 | 14.52 | 12.9 |

The admissible span moves 0.057 -> 0.080 -> 0.157 (`great_emu_war`) and 0.049 -> 0.059 -> 0.068 (`robot_friendship`).

**What it shows.** The "no window" verdict is not an $N=1000$ accident: it holds at 500 and 750 too. The two trajectories behave differently for a reason that is now understood: `robot_friendship`'s estimators are almost flat in $N$ (TwoNN +0.36, chord +0.56, MLE +0.5 over a doubling), the expected mild bias of a nearest-neighbour estimator, while `great_emu_war`'s collapse (chord -5.3, TwoNN -0.7, MLE -1.9) coincides exactly with the entry of the repeated tail at $t\approx800$ when the prefix length reaches 1000. The outlier is therefore a repetition effect, not evidence of a finite-$N$ correction with a different sign.

**What it does not show.** Nested prefixes are not independent samples: the $N=500$ cloud is a subset of the $N=750$ and $N=1000$ clouds; the panels show $W=0$ only, and $W=500$ would be undefined at $N=500$. And the exercise cannot simulate the $N$ that would settle the scaling question: extrapolating the measured chord below the depletion floor to where the admissible range would be 0.40 decades wide requires $N\sim4.6\times10^7$-$1.5\times10^8$ for the 11 clean runs (a factor $4.6\times10^4$-$1.5\times10^5$ more tokens than these runs), and $5.5\times10^4$ for `great_emu_war` only because its loop makes the local slope artificially shallow. The extrapolation assumes the chord slope holds for another 0.33 decades of $\epsilon$; it is an order-of-magnitude statement, not a measurement.

---

## 4. Figure `q35d2_time_shuffle.png` — the estimator at $W=0$ does not see time at all

![time shuffle](figures/q35d2_time_shuffle.png)

**Setup.** Two trajectories, `interstellar` (a clean mid-range run) and `great_emu_war` (the repeat outlier). For each, the raw distance matrix $D$ is compared with $D'=PDP^{\top}$, where $P$ is a random permutation matrix (seed 20261121) — equivalently, the same point cloud with its time labels permuted. This changes no distance: the multiset $\{d_{ij}\}$ is identical. $C(\epsilon)$ is evaluated on the same $\epsilon$ grid for both, so at $W=0$ the comparison is exact algebra, not a numerical check. The solid lines are raw, the dashed lines the shuffled matrix; blue/red/green are $W=0/50/500$. The grey vertical lines are the admissible edges of the raw run; the violet dotted line is $2\log_{10}N=6$. Chords in the text block are fine-grid values.

**Axes.** $x$: $\epsilon/d_{typ}$ log, 0.008-3; $y$: local slope, 0-22. The text block gives the chord $D_2$ and TwoNN for raw and shuffled.

**Numbers.**

| | `interstellar` raw / shuffled | `great_emu_war` raw / shuffled |
|---|---:|---:|
| $W=0$: $\max\lvert\Delta C\rvert$ | **0** | **0** |
| $W=0$: chord $D_2$ | 12.23 / 12.23 | 6.64 / 6.64 |
| $W=0$: TwoNN | 6.968 / 6.968 | 6.047 / 6.047 |
| $W=50$: chord $D_2$ | 14.04 / 12.24 | 8.98 / 6.70 |
| $W=500$: chord $D_2$ | 16.04 / 12.57 | 16.06 / 6.32 |
| $W=50$: $\max\lvert\Delta C\rvert$ | 0.018 | 0.023 |
| $W=500$: $\max\lvert\Delta C\rvert$ | 0.063 | 0.095 |
| valid window, any $W$ | none / none | none / none |

**What it shows.** At $W=0$ the correlation integral and TwoNN are **exactly invariant** under the permutation: the estimator is a point-cloud statistic by construction. This is the same control that made the old 1.5B "5.75 raw vs 5.69 shuffled" comparison a tautology. The only genuine temporal information enters through $W>0$, and the two trajectories show two different mechanisms. On `interstellar`, excluding near-in-time pairs raises the chord by 1.8 ($W=50$) and 3.8 ($W=500$): consecutive states are 0.756 $d_{typ}$ apart while the typical separation is $1\,d_{typ}$, so the trajectory is mildly smooth in time, not low-dimensional. On `great_emu_war`, the same exclusion raises the chord by 2.3 and **9.4** because it removes exactly the lag-7/56 repetitions that created the shelf; with the repeats excluded, its chord (16.06) is in line with the clean runs (12.2-15.3), which is independent confirmation that its low chord is a repetition artefact.

**What it does not show.** The shuffle preserves the point cloud exactly, so it cannot test whether a *different* cloud with the same time series would have a different dimension; it tests only the estimator's sensitivity to the ordering of a fixed cloud. It uses one permutation (the $W=0$ result is exact for every permutation, the $W>0$ result is one draw). It does not test phase-randomised surrogates, which destroy the point cloud as well as the order. And it says nothing about whether the last-layer state should be expected to carry dynamics at all.

---

## 5. Comparison with both previous models

### 5.1 The three datasets side by side

| quantity | 1.5B (20 local runs) | DS-14B (13 runs) | Qwen3.5-9B (12 runs, this section) |
|---|---|---|---|
| architecture | dense transformer | dense transformer (distilled Qwen-14B) | hybrid: 3 Gated DeltaNet + 1 full attention per period of 4 |
| hidden size | 1536 | 5120 | 4096 |
| horizons | three 4095-token runs, sixteen 1000-token runs, one 1026-token run | twelve 1000-token runs, one 1500-token run truncated to 1000 | twelve 1000-token runs |
| valid scaling window | none on the two healthy runs; one spurious 0.54-dec window ($\hat D_2=1.09$) on a period-53 loop | **none (0/13)** | **none (0/12)**, at $W=0,50,500$ |
| admissible span, healthy | 0.08 decades | 0.08-0.11 decades | 0.065-0.082 decades (0.157 repeat run) |
| T4-style fit width | 0.039-0.102 decades | 0.02-0.07 decades | 0.013-0.066 decades |
| T4-style $D_2$ | 9.20-11.35 healthy, 2.98-3.89 loops | 8.69-11.75 | **12.58-22.36** |
| TwoNN | 5.44-6.23 (all four validated runs) | 4.66-8.90, median 8.19 | **6.05-8.09, median 7.51** |
| MLE $k=10$ | 7.9-10.1 healthy, 3.1 loops | 10.97-15.87 | **8.90-12.89** |
| chord over admissible | — | 8.10-10.27 (90-pt grid) | 6.64-15.33 (900-pt grid; 7.11-15.62 on the 90-pt grid) |
| pairwise-distance CV | — | 0.110-0.135 | **0.057-0.079** (0.110 repeat run) |
| repetition loops | 9 full + 2 partial / 20; 6 full + 2 partial / 16 at 1000 tokens | **0 full + 0 partial / 13** | **0 full + 1 partial / 12** |
| loop prominence | 0.04-0.22 full (validated 0.058/0.098), 0.21/0.45 partial, 0.91-0.99 healthy | 0.961-0.992 | 0.981-0.995 clean; 0.849 partial |
| loop coverage | 70-95% | 0% | 0%; 13.8% in the partial |
| nearest-twin median ($d_{typ}$) | 0.016-0.017 loops, 0.42-0.54 healthy | 0.527-0.670 | 0.671-0.772 clean; 0.533 partial |
| old headline | 5.75 raw / 5.69 shuffled (max-$R^2$ chords) | no comparable headline; chords 8.10-10.27 | no comparable headline; chords 6.64-15.33, T4 12.58-22.36 |

### 5.2 (i) Does the 9B show a valid scaling region where the 1.5B and 14B did not?

**No.** Not one of the 12 passes the rule at any Theiler window, and the reason is the same one that made both previous healthy runs fail: the interval between the depletion floor and saturation is only 0.065-0.082 decades, sampled by 2-3 grid points, while the rule needs 0.40 decades and 5 points. The 9B is the **most** decisively windowless of the three in the sense that its distance distribution is the narrowest (CV 0.057-0.079): the admissible band is a 17-21% change in $\epsilon$, the chord must climb 12-15 units of slope to cross it, and the T4 estimator's best sub-intervals are 0.013-0.066 decades wide, narrower than the 14B's 0.02-0.07 and the 1.5B's 0.039-0.102.

This is not a claim that the 9B cloud is "more random" or dynamically higher-dimensional. It is a statement about resolution: with 1000 points and a pairwise-distance distribution this narrow, no estimator can extract a 0.4-decade slope. The hybrid architecture does not change that arithmetic. If anything it moves the required $N$ further away: the chord extrapolation puts a 0.40-decade window at $N\sim4.6\times10^7$-$1.5\times10^8$, versus $1.5\times10^4$-$1.7\times10^5$ for the 14B, because the local slope is steeper.

### 5.3 (ii) Are the 9B trajectories more or less prone to repetition loops?

**Far less than the 1.5B; statistically indistinguishable from the 14B.** Counts first.

- 1.5B: 9 full loops + 2 partial out of 20 local trajectories. At the same 1000-token horizon: 6 full + 2 partial out of 16.
- DS-14B: 0 full + 0 partial out of 13.
- Qwen3.5-9B: 0 full + **1 partial** out of 12. By the fixed prominence rule (<0.8) it is 0 full + 0 partial; the partial is identified by coverage (13.8%) and by the nearest-neighbour screen (0.059 $d_{typ}$).

The two near-recurrence definitions agree once their scales are matched.

**The GPU screen (`summary.json`).** It takes the last three quarters of the states ($h[250{:}]$), computes pairwise distances, excludes $\lvert i-j\rvert<50$, and reports the median and p05 of the nearest remaining distance divided by $\mathrm{median}_i\lVert h_i-\bar h\rVert$ (a centroid scale, not the pairwise $d_{typ}$). The values are median **0.682-1.120** and p05 **0.113-0.762**. Unlike the DS-14B case I cannot reproduce these exact numbers from the distance matrices alone (the centroid scale and the $h[250{:}]$ truncation need the states, which were not pulled), so the laptop and GPU numbers are quoted as two conventions, not as a reproduction.

**The laptop screen.** It uses all $N$ points, the same $\lvert i-j\rvert\ge50$ exclusion, and $d_{typ}$ as the scale; it reports the *fraction* below $10^{-3}d_{typ}$, plus median and p05. The values are median **0.533-0.772**, p05 **0.082-0.531**, fraction below $10^{-3}$ **0.000** in every run. Within either convention, 11 runs are far from a loop: the 1.5B loop runs' nearest-twin median was 0.016-0.017 $d_{typ}$, a factor **39-48** below the 9B clean medians. The one exception is `great_emu_war`: median 0.533, p05 0.082 — still 4.8-5.1x above the 1.5B loop median at p05, but 4.6-6.5x below the other clean runs at p05, and the only run where the GPU screen's p05 (0.113) approaches the 1.5B loop scale.

**Reconciling the screens is also a warning about the $10^{-3}$ threshold.** The absolute-threshold screen returns 0.000 for the 1.5B loops too, because those loops *drift*: their repeats sit at 1.6% of the cloud scale, not at $10^{-3}$. So the "fraction below $10^{-3}$" is not the loop detector, and the fact that it is zero here does not by itself prove no loops. The detector that found the 1.5B loops is the lag-distance prominence; the GPU `rel_nearest` median is a cruder version of the same signal. On the 9B, both agree: eleven runs have no near-recurrence at any lag, and one has a late 7-token one.

### 5.4 What the point-cloud values say about the hybrid architecture

The 9B is not a lower-dimensional object than either dense model, and it is not more regular. TwoNN's median (7.51) is above the 1.5B's 5.44-6.23 and below the 14B's 8.19; the MLE at $k=10$ (8.9-12.9) is above the 1.5B healthy range (7.9-10.1) and below the 14B's (11.0-15.9); the chord (median 14.5) and T4 (12.6-22.4) are above both. The rank order of "dimension" therefore depends on the estimator, which is the same conclusion as for the 14B, only more strongly: **the 9B last-layer cloud is locally 6-13 dimensional by nearest-neighbour estimators and 12-22 by global chords, with no estimator converged in $N$, and no valid scaling region at $N=1000$.**

The architectural reading is narrow but clear. A hybrid stack with linear-attention blocks carrying a fixed-size recurrent state might have been expected to compress the trajectory into a low-dimensional manifold or to make it periodic. Neither happens at the last layer. The last block of this model is a **full-attention** block and `h_last` is its post-final-norm output, so it is not a Gated-DeltaNet state; the recurrent states themselves were not saved, and this section cannot say whether they are low-dimensional. What it can say is that whatever compression the linear-attention blocks perform is not visible in the last-layer cloud: its dimensionality is in the dense-model range or above, and its loop propensity is in the dense-14B range. One caveat on the CV comparison: because `h_last` is post-RMSNorm, part of the narrow distance distribution is a property of the normalization and the learned gain rather than of the trajectory — but the 14B is measured at the same slot and still has almost twice the CV, so the difference between the clouds is real at this layer.

---

## 6. What this does not show

1. **One layer, one horizon, one decoding mode.** Only the last layer, only 1000 generated tokens, only FP32 greedy. Earlier layers, longer runs and sampling could differ; the earlier sections show that layer, window regime and prompt length change the state geometry.
2. **The linear-attention state was not measured.** The Gated-DeltaNet blocks' recurrent states are not in `h_last` (and the last block is full attention), so nothing here tests whether *that* state is low-dimensional or periodic. The collector saved block outputs 8, 16, 24 and 31 but they were not pulled to the laptop.
3. **Twelve prompt topics are not a distribution.** All 12 are long expository prompts from one list; absence of loops is a statement about these runs, not a loop rate for the model. The cross-model comparison uses a different prompt list, so architecture and prompt set are confounded.
4. **The 0.40-decade rule is a resolution requirement, not a physical law.** A cloud can be low-dimensional and still fail it if the distance distribution is narrow; the test is one-sided. "No scaling region" means "not measurable at this $N$", not "no low-dimensional structure exists". The extrapolation in section 3.3 puts the required $N$ at $10^{7.7}$-$10^{8.2}$ points, which would still have to pass the flatness condition.
5. **Chords and T4 numbers are not dimensions.** The 6.64-15.33 chord and the 12.58-22.36 T4 value are quoted only so that numbers in circulation can be recognized as chords. The report deliberately does not promote either to a $\hat D_2$.
6. **The loop screen is a state-recurrence test.** It can miss a loop with period $>600$ tokens in a 1000-token run, and it does not inspect the token text. The `great_emu_war` repeat is detected through coverage and nearest neighbours, not through the prominence cut; a text-level repetition that does not return the state to $0.1d_{typ}$ would not be flagged.
7. **Time shuffle is not a dynamics test.** It shows that the $W=0$ estimator is blind to order (exactly) and that $W>0$ responds to temporal smoothness and to the repeated tail. It does not show that the trajectory has no dynamics, only that the measured dimension is a static property of the cloud.
