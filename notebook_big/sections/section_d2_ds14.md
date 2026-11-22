# Part D2 (DS-14B). No scaling region and no repetition loops in thirteen 1000-token trajectories

## Summary

The last-layer states of `DeepSeek-R1-Distill-Qwen-14B` (FP32, greedy, attention window 4096, 1000 generated tokens, $d=5120$) form a point cloud on which the validated correlation-dimension pipeline finds **no scaling region in any of the 13 trajectories**. Under the same fixed rule that was applied to the 1.5B, every run has only a **0.08-decade** usable range of $\epsilon$ (0.11 for one run), sampled by **4 grid points** (5 for that one run); the rule requires at least 5 points over at least 0.40 decades. The local slope across that range moves by 0.2-1.2 units and sits at 7.0-10.2, above the finite-$N$ ceiling $2\log_{10}N=6.0$; in 10 of 13 runs the edge-to-edge drift alone exceeds the 6% flatness tolerance. There is therefore no $\hat D_2$ to quote: the number you get by ignoring the rule is a chord of 8.10-10.27, and the old max-$R^2$ estimator returns 8.69-11.75 over windows 0.02-0.07 decades wide.

The point-cloud estimates disagree with each other, which is itself the result. TwoNN gives **4.66-8.90** (median 8.19), the Levina-Bickel MLE at $k=10$ gives **10.97-15.87**, and the two differ by a factor 1.7-2.4. Neither is converged at $N=1000$: on `great_emu_war`, TwoNN falls 7.82 -> 6.30 -> 4.66 as $N$ goes 500 -> 750 -> 1000, while on `interstellar` it moves 9.51 -> 8.75 -> 8.90. The MLE moves much less (12.6 -> 11.0 and 14.6 -> 15.1). The correct statement is a range, not a dimension: **the 14B last-layer cloud is locally 4.7-15.9 dimensional depending on the estimator and the neighbourhood size, with no estimator stable in $N$.**

**No trajectory repeats.** The lag-distance prominence is **0.961-0.992** in all 13 (a loop is $<0.6$; the 1.5B's full loops were 0.04-0.22), the fraction of positions returning within $0.1\,d_{typ}$ at the best lag is **0.0%** (1.5B loops: 70-95%), and 0% of points have a nearest non-adjacent neighbour below $10^{-3}d_{typ}$. The GPU-side `summary.json` agrees exactly once its convention is reproduced: median `rel_nearest` **0.728-0.996**, p05 **0.355-0.557** (the 1.5B loop runs had nearest-twin median 0.016-0.017 $d_{typ}$). So the 14B trajectories are **less** loop-prone than the 1.5B: **0 full + 0 partial out of 13**, against 9 full + 2 partial out of 20 on the 1.5B (6 full + 2 partial out of the 16 runs at the same 1000-token horizon).

Two controls make the interpretation explicit. **N-dependence:** at $N=500/750/1000$ the valid window is still empty at every $N$; the admissible range grows only 0.054 -> 0.081 decades. **Time shuffle:** permuting the distance-matrix rows and columns identically leaves the point cloud unchanged, and at $W=0$ the correlation integral and TwoNN are **bit-identical**; only the Theiler band moves, changing the chord by at most 0.51 ($W=50$) or 1.33 ($W=500$). The measured $D_2$ is a static geometry statistic, not a dynamical one.

The answer to the two explicit 1.5B questions is therefore: **(i) no**, the 14B has no valid scaling region where the 1.5B did not -- neither has one on a healthy trajectory, and the only 1.5B window was the spurious 0.54-decade window on a loop, which has no 14B counterpart because there are no 14B loops. **(ii) less prone**, by a wide margin (0/13 against 9/20, and 0/13 against 6 full + 2 partial of 16 at equal horizon), with the two nearest-neighbour screens agreeing once their scales are matched.

---

## 1. Data, definitions, and the fixed decision rule

### 1.1 The 13 trajectories

| item | value |
|---|---|
| model | `DeepSeek-R1-Distill-Qwen-14B`, 48 blocks, hidden size 5120 |
| precision / decoding | FP32 (TF32 off), greedy |
| attention | window 4096, so no clipping inside the 1000-1500 token horizon |
| states | last layer (`h_last`), one $[N,5120]$ cloud per trajectory |
| input | `results_big/ds14/d2/dist_<name>_h_last.npy`, fp32 Euclidean $[N,N]$, plus `summary.json` |
| horizon | 1000 tokens for 12 runs; the 13th is `interstellar_3000`, run to 1500 and **truncated to its first 1000 states** |
| prompts | 12 prompt topics: the 12 lines of `prompts_d2.txt`, plus a second, longer interstellar prompt from the first line of `prompts.txt` (65-token prompt) |
| state norms | median 179.7-193.4, late/early norm ratio 0.987-1.048 (no drift or blow-up) |

One data fact must precede the comparison with the "12 prompts" description: `dist_interstellar_h_last.npy` (1000 states, 51-token prompt from `prompts_d2.txt`) and `dist_interstellar_3000_h_last.npy` (1500 states, 65-token prompt from `prompts.txt`) are **two different trajectories**, not a prefix pair. Their first 1000 generated tokens differ from token 0 and their states differ by up to 153 in Euclidean norm. Both are kept, so the 13 trajectories come from 12 prompt topics, with interstellar represented twice by two different texts (the `prompts.txt` variant adds one sentence: 65 prompt tokens against 51).

The distance matrices are float32 with diagonal entries of 0-0.18 and a smallest off-diagonal distance of 35.9 (for `interstellar`); every $\epsilon$ in the admissible range is $\ge 0.6\,d_{typ}\approx 140$, so the diagonal round-off cannot enter any count that matters here.

### 1.2 Definitions

Let $d_{ij}=\lVert x_i-x_j\rVert$, $N$ points, Theiler window $W$.

| quantity | expression | note |
|---|---|---|
| typical scale | $d_{typ}=\mathrm{median}_i\left(\frac{1}{N}\sum_j d_{ij}^2\right)^{1/2}$ | median over points of the rms distance to the cloud (512-point subsample, seed 0); makes the 14B ($d=5120$) and 1.5B ($d=1536$) comparable |
| correlation integral | $C(\epsilon)=\dfrac{\#\{(i,j):\ |i-j|>W,\ d_{ij}<\epsilon\}}{\#\{(i,j):\ |i-j|>W\}}$ | denominator $N(N-1)-2\sum_{k\le W}(N-k)$; at $W=0$ exactly $\frac{2}{N(N-1)}\#\{i<j:d_{ij}<\epsilon\}$ |
| grid | 90 log-spaced $\epsilon/d_{typ}\in[10^{-2},2.5]$ | one sort of the upper triangle, then `searchsorted`; $W$-band counts subtracted from the diagonals |
| local slope | $\mathrm{d}\log C/\mathrm{d}\log\epsilon$ | central differences, second order at the edges |
| TwoNN | $\mu_i=r_2^{(i)}/r_1^{(i)}$, $\hat d=1/\langle\ln\mu\rangle$ | Pareto MLE for $P(\mu>M)=M^{-d}$; the CDF-OLS variant is the same estimator and agrees; the "$\log 2/\langle\log_2\mu\rangle$" variant returns $0.48d$ and is not used |
| Levina-Bickel MLE | $m_k(i)=\frac{1}{k-1}\sum_{j<k}\ln\frac{d_k(i)}{d_j(i)}$, $\hat d_k=\langle 1/m_k\rangle$ | $k=5,10,20$ |
| repetition screen | fraction of points with a nearest neighbour at $\lvert i-j\rvert\ge50$ below $10^{-3}d_{typ}$ | median and p05 of that relative distance are also reported |
| lag-distance loop screen | $p(k)=\overline{\lVert x_t-x_{t+k}\rVert}/d_{typ}$, prominence $=p(k)/\mathrm{median}\,p(k\pm1,2,3)$ | a drifting loop has a sharp dip; coverage is the fraction of positions with $d_{t,t+p}<0.1d_{typ}$ |
| admissible range | $\epsilon\in[\epsilon_{10/N},\epsilon_{0.1}]$, the interpolation points where $C=10/N$ and $C=0.1$ | at least $\sim5N$ pairs inside the ball at the left edge; below saturation at the right |
| chord $D_2$ | OLS slope of $\log C$ on $\log\epsilon$ over the whole admissible range | what you get if you do not demand a plateau |
| T4-style estimator | the reference `fit_plateau`: maximum $R^2$ over every sub-interval with $\ge6$ grid points, on percentiles 2-60 of a 1000-point subsample, Theiler 20 | reproduces `results/d2/` exactly on the 1.5B; not a plateau test |

### 1.3 The valid-window rule, unchanged from the 1.5B validation

A window is valid only if all three hold simultaneously.

1. **Flatness.** The longest run of $\epsilon$ in which the local slope stays within $\mathrm{tol}=\max(0.15,\,0.06\lvert\hat D_2\rvert)$ of its own median, on a 3-point median-filtered slope, and the run spans at least **0.40 decades**. Five grid points minimum.
2. **Depletion floor.** $\epsilon\ge\epsilon_{10/N}$ (so at least $\sim5N$ pairs are inside the ball).
3. **Below saturation.** $C(\epsilon)\le0.1$.

The width is reported whether or not it is valid. There is no post-hoc choice of sub-interval: the rule, the 90-point grid, the 0.40-decade and 6% thresholds, and the $10/N$ / $0.1$ edges are the ones in `scripts_spark/d2_local.py`, fixed on the 1.5B before these matrices were read. The Theiler windows are $W=0,50,500$; at $N=1000$, $W=500$ keeps only pairs with $\lvert i-j\rvert>500$, i.e. one quarter of the unordered pairs, and at $N=500$ it leaves none, so the $N=500$ N-dependence check uses $W=0,50$ only.

### 1.4 All per-trajectory numbers

$W=0$, $N=1000$ for every row (`interstellar_3000` truncated).

| trajectory | $\epsilon_{10/N}$ | $\epsilon_{0.1}$ | span (dec) | slope at edges | $D_2$ chord | T4-style (fit dec) | valid window |
|---|---:|---:|---:|---|---:|---|---:|
| `childhood` | 0.649 | 0.824 | 0.08 | 9.6 -> 8.9 | 9.67 | 9.00 (0.03) | none (0.00 dec) |
| `cybersecurity` | 0.654 | 0.833 | 0.08 | 10.1 -> 8.9 | 9.40 | 9.38 (0.03) | none (0.00 dec) |
| `future_of_work` | 0.655 | 0.843 | 0.08 | 9.7 -> 8.8 | 9.09 | 10.22 (0.02) | none (0.00 dec) |
| `great_emu_war` | 0.672 | 0.846 | 0.08 | 9.1 -> 9.3 | 10.27 | 10.00 (0.02) | none (0.00 dec) |
| `healthy_habits` | 0.634 | 0.827 | 0.11 | 8.9 -> 8.0 | 8.70 | 9.98 (0.02) | none (0.00 dec) |
| `interstellar` | 0.651 | 0.826 | 0.08 | 9.4 -> 9.0 | 9.73 | 8.69 (0.04) | none (0.00 dec) |
| `interstellar_3000` | 0.670 | 0.851 | 0.08 | 9.1 -> 10.0 | 9.59 | 10.54 (0.03) | none (0.00 dec) |
| `personal_finance` | 0.601 | 0.803 | 0.08 | 7.0 -> 7.8 | 8.10 | 9.05 (0.02) | none (0.00 dec) |
| `quantum_consciousness` | 0.669 | 0.845 | 0.08 | 10.2 -> 9.6 | 9.86 | 9.74 (0.06) | none (0.00 dec) |
| `remote_work` | 0.663 | 0.846 | 0.08 | 9.8 -> 9.2 | 9.44 | 10.38 (0.02) | none (0.00 dec) |
| `robot_friendship` | 0.658 | 0.840 | 0.08 | 9.8 -> 9.0 | 9.44 | 10.02 (0.02) | none (0.00 dec) |
| `smartphones` | 0.662 | 0.849 | 0.08 | 9.8 -> 9.4 | 9.18 | 11.75 (0.02) | none (0.00 dec) |
| `transformers` | 0.654 | 0.835 | 0.08 | 10.1 -> 9.1 | 9.26 | 9.25 (0.07) | none (0.00 dec) |

| trajectory | TwoNN | MLE 5 / 10 / 20 | prominence (coverage) | nearest-twin median / frac $<10^{-3}$ | GPU `rel_nearest` (p05) |
|---|---:|---|---:|---|---:|
| `childhood` | 8.13 | 15.0 / 14.2 / 15.0 | 0.990 (0%) | 0.60 / 0.000 | 0.907 (0.497) |
| `cybersecurity` | 8.21 | 17.3 / 15.8 / 16.3 | 0.986 (0%) | 0.62 / 0.000 | 0.964 (0.540) |
| `future_of_work` | 7.42 | 14.7 / 13.8 / 14.6 | 0.984 (0%) | 0.60 / 0.000 | 0.884 (0.534) |
| `great_emu_war` | 4.66 | 11.6 / 11.0 / 11.6 | 0.961 (0%) | 0.53 / 0.000 | 0.728 (0.377) |
| `healthy_habits` | 7.23 | 17.2 / 15.7 / 16.2 | 0.990 (0%) | 0.67 / 0.000 | 0.996 (0.478) |
| `interstellar` | 8.90 | 16.5 / 15.1 / 15.3 | 0.988 (0%) | 0.62 / 0.000 | 0.919 (0.481) |
| `interstellar_3000` | 8.43 | 15.2 / 14.6 / 15.1 | 0.987 (0%) | 0.62 / 0.000 | 0.875 (0.422) |
| `personal_finance` | 8.19 | 16.1 / 15.8 / 16.7 | 0.984 (0%) | 0.64 / 0.000 | 0.923 (0.355) |
| `quantum_consciousness` | 8.04 | 16.3 / 15.6 / 16.2 | 0.989 (0%) | 0.63 / 0.000 | 0.930 (0.544) |
| `remote_work` | 8.63 | 15.2 / 14.4 / 15.6 | 0.987 (0%) | 0.64 / 0.000 | 0.936 (0.519) |
| `robot_friendship` | 8.19 | 15.2 / 14.1 / 14.4 | 0.991 (0%) | 0.58 / 0.000 | 0.886 (0.522) |
| `smartphones` | 8.64 | 15.4 / 14.4 / 15.1 | 0.992 (0%) | 0.64 / 0.000 | 0.925 (0.557) |
| `transformers` | 8.43 | 16.6 / 15.9 / 16.5 | 0.986 (0%) | 0.65 / 0.000 | 0.990 (0.547) |

How to read the two tables together. $\epsilon$ is in units of $d_{typ}$; all 13 clouds put the $C=10/N$ edge at 0.60-0.67 and the $C=0.1$ edge at 0.80-0.85. The usable interval is therefore a 20% change in $\epsilon$, sampled by four of the 90 grid points. The average slope needed to take $C$ from the depletion floor to saturation across that interval is 8.7-12.4 (depending on the 0.08-0.11 span); the observed chord is lower (8-10) only because the grid endpoints are inside $[0.01,0.1]$ and the curve bends. Both numbers are set by the width of the distance distribution, not by a fractal dimension. The 0.40-decade rule cannot be met by construction; in 10 of the 13 runs the slope also drifts by more than the flatness tolerance ($\pm0.48$ to $\pm0.59$) across the admissible range, and the slope exceeds the $2\log_{10}N$ ceiling in every run.

---

## 2. Figure `ds14d2_overview.png` — four clouds, four non-plateaus

![overview](figures/ds14d2_overview.png)

**Setup.** Four of the 13 trajectories, chosen to span the TwoNN range and the T4 range: `great_emu_war` (TwoNN 4.66, lowest), `healthy_habits` (7.23, the widest admissible range), `interstellar` (8.90, highest TwoNN), `transformers` (8.43; T4 chord 9.25 and MLE 15.9). For each, $C(\epsilon)$ is drawn for Theiler $W=0,50,500$ on the common $\epsilon/d_{typ}$ grid; the grey vertical band is the admissible range $[\epsilon_{10/N},\epsilon_{0.1}]$; dotted horizontals in the top row are $10/N$ and $0.1$. The bottom row is the local slope, with the finite-$N$ ceiling $2\log_{10}N=6.0$ (violet dotted) and the $W=0$ chord over the admissible range (grey dash-dot). No valid window is shaded because there is none.

**Axes.** Top row: $\epsilon/d_{typ}$ log, $10^{-2}$ to $2.5$; $C(\epsilon)$ log, $10^{-5}$ to $1$. Bottom row: same $x$; $d\log C/d\log\epsilon$ linear, 0 to 11-13. Each panel is one trajectory, $N=1000$.

**Numbers.** $W=0$, in the order of the panels:

| quantity | `great_emu_war` | `healthy_habits` | `interstellar` | `transformers` |
|---|---:|---:|---:|---:|
| admissible span (dec) | 0.08 | 0.11 | 0.08 | 0.08 |
| $\epsilon$ at $C=10/N$ / $C=0.1$ | 0.672 / 0.846 | 0.634 / 0.827 | 0.651 / 0.826 | 0.654 / 0.835 |
| slope at those edges | 9.1 -> 9.3 | 8.9 -> 8.0 | 9.4 -> 9.0 | 10.1 -> 9.1 |
| chord $D_2$ | 10.27 | 8.70 | 9.73 | 9.26 |
| TwoNN | 4.66 | 7.23 | 8.90 | 8.43 |
| peak local slope (any $\epsilon$) | 24.2 | 29.0 | 30.3 | 27.9 |

The $C(\epsilon)$ curves are nearly indistinguishable across trajectories and across Theiler windows: the $W=50$ and $W=500$ curves peel off only at the smallest $\epsilon$ (where removing adjacent pairs matters) and coincide through and above the admissible band. The slope curves have the same shape in all four panels: a noisy steep rise at small $\epsilon$, a bowl at 3-5, then the bulk rise to 9-11 and a drop to zero at saturation. No panel has a horizontal segment.

**What it shows.** The absence of a plateau is not idiosyncratic to one prompt. It is the same geometric object in all four: a distance distribution whose $1\%$ to $10\%$ quantiles are 20% apart in $\epsilon$. The four panels also show why the "D2" numbers move with the Theiler window: the only visible changes from $W=0$ to $W=500$ are at small $\epsilon$, below the depletion floor, so they cannot create or destroy a valid window.

**What it does not show.** The 4-point sampling of the admissible band means the plotted slope inside the grey band is essentially an interpolation between grid points; the curve is not evidence of a fractal at any finer resolution. The panels are at $N=1000$ only. The fact that the slope inside the band is close to the chord (by construction) does not mean the chord is a dimension: the same construction would return 12 for any smooth distance distribution with a 0.08-decade 1%-to-10% range. And the (visible) similarity of the four prompts is not a test of prompt-independence: these prompts are all long expository requests, and 12 of 13 are from one prompt list.

---

## 3. Figure `ds14d2_summary.png` — estimates, loop screen, and N dependence

![summary](figures/ds14d2_summary.png)

### 3.1 (a) Filled circles are TwoNN; open squares are chords; there is no filled D2

**Setup.** All 13 trajectories, sorted by TwoNN. The open violet squares are the $W=0$ chord over the admissible range -- the number a reader gets who does not demand a plateau. The filled blue circles are TwoNN. No filled symbol of the "valid window $D_2$" kind appears because 0/13 passed. Horizontal bands are the 1.5B reference values: TwoNN 5.44-6.23 (grey), the period-53/69 loop T4 chords 2.98-3.89 (red), the healthy T4 chords 9.20-11.35 (blue), and the old headline 5.75 (black dashed, from `docs/MEASUREMENTS.md`, with 5.69 on its time-shuffled control).

**Axes.** $x$: the 13 trajectory names, sorted by TwoNN; $y$: $\hat D$ / TwoNN, 0-13.6. Numbers beside the symbols are the plotted values.

**Numbers.** TwoNN 4.66-8.90 (median 8.19); chord 8.10-10.27; T4-style 8.69-11.75 with fit windows 0.02-0.07 decades. The MLE $k=10$ (not drawn, table above) is 10.97-15.87. So the 14B TwoNN median (8.19) sits above the 1.5B TwoNN band, the chords partly overlap its healthy T4 band (8.10-10.27 against 9.20-11.35), and the MLE (10.97-15.87) is above the TwoNN band with its lower end overlapping the top of the T4 band. The lowest TwoNN (`great_emu_war`, 4.66) is below the 1.5B band but is also the least $N$-stable of the 13 (7.82 at $N=500$).

**What it shows.** Three things. (1) The old 1.5B headline 5.75 has no 14B analogue that means the same thing: the 14B chords are 8-10 and the 14B TwoNN spans the 1.5B band and both sides of it. (2) The point-cloud estimators disagree by a factor 1.7-2.4 (MLE/TwoNN), so "the dimension of the 14B state cloud" is not a single number even before the scaling question. (3) Any 14B number quoted as "D2" would be a chord, and chords here are 0.02-0.07 decades wide -- narrower than the 1.5B's already-narrow 0.04-0.10.

**What it does not show.** The bands are reference *values*, not confidence intervals; the 1.5B and 14B use different $N$, $d$ and prompt sets. TwoNN is not corrected for the finite-$N$ bias visible in panel (c), so the vertical position of a point is not an unbiased dimension. The T4-style chords are computed on the 1000-point matrix, not on the 2000-state rebuild which reproduced the published 1.5B numbers exactly, so they are the same code path but not a published-number reproduction.

### 3.2 (b) The loop screen is flat: every trajectory is in the healthy cluster

**Setup.** For each trajectory, the minimum of $p(k)/\mathrm{median}\,p(k\pm1,2,3)$ over lags $k=1\ldots599$, where $p(k)$ is the mean distance at lag $k$ in units of $d_{typ}$. The red band reaches 0.45, the largest prominence of any 1.5B loop (the nine full loops were 0.04-0.22; two partial ones 0.21 and 0.45); the dashed lines are the pipeline's loop cut 0.6 and partial cut 0.8; the grey dotted line is the 1.5B healthy floor 0.91. Bars are the 13 values.

**Axes.** $x$: trajectories sorted by prominence; $y$: prominence, 0-1.06. A loop is a *low* bar. The coverage (fraction of positions returning within $0.1d_{typ}$ at the best lag) is 0% for every bar.

**Numbers.** Prominence 0.961-0.992; coverage exactly 0.0 for all 13. The lowest is `great_emu_war` (0.961), which also has the lowest nearest-twin distance (0.53 $d_{typ}$). The 1.5B full loops were 0.04-0.22 (the two validated runs 0.058 and 0.098; coverage 70-95%), and the 1.5B healthy runs 0.91-0.99. All 13 14B bars sit inside the 1.5B healthy range, and the margin to the loop cut is 0.36 (0.961-0.6).

**What it shows.** The absence of loops is not a threshold artefact: the values are at the *top* of the metric's range, not near the cut. The lag-distance curve has no dip at any lag: even the shortest lag (1 token) has mean distance 0.79-0.83 $d_{typ}$ and the minimum over lags 1-49 is at lag 1 in every trajectory, so there is no short-period alternation either. The 1.5B loop signature (a dip to 6-10% of the neighbouring lags at a period of 12-81) is simply absent.

**What it does not show.** A repetition whose period exceeds 600 tokens cannot produce a prominence dip in a 1000-token run (at most one recurrence); the screen would miss a slow loop that appears only once. The screen also cannot rule out repetition in the *text* that does not return the state to within $0.1d_{typ}$; it measures state recurrence, not token n-grams. And a 1000-token horizon is short relative to the 1.5B runs that looped at 4095 tokens, so the comparison in section 5 uses the matched 1000-token subset as well as the full 20.

### 3.3 (c) N dependence: the window stays empty and TwoNN is not converged

**Setup.** Nested prefixes of two trajectories chosen at the TwoNN extremes: `great_emu_war` (lowest TwoNN) and `interstellar` (highest). At each $N\in\{500,750,1000\}$ the same pipeline is rerun with $W=0$: the top panel is TwoNN, the bottom panel the chord over the admissible range. The admissible span grows from 0.054 to 0.081 decades.

**Axes.** $x$: $N=500,750,1000$ (nested prefixes). Top $y$: TwoNN, 4.0-10.2, with the 1.5B TwoNN band 5.44-6.23 in grey. Bottom $y$: chord $D_2$, 9.0-10.8. The valid-window width is 0.00 dec at every point.

**Numbers.**

| $N$ | `great_emu_war` TwoNN | `great_emu_war` chord | `interstellar` TwoNN | `interstellar` chord |
|---:|---:|---:|---:|---:|
| 500 | 7.82 | 10.37 | 9.51 | -- |
| 750 | 6.30 | 10.18 | 8.75 | 9.39 |
| 1000 | 4.66 | 10.27 | 8.90 | 9.73 |

For `interstellar` at $N=500$ the admissible range contains fewer than three grid points, so not even the chord is defined. The MLE $k=10$ moves 12.6 -> 12.2 -> 11.0 (`great_emu_war`) and 14.6 -> 15.2 -> 15.1 (`interstellar`), i.e. much less than TwoNN.

**What it shows.** The "no window" verdict is not an $N=1000$ accident: it holds at 500 and 750 too. The chord is remarkably stable (10.2-10.4 and 9.4-9.7) while TwoNN moves by 3.2 units on one trajectory and 0.8 on the other; the chord is stable because it is measuring the width of the distance distribution, which is set by $d$ and the cloud shape, not by $N$. TwoNN's drift is the expected finite-$N$ bias of a nearest-neighbour estimator in a high-dimensional, non-uniform cloud; the direction differs between trajectories, so there is no single correction.

**What it does not show.** Nested prefixes are not independent samples: the $N=500$ cloud is a subset of the $N=750$ and $N=1000$ clouds, so the curves are correlated. The $W=500$ curve is undefined at $N=500$ (no pairs with $\lvert i-j\rvert>500$) and is not shown. And the exercise cannot simulate the $N$ that would actually settle the question: extrapolating the measured $C(\epsilon)$ (e.g. $C(0.33d_{typ})\approx 1.3\times10^{-4}$) puts the $10/N$ floor 0.4 decades below $C=0.1$ only at $N\sim7\times10^{4}$ (range $1.5\times10^{4}$-1.7$\times10^{5}$), 15-170 times more tokens than these runs contain.

---

## 4. Figure `ds14d2_time_shuffle.png` — the estimator at $W=0$ does not see time at all

![time shuffle](figures/ds14d2_time_shuffle.png)

**Setup.** Two trajectories, `interstellar` and `great_emu_war`. For each, the raw distance matrix $D$ is compared with $D'=PDP^{\top}$, where $P$ is a random permutation matrix (seed 20261121) -- equivalently, the same point cloud with its time labels permuted. This changes no distance: the multiset $\{d_{ij}\}$ is identical. $C(\epsilon)$ is evaluated on the same $\epsilon$ grid for both, so at $W=0$ the comparison is exact algebra, not a numerical check. The solid lines are raw, the dashed lines the shuffled matrix; blue/green/red are $W=0/50/500$. The grey vertical lines are the admissible edges of the raw run; the violet dotted line is $2\log_{10}N=6$.

**Axes.** $x$: $\epsilon/d_{typ}$ log, 0.008-3; $y$: local slope, 0-12. The text block gives the chord $D_2$ and TwoNN for raw and shuffled.

**Numbers.**

| | `interstellar` raw / shuffled | `great_emu_war` raw / shuffled |
|---|---:|---:|
| $W=0$: $\max\lvert\Delta C\rvert$ | **0** | **0** |
| $W=0$: chord $D_2$ | 9.73 / 9.73 | 10.27 / 10.27 |
| $W=0$: TwoNN | 8.897 / 8.897 | 4.661 / 4.661 |
| $W=50$: chord $D_2$ | 10.03 / 9.75 | 10.76 / 10.26 |
| $W=500$: chord $D_2$ | 11.05 / 9.73 | 11.52 / 10.45 |
| $W=50$: $\max\lvert\Delta C\rvert$ | 0.011 | 0.018 |
| $W=500$: $\max\lvert\Delta C\rvert$ | 0.039 | 0.020 |
| valid window, any $W$ | none / none | none / none |

**What it shows.** At $W=0$ the correlation integral and TwoNN are **exactly invariant** under the permutation: the estimator is a point-cloud statistic by construction. This is the control that the earlier "raw vs time-shuffled" comparison in `results/d2/` implicitly relied on: when the old numbers were 5.75 raw vs 5.69 shuffled, the near-equality was not evidence that the dynamics are trivial; it was a tautology at $W=0$ (and the old code used Theiler 20 for raw and 0 for shuffled, so even the comparison was not matched). The only genuine temporal information enters through $W>0$: excluding near-in-time pairs changes the chord by 0.28-0.50 at $W=50$ and 1.07-1.32 at $W=500$, because consecutive last-layer states are 0.79-0.83 $d_{typ}$ apart while the typical separation is $1\,d_{typ}$ by construction. This is a mild temporal smoothness, not a low-dimensional trajectory: it shifts a chord but does not open a window.

**What it does not show.** The shuffle preserves the point cloud exactly, so it cannot test whether a *different* cloud with the same time series would have a different dimension; it tests only the estimator's sensitivity to the ordering of a fixed cloud, which is the right control for a point-cloud estimator but is silent about dynamics. It uses one permutation (the $W=0$ result is exact for every permutation, the $W>0$ result is one draw). It does not test phase-randomised surrogates, which destroy the point cloud as well as the order and therefore cannot separate the two effects. And it says nothing about whether the last-layer state should be expected to carry dynamics at all; a state cloud can have a low dimension for reasons unrelated to temporal recurrence.

---

## 5. Comparison with the 1.5B

### 5.1 The two datasets side by side

| quantity | 1.5B (20 local runs) | 14B (13 runs, this section) |
|---|---|---|
| hidden size | 1536 | 5120 |
| horizons | three 4095-token runs, sixteen 1000-token runs, one 1026-token run | twelve 1000-token runs, one 1500-token run truncated to 1000 |
| valid scaling window | none on the two healthy runs; one spurious 0.54-dec window ($\hat D_2=1.09$) on a period-53 loop | **none (0/13)** |
| admissible span, healthy | 0.08 decades | 0.08-0.11 decades |
| T4-style fit width | 0.039-0.102 decades | 0.02-0.07 decades |
| T4-style $D_2$ | 9.20-11.35 healthy, 2.98-3.89 loops | 8.69-11.75; no loops |
| TwoNN | 5.44-6.23 (all four validated runs) | 4.66-8.90, median 8.19 |
| MLE $k=10$ | 7.9-10.1 healthy, 3.1 loops | 10.97-15.87 |
| repetition loops | 9 full + 2 partial / 20; 6 full + 2 partial / 16 at 1000 tokens | **0 full + 0 partial / 13** |
| loop prominence | 0.04-0.22 full (validated runs 0.058/0.098), 0.21/0.45 partial, 0.91-0.99 healthy | 0.961-0.992 |
| loop coverage | 70-95% | 0% |
| nearest-twin median ($d_{typ}$) | 0.016-0.017 loops, 0.419-0.538 healthy | 0.527-0.670 |
| old headline | 5.75 raw / 5.69 shuffled (max-$R^2$ chords) | no comparable headline; chords 8.10-10.27 |

### 5.2 (i) Does the 14B show a valid scaling region where the 1.5B did not?

**No.** Not one of the 13 passes the rule, and the reason is the same one that made the 1.5B healthy runs fail: the interval between the depletion floor $10/N$ and saturation $C=0.1$ is only 0.08-0.11 decades, sampled by 4-5 grid points, while the rule needs 0.40 decades and 5 points. The 14B is if anything *more* decisively windowless: its T4-style best fits are 0.02-0.07 decades wide (1.5B: 0.039-0.102), and its MLE at $k=10$ (11-16) is above the 1.5B healthy range (7.9-10.1). The only window ever found on the 1.5B was the **spurious** 0.54-decade, $\hat D_2=1.09$ window on the period-53 loop `cyber`; the 14B has no loop, so there is no counterpart, spurious or otherwise.

This is not a claim that the 14B cloud is "more random" or "higher-dimensional" in a dynamical sense. It is a statement about resolution: at $N=1000$ and $d=5120$, the 1%-to-10% range of the distance distribution is 20% wide, and no estimator can extract a 0.4-decade slope from it. The 1.5B at $d=1536$ has the same 0.08-decade range, so the resolution limit is not set by $d$ alone; it is set by the shape of the state cloud and by $N$.

### 5.3 (ii) Are the 14B trajectories more or less prone to repetition loops?

**Less, by a wide and screen-independent margin.** Counts first.

- 1.5B: 9 full loops + 2 partial out of 20 local trajectories. At the same 1000-token horizon: 6 full + 2 partial out of 16.
- 14B: 0 full + 0 partial out of 13 trajectories (0 of the 12 prompt topics, with interstellar counted twice).

The two nearest-neighbour definitions agree once their scales are matched, and neither is a threshold trick.

**The GPU screen (`summary.json`).** It takes the last three quarters of the states, $h[250{:}]$ for $N=1000$; computes all pairwise distances; excludes $\lvert i-j\rvert<50$; and reports the median and p05 of the nearest remaining distance divided by $\mathrm{median}_i\lVert h_i-\bar h\rVert$, the distance to the centroid. That is a *centroid* scale, not the pairwise $d_{typ}$ used by the laptop pipeline, so the numbers are not directly comparable to the 1.5B's 0.016 $d_{typ}$ until rescaled. The measured values are median **0.728-0.996** and p05 **0.355-0.557**.

I recomputed this statistic from the `.npz` states and reproduced the GPU values for 12/13 rows to all printed digits (e.g. `great_emu_war` 0.728/0.377, `transformers` 0.990/0.547). The 13th, `interstellar_3000`, reads 0.875/0.422 in `summary.json` because the GPU used the full 1500 states starting at index 375 (1125 points), while the truncated analysis uses 1000 states starting at index 250 (750 points); recomputing on the full run gives exactly 0.8751/0.4216. This is a definitional difference from truncation, not a disagreement.

**The laptop screen.** It uses all $N$ points, the same $\lvert i-j\rvert\ge50$ exclusion, and $d_{typ}$ (median rms pairwise distance) as the scale; it reports the *fraction* of points below $10^{-3}d_{typ}$, plus median and p05. The values are median **0.527-0.670**, p05 **0.270-0.369**, fraction below $10^{-3}$ **0.000** in every run. The ratio between the two conventions is 1.38-1.55, roughly $\sqrt2$, as expected when a centroid distance is compared with a pairwise one. Within either convention the 14B is far from a loop: the 1.5B loop runs' nearest-twin median was 0.016-0.017 $d_{typ}$ (about 0.023 in centroid units), a factor **31-43** below the 14B medians (0.527-0.670 $d_{typ}$ / 0.728-0.996 centroid), and the 14B's lowest p05 (0.270 $d_{typ}$ / 0.377 centroid) is still 16-17 times the 1.5B loop median.

**Reconciling the two screens is also a warning about the $10^{-3}$ threshold.** The absolute-threshold screen returns 0.000 for the 1.5B loops too, because those loops *drift*: their repeats sit at 1.6% of the cloud scale, not at $10^{-3}$. So the "fraction below $10^{-3}$" is not the loop detector, and the fact that it is zero here would not by itself prove no loops. The detector that found the 1.5B loops is the lag-distance prominence; the GPU `rel_nearest` median is a second, cruder version of the same signal (a near-recurrence at some lag lowers the nearest-neighbour distance). On the 14B, the prominence screen is at the top of its range (0.961-0.992) and the GPU nearest-neighbour median is at 0.73-1.00, so both agree: no repeating trajectory. The two statements are consistent because they measure the same near-recurrence, one through the minimum over lags and one through the nearest neighbour among all $j$ with $\lvert i-j\rvert\ge50$.

### 5.4 What the point-cloud values say about the 14B

The 14B is not a lower-dimensional object than the 1.5B; if anything the point-cloud estimates are higher. TwoNN's median (8.19) is above the 1.5B's 5.44-6.23, and the MLE at $k=10$ (11-16) is above the 1.5B healthy range (7.9-10.1). The chord $D_2$ partly overlaps the 1.5B healthy T4 range but is not the same estimator: it is a 4-point fit over a 0.08-decade interval, whereas the 1.5B T4 values came from 6-8 of 22 percentile grid points. The honest cross-model statement is therefore narrow: **both models' last-layer clouds are too high-dimensional and too concentrated for $N\sim10^3$ to resolve a correlation dimension; the 14B is at least as high-dimensional as the 1.5B by the nearest-neighbour and MLE estimators, and neither model has a valid scaling region on a healthy trajectory.** The 14B is clearly different in one respect: its greedy trajectories do not fall into the drifting repetition loops that affected 9 of the 20 1.5B runs.

---

## 6. What this does not show

1. **One layer, one horizon, one decoding mode.** Only the last layer, only 1000 generated tokens, only FP32 greedy. The earlier sections show that layer, window regime and prompt length change the state geometry; this section cannot say whether an earlier layer or a longer run has a plateau.
2. **Twelve prompt topics are not a distribution.** The 13 trajectories are 12 expository prompt topics (interstellar twice), several on adjacent topics; the absence of loops is a statement about these runs, not a loop rate for the model. The 1.5B comparison uses a different prompt list, so model size and prompt set are confounded.
3. **The 0.40-decade rule is a resolution requirement, not a physical law.** A cloud can be low-dimensional and still fail it if the distance distribution is narrow; the test is one-sided. The correct reading of "no scaling region" is "not measurable at this $N$", not "no low-dimensional structure exists". The extrapolation in section 3.3 puts the required $N$ at $\sim10^{4.5}$-10$^{5}$, which would still have to pass the flatness condition.
4. **Chords and T4 numbers are not dimensions.** The $8.10-10.27$ chord and the $8.69-11.75$ T4 value are quoted only so that the numbers in circulation can be recognized as chords. The report deliberately does not promote either to a $\hat D_2$.
5. **The loop screen is a state-recurrence test.** It can miss a loop with period $>600$ tokens in a 1000-token run, and it does not inspect the token text. A text-level repetition that does not return the state to $0.1d_{typ}$ would not be flagged, though the 1.5B loops did return that closely (70-95% coverage), so the screen is calibrated on the phenomenon it is meant to catch.
6. **Time shuffle is not a dynamics test.** It shows that the $W=0$ estimator is blind to order (exactly) and that $W>0$ responds weakly to the mild temporal smoothness. It does not show that the trajectory has no dynamics, only that the measured dimension is a static property of the cloud.
