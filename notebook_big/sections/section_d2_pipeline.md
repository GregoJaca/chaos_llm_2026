# Part D2. The correlation-dimension pipeline, run locally from distance matrices

## Summary

The D2 analysis is now split so that the GPU only does what it is needed for. A GPU job (`scripts_spark/d2_distances.py`) writes one full $[N,N]$ Euclidean distance matrix per trajectory; a laptop script (`scripts_spark/d2_local.py`) turns those matrices into the correlation integral $C(\epsilon)$, its local slope, the valid-window bookkeeping, TwoNN, the Levina–Bickel MLE and two loop screens — figures and JSON, no GPU, no model, no `torch`. Run time is **6–7 s per 4095-token trajectory** including building the matrix from the saved states, and **2–3 s** per trajectory at $N=4000$ from a precomputed `.npy` (1000 tokens: 1.1 s).

Validated against the four 1.5B trajectories that exist locally. The reference estimator of `results/d2/` is reproduced **exactly** — 2.978 vs 2.978, 3.894 vs 3.894, 9.195 vs 9.195 — which is the check that matters, because it means any disagreement below is about interpretation, not arithmetic. The interpretation does change: the reference "plateau" is a chord across 6–8 adjacent points of a percentile-spaced grid, **0.04–0.10 decades wide**, with the $R^2$ of 0.9999 that any smooth curve gives over such a span; and on the two healthy trajectories the whole eps range between the depletion floor $10/N$ and saturation is only **0.08 decades** wide, so no plateau can exist there by construction. The loop trajectories are identified by a lag-distance dip (period 53 and 69, dip to 6–10% of the neighbouring lags), and one of them produces a *convincing but spurious* 0.54-decade window with $\hat D_2=1.09$. Run over all 20 local trajectories the same test finds **nine repetition loops and two partial ones**, including six of the sixteen 1000-token runs that had been recorded as non-loops.

---

## 1. What the pipeline is

### 1.1 Input, output, command

| item | value |
|---|---|
| input (GPU) | `dist_<name>_h_last.npy`, fp32 $[N,N]$ Euclidean distance matrix, one per trajectory |
| input (GPU, optional) | `summary.json`: `n`, `dim`, `norm_median`, `norm_growth`, `rel_nearest`, `rel_nearest_p05` |
| input (laptop, validation only) | `.pt` (torch save, read here without torch) or `.npz` with `h_last` $[N,D]$ |
| output | `notebook_big/figures/d2pipe_<tag>_<name>.png`, `…_summary.png`, `d2pipe_<tag>.json` |
| command | `python scripts_spark/d2_local.py --dir results_big/ds14/d2 --theiler 0,50,500 --out notebook_big/figures --tag ds14` |
| alternative | `--pt <files or dirs>` (or `--dist <matrix.npy> …`) computes the matrices locally in float64 |

Every epsilon below is quoted in units of $d_{typ}$, the median over points of the rms distance to the other points (a robust typical separation). That makes a 1.5B trajectory (state norm ≈ 10²) and a 14B one comparable on the same axis.

### 1.2 Definitions

Let $d_{ij}=\lVert x_i-x_j\rVert$, $N$ points, Theiler window $W$.

| quantity | expression | note |
|---|---|---|
| correlation integral | $C(\epsilon)=\dfrac{\#\{(i,j):\ |i-j|>W,\ d_{ij}<\epsilon\}}{\#\{(i,j):\ |i-j|>W\}}$ | denominator $N(N-1)-2\sum_{k\le W}(N-k)$, so $C\to1$; for $W=0$ it is exactly $\frac{2}{N(N-1)}\#\{i<j:d_{ij}<\epsilon\}$ |
| evaluation | sort the upper triangle once, then `searchsorted` at each $\epsilon$ | 90 log-spaced values of $\epsilon/d_{typ}\in[10^{-2},2.5]$; the $W$-band counts are subtracted from the $k=1\ldots W$ diagonals |
| local slope | $\mathrm{d}\log C/\mathrm{d}\log\epsilon$ | central differences, second-order at the edges |
| TwoNN | $\mu_i=r_2^{(i)}/r_1^{(i)}$, $\ \hat d=1/\langle\ln\mu\rangle$ | MLE for the Pareto tail $P(\mu>M)=M^{-d}$; the reference code's OLS fit of $-\ln(1-F)$ vs $\ln\mu$ is the same estimator and is returned too |
| Levina–Bickel MLE | $m_k(i)=\frac{1}{k-1}\sum_{j<k}\ln\frac{d_k(i)}{d_j(i)}$, $\ \hat d_k=\langle 1/m_k\rangle$ | $k=5,10,20$ |
| repetition screen | fraction of points with nearest distance at $\lvert i-j\rvert\ge50$ below $10^{-3}d_{typ}$ | plus the median and p05 of that relative distance |
| loop-period screen | $p(k)=\overline{\lVert x_t-x_{t+k}\rVert}/d_{typ}$; prominence $=p(k)/\mathrm{median}\,p(k\pm1,2,3)$ | a drifting loop has a sharp dip at its period; a healthy trajectory has none |

### 1.3 The valid-window rule (this replaces the max-$R^2$ search)

A window is marked valid only if all three hold simultaneously:

1. **flatness.** The longest run of $\epsilon$ in which the local slope stays within $\mathrm{tol}$ of its own median, with $\mathrm{tol}=\max(0.15,\ 0.06\,\lvert\hat D_2\rvert)$ — i.e. "within 6% of its own value", with an absolute floor so low dimensions are not judged in relative units — and the run at least **0.4 decades** wide. The test is applied to a 3-point median filter of the slope so a single grid point cannot open a window.
2. **depletion floor.** $\epsilon$ above the point where $C(\epsilon)=10/N$ (at least $\sim5N$ pairs inside the ball). This is the $10/N$ edge of `docs/FINDINGS_AND_PROCEDURE.md`.
3. **below saturation.** $C(\epsilon)\le0.1$.

The window width in decades is reported whether or not it is valid, together with the slope at both edges of the admissible range, so "no plateau" is a number and not a verdict. The finite-$N$ slope ceiling $2\log_{10}N$ (7.2 at $N=4095$) is drawn on every slope panel. If the window is empty the verdict is *no scaling region*; the pipeline also reports $\hat D_2$ over the whole admissible range ($C$ from $10/N$ to $0.1$) so the window dependence stays visible.

## 2. Validation on the 1.5B trajectories

Four trajectories: two of the local long runs are period-53 / period-69 loops (`cyber`, `great`, both 4095 tokens at the last layer), two are healthy (`interstellar`, 1026 tokens; `astronomy_observations`, 1000 tokens). All four are 1536-dimensional last-layer states, greedy FP32, window 4096.

![pipeline summary](figures/d2pipe_validation_summary.png)

| trajectory | $N$ | loop period (coverage) | nearest-twin median | $\hat D_2$ valid window | window width | T4 estimator (reproduced / published) | TwoNN | MLE $k{=}10$ |
|---|---:|---|---:|---|---:|---|---:|---:|
| `great` (loop) | 4095 | **69** (90%) | 0.016 | — (none) | 0.00 dec | **2.978 / 2.978** | 5.44 | 3.19 |
| `cyber` (loop) | 4095 | **53** (94%) | 0.017 | **1.09** (spurious) | 0.54 dec | **3.894 / 3.894** | 6.21 | 3.12 |
| `interstellar` (healthy) | 1026 | — (0%) | 0.419 | — (none) | 0.00 dec | **9.195 / 9.195** | 6.23 | 7.91 |
| `astronomy_observations` (healthy) | 1000 | — (0%) | 0.538 | — (none) | 0.00 dec | 11.35 / — | 6.12 | 10.13 |

Four things to read off it.

**The reference numbers are reproduced exactly.** The T4 estimator column is `dimension.py:d2_curve` run on the first 2000 states with Theiler 20, i.e. the code path that produced `results/d2/d2_raw_vs_shuffled.json`; it matches to the published four digits (2.9777, 3.8943, 9.1947). So the correlation integral and the fit search here are the same objects as in the reference implementation, and the rest of this section is a statement about what those numbers mean, not about whether they were computed right.

**The healthy trajectories have no scaling region, and there is no room for one.** For `interstellar` the admissible range runs from $\epsilon/d_{typ}=0.61$ ($C=10/N$) to $0.81$ ($C=0.1$) — **0.08 decades** — and across it the local slope climbs 6.1 → 8.9. A 9-dimensional cloud with 1000 points simply cannot be measured: the depletion floor and saturation are 0.08 decades apart, which is why the whole "valid interval" in the earlier analysis was 0.00–0.04 decades wide. The T4 value 9.195 comes from a window 0.102 decades wide, 6 of 22 grid points, inside which $C$ rises by 0.14 (a seventh of the full range): a chord, not a plateau.

**Loops give a low number, and one of them gives a *valid-looking* one.** `cyber` has a 0.54-decade window with $\hat D_2=1.09$ that passes all three criteria — the drifting period-53 orbit really does trace a one-dimensional curve with $C\sim\epsilon^{1.1}$ over that range. The verdict is decided by the loop screens, not by the fit: 94% of its positions return to within $0.1\,d_{typ}$ after 53 tokens (90% for `great`), against 0% for both healthy trajectories, and the lag-distance dip is 0.06 of its neighbouring lags. `great` has the same structure but no flat run at all. The correct reading of a low $\hat D_2$ is therefore always *"is this a loop?"* first.

**What does not separate the groups:** TwoNN sits at 5.4–6.2 on all four, loops included, and the $k$-neighbour MLE at $k=10$ agrees with the truth on the healthy pair (7.9, 10.1) but is 3.1–3.2 on the loops — both are point-cloud statistics and neither is a loop detector. The repetition screen at the $10^{-3}$ threshold in the brief finds **nothing** in any of the 20 local trajectories: these loops drift, so a repeat sits at 1.6% of the cloud scale, not at $10^{-3}$. The usable screen is the lag-distance dip: over all 20 local trajectories the prominence is **≤0.22 for the nine full loops, 0.21 and 0.45 for two partial ones, and 0.91–0.99 for the nine clean ones** — a factor-2 margin with no free parameter. There is no smoothness confound in this metric: consecutive last-layer states are already 0.84–1.03 $d_{typ}$ apart, so the lag curve is flat at $\approx1$ for a healthy run and only a genuine return produces a dip.

Theiler windows change nothing for the healthy trajectories (no window at $W=0$ or $W=500$) and remove even the spurious window of `cyber` at $W=500$; the same run at $N/2$ and $N/4$ loses it too, so it is a small-$N$ feature of the orbit.

### 2.1 The screens applied to all 20 local trajectories

The same two screens, run over every 1.5B trajectory that exists on this machine (three 4095-token prompt runs and the sixteen 1000-token runs, plus `interstellar`), split them **nine loops / two partial / nine clean**:

| period found | coverage | trajectories |
|---|---:|---|
| 12, 17, 27 | 95%, 76%, 89% | `renewable_energy_economics`, `history_of_music`, `sleep_and_memory` |
| 53, 63, 69, 74, 81 | 94%, 77%, 90%, 91%, 70% | `cyber`, `history_of_computing`, `great`, `future`, `ocean_currents` |
| 53 | 80% | `cybersecurity` (the 1000-token run of the same prompt as `cyber`) |
| 50, 79 | 39%, 50% | `transformer_architecture`, `urban_transportation` (partial: prominence 0.45, 0.21) |
| — | 0–5% | `interstellar`, `astronomy_observations`, `childhood_experiences`, `evolution_of_language`, `healthy_habits`, `history_of_mathematics`, `interstellar_travel`, `software_engineering_practices`, `volcanoes_and_plate_tectonics` |

This contradicts the earlier note that the sixteen fresh 1000-token trajectories "flag none as loops (worst relative distance 0.135)". The two statements are reconcilable only if that screen thresholded something other than the *fraction of positions* that return — for `renewable_energy_economics`, 95% of positions come back to within $0.1d_{typ}$ after 12 tokens, with a mean lag-12 distance of $0.037d_{typ}$ against $1.06d_{typ}$ at the surrounding lags, so the loop shows up as a *dip* that a median or a maximum over points can easily miss. Whatever the earlier screen was, **six of those sixteen trajectories (plus two partial ones) are repetition loops and their D2/MLE numbers are not usable as non-loop controls.** The twelve trajectories behind `results/d2/d2_raw_vs_shuffled.json` live on the T4 and cannot be screened here; the three that are loops were correctly identified there, and the other nine need this check before their $\hat D_2$ of 9–11 is quoted.

## 3. Three problems with the earlier estimator, worth knowing

Everything in `results/d2/` was produced by `fit_plateau`, which slides over every sub-interval of the $\epsilon$ grid and keeps the one with the largest $R^2$. Three consequences, all visible in the numbers above:

1. **The reported $R^2$ is a maximum over ~200 candidate windows, not a goodness of fit.** A smooth log–log curve gives $R^2>0.9999$ over two or three adjacent points, so quoting 0.9998 as evidence of a plateau is circular. The winning windows here are 0.039, 0.060, 0.083 and 0.102 decades wide.
2. **`min_points=6` counts grid points, not decades.** The $\epsilon$ grid is made of percentiles of pairwise distances, so its spacing is very uneven (one 0.19-decade jump near the 2nd percentile, then ~0.01-decade steps near the 60th). Six adjacent points can be 0.02 or 0.3 decades; the width is never checked, so the "plateau" can be one grid step.
3. **The grid itself depends on $N$.** Percentiles 2–60 of a 1000-point subsample move with the sample, so the quoted $\hat D_2$ changes with $N$ both through the data and through the grid — which is why the earlier N-convergence check could not separate the two. (Two smaller ones: `d2_all_controls.py` ran the raw curves at Theiler 20 and the shuffled ones at Theiler 0, so even the tautological raw-vs-shuffled comparison was not matched; and `rqa` passes `theiler=0` to `correlation_curve` regardless of its argument, using the argument only for the recurrence mask.)

TwoNN is implemented correctly there (the CDF slope fit is the Pareto MLE), but the widely quoted variant $\log 2/\langle\log\mu\rangle$ is **not**: it returns $\ln^2 2\cdot d=0.48\,d$, which was checked on uniform hypercubes of dimension 2, 5 and 10. The pipeline returns both so the difference stays visible. A related trap: exact duplicates give $r_1=0$, so $\mu=\infty$ and the point is dropped — the published `vocab` control lost 16 of 2000 points this way, and a bit-exact loop would hide its own degeneracy.

## 4. Known artifacts, in plain language

- **Repetition loops.** A model that repeats a phrase produces states that revisit almost the same point every period. The point cloud then contains a thin closed curve, and $C(\epsilon)\propto\epsilon^{1}$ over a wide range: a low, clean, completely spurious $\hat D_2$. Screen first, fit second.
- **First-layer exact duplicates.** At the embedding output, equal tokens give bit-identical vectors. That is a pile of exactly coincident points, which inflates $C$ at small $\epsilon$ and depresses the slope; the earlier fix was to de-duplicate, which changes $N$ and destroys the comparison with the last layer.
- **Fill dimension from delay embedding.** Building vectors $(x_t,x_{t+1},\ldots,x_{t+m-1})$ and measuring their dimension reports roughly the embedding dimension, not the attractor's, whenever the data are smooth in time. That is why the delay-D2 rose 5.9 → 6.3 → 11.1 → 14.9 → 17.6 as $m$ and $\tau$ grew.
- **First-PC saturation.** If one direction carries most of the variance, distances saturate as soon as $\epsilon$ reaches it, and everything above that is a step, not a slope.
- **Self-pairs and linear $\epsilon$ grids.** Counting $i=j$ adds $N$ zero distances and drags the slope at small $\epsilon$; a linear $\epsilon$ grid concentrates all points where nothing happens. Both are removed here (upper triangle only, log grid).

All five mechanisms were reproduced by the earlier pass on synthetic data, and the estimator recovers Hénon ($\hat D_2=1.18$ over a 1.1-decade window) and a uniform 3-cube (2.82, no valid window at $N=1500$ because the admissible range is 0.40 decades) — so the absence of scaling in the LLM states is a property of the data, not of the estimator.

## 5. Running it on the new matrices

```
# on the laptop, once the GPU job has written results_big/<model>/d2/
python scripts_spark/d2_local.py --dir results_big/ds14/d2 \
    --theiler 0,50,500 --out notebook_big/figures --tag ds14
```

`summary.json` from the GPU side is picked up automatically and its `rel_nearest` is stored next to the laptop-side screen, so the two can be compared for every trajectory. Per-trajectory figures carry eight panels: $C(\epsilon)$ for all Theiler windows with the valid window shaded, the local slope against the $2\log_{10}N$ ceiling, the same slope at $N$, $N/2$, $N/4$, the TwoNN $\mu$ CDF, the MLE against $k$, the repetition histogram, the lag-distance loop test, and a numbers block. The JSON holds every quoted number including the T4-style reproduction and its window width.

**What this does not show.** One model, one layer (the last), one prompt set per trajectory. The valid-window rule has three thresholds (0.4 decades, 6%, 0.1 saturation) that were chosen before looking at the new data and are reported in the JSON for every run; the verdicts are robust to them here (the healthy trajectories have no admissible range at all, and the loops are caught by an independent period test), but a marginal case would need the width quoted, not just the verdict.
