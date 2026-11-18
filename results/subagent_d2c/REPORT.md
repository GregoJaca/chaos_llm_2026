# D2 scaling and predictability on 16 fresh 1000-token trajectories

**Subagent D2c report — 18 Nov 2026 — CPU-only, float64, all numbers from the JSONs in this folder
(`d2_results.json`, `pred_results.json`, `delay_results.json`, `props_results.json`; scripts
`common_d2c.py`, `exp1_d2.py`, `exp3_predict.py`, `exp5_delay.py`, `exp6_props.py`,
`patch_exp1_miaft.py`, `aggregate.py`; figures fig1-fig6).**

Data: `results/local_data/long_last/*.npz`, 16 prompts x `h_last[1000,1536]` fp32 + `gen_ids[1000]`.

---

## Verdict on the two questions

**(a) Does any D2/RQA scaling exist at 1000 tokens? NO.** 0/16 trajectories have a valid scaling
interval at any config (N=500/1000, Theiler w=0/10/50): 0 of 64 (trajectory, config) runs contain a
slope-stable interval that is inside the Eckmann-Ruelle-legal band C(eps) in [10/N, 0.5] with
slope <= 2 log10(N). Every "stable" run the detector finds is one of three non-dynamical objects:
the saturation shelf at C ~ 1 (slope 0.00, e.g. healthy_habits 0.26 dec @ slope 0.00), the
count-noise depletion zone (interstellar 0.40 dec @ slope 1.51 at C ~ 1e-4), or the loop-tube edge
(cyber 0.22 dec @ 0.83 at C just below the 10/N floor). The paired aggregate: ER-legal band width is
**0.000 decades for all 8 non-repetitive trajectories** (same as the old 4000-token data: 0.00-0.04)
and 0.26-1.32 decades for the 8 repetition loops - but inside those bands the local slope still
drifts monotonically (loop knee/steps), so no run passes the stability test. Naive widest
r^2>=0.99 log-log fits exist on 15/16 (slopes 0.64-6.93, widths 0.40-1.10 decades) - the artifact
trap from d2b - and 15 of them live entirely below C = 10/N; the single ER-legal one
(sleep_and_memory: slope 0.64 over 0.48 dec at C in [0.014, 0.031]) is a period-27 loop, not a
scaling law. RQA DET at RR=10% is elevated (data 0.31-0.76 vs time-shuffle 0.18-0.20) but is
**bracketed by the second-order nulls**: phase surrogates (exact cross-periodogram) give 0.34-0.72
and MIAAFT (marginal-exact) 0.31-0.58; data - max(phase, MIAAFT) has median +0.017, range
[-0.20, +0.18] -> no independent determinism signal.

**(b) Does the predictability lead replicate? Only in its repetition-loop form - the "launch-style"
moderate gap does NOT replicate.** The 16 trajectories split cleanly into two classes of 8
(this also corrects the supplied repetition screen, see bullet 1):

| class | kNN NSE h=1 / h=8 | ridge h=8 | MIAAFT h=8 | gap vs best surrogate h=8 | h_half |
|---|---|---|---|---|---|
| **8 repetition loops** (cybersecurity, transformer_architecture, history_of_computing, renewable_energy_economics, history_of_music, ocean_currents, sleep_and_memory, urban_transportation) | **+0.93..+1.00 / +0.67..+1.00** | +0.39..+0.99 (med +0.73) | +0.16..+0.64 (med +0.51) | **+0.27..+0.67** (z vs MIAAFT up to 26) | never reached by h=32 |
| **8 non-repetitive** (interstellar_travel, childhood_experiences, healthy_habits, astronomy_observations, evolution_of_language, history_of_mathematics, software_engineering_practices, volcanoes_and_plate_tectonics) | +0.13..+0.39 / -0.07..+0.02 | -0.09..+0.04 (med -0.03) | -0.06..-0.02 (med -0.04) | **-0.04..+0.05** | 1.5-2.3 tokens |

The old launch gap was +0.33 above every surrogate and +0.15 above its own linear ceiling at h=8;
the best non-repetitive trajectory here reaches +0.053 vs MIAFT (volcanoes) and +0.042 vs its own
ridge (interstellar_travel). The clean null of the
old interstellar replicates exactly on the fresh interstellar_travel: kNN h=8 = +0.001, ridge
-0.041, MIAAFT gap +0.017. Paired across all 16 the kNN-vs-MIAAFT gap is significant at every
horizon (h=1: +0.66 median, p = 3.1e-5; h=8: +0.20, p = 0.001) but the distribution is bimodal,
carried entirely by the loop half - it is a repetition effect, not a within-trajectory chaos
signature. The linear part is fully second-order (ridge on data - ridge on phase surrogates:
median +0.017), and the loops beat even the exact-cross-spectrum phase surrogate, i.e. their
predictability is genuinely nonlinear - and genuinely repeated text.

---

## Bullets (all numbers exact, from the JSONs)

1. **The dataset is again two datasets, and the supplied repetition screen under-reports it.**
   My loop screen - min over P in [2, 250] (>= 50 pairs) of median_t ||x_t - x_{t+P}|| / median
   pairwise distance - finds **8 of 16 are period-12..81 quasi-loops**: renewable_energy_economics
   **0.018 at P=12** (12 unique tokens in 1000, H = 3.6 bits, PC participation ratio 4.8, 9 PCs for
   90% variance), history_of_music 0.019 @ P=17, sleep_and_memory 0.025 @ P=27 (top token = 33% of
   the stream), cybersecurity 0.035 @ P=53 (41 unique tokens, longest repeated n-gram = 219+,
   i.e. the whole stream is periodic), history_of_computing 0.047 @ P=63, ocean_currents 0.049 @
   P=81, urban_transportation 0.094 @ P=79, transformer_architecture 0.127 @ P=50. The other 8
   never recur (best distance 0.85-0.90 at the trivial lag 2). This contradicts the task's screen
   ("0.135 for renewable, >= 0.26 for the rest") - that screen should be re-run; under the
   definition above its numbers do not reproduce. renewable_energy_economics is indeed the extreme
   case and is treated separately everywhere below (red in every figure), but it is not alone:
   7 more prompts are loop-like, and they are precisely the 8 that carry every "positive" number.

2. **No D2 scaling region - quantified per trajectory (fig1).** On the 96-point log-eps grid
   (1e-3 to 3x median distance): non-repetitive 8 have ER-legal width **0.000 decades** (the whole
   C(eps) rise fits inside ~0.3 decades, slopes 6.6 at C=0.1 -> 4.0 at C=0.5, exceeding the N=1000
   ER ceiling of 6.0 below C ~ 0.2); repetitive 8 have ER-legal width 0.26-1.32 decades
   (renewable 1.32, music 1.17, sleep 1.10) but **no slope-stable run inside any of them**
   (widest legal-candidate stable run: 0.22-0.33 dec, all rejected - saturation shelf, depletion
   zone, or loop-tube edge). Aggregate: **0/16 valid**; naive r^2>=0.99 fits report D2 anywhere
   from 0.64 (sleep, loop) to 6.93 (software, depletion zone) - the "D2 is a window choice"
   pathology, now on fresh data.

3. **Time-shuffle preserves C(eps) exactly - verified to the last pair (fig2a).** With w=0 the
   pair-distance multiset is order-invariant, so the shuffled C(eps) equals the raw C(eps) as an
   integer count: max pair-count deviation = **0** over all 16 trajectories (this control can never
   reject statics; it is a tautology and we say so). At w=50 the temporal mask makes the statistic
   order-sensitive: max |dlog C| = 1.47 (median 0.26) because short-lag pairs (steps ~ 0.8x median
   distance) are systematically closer than long-lag pairs - a mild temporal-correlation effect
   present in both classes, not an attractor.

4. **C(eps) shapes are statics-determined (fig2b,c).** Cloud MIAFT surrogates (per-PC marginal
   exact, per-PC spectrum to 0.4%, top-50 cross-covariance preserved to rel err 0.09-0.46 per
   trajectory) reproduce the data's C(eps) mid-range; phase surrogates (exact cross-periodogram,
   Gaussianized marginals) steepen it - every curve an innocent pipeline fits is one of these two
   statics objects plus a loop knee. RQA DET (bullet in verdict): data elevated over shuffle but
   bracketed by phase and MIAAFT - median residual vs the stronger null +0.017.

5. **Predictability (fig3): the lead is real but bimodal and 100% repetition-driven.** Loopy 8:
   kNN NSE h=8 median **+0.967** vs ridge +0.727, MIAAFT +0.509, phase +0.556, shuffle -0.120,
   constant 0; gap vs the best surrogate class **+0.27..+0.67**; per-trajectory z vs MIAAFT 3.3-26.4.
   Non-repetitive 8: kNN h=8 median -0.017 (range -0.070..+0.018) with MIAAFT -0.040 and phase
   -0.036, i.e. the data is indistinguishable from its nulls; gap vs MIAAFT **+0.018 median / +0.053 max**, gap vs own
   ridge +0.002 median / +0.042 max - **the launch-style gap (+0.33 / +0.15) appears nowhere**.
   Persistence is strongly negative everywhere (-0.45 median at h=1: states jump ~0.8x the cloud
   scale per token), the drift indicator is only -0.05 (train-mean constant is fine), and the
   linear ceiling is second-order-explained (ridge data - ridge phase: +0.017 median, max +0.146).
   Skill half-life: non-repetitive 8 lose half their (linear-scale) skill in **1.5-2.3 tokens -
   the same timescale as their ridge and MIAFDT nulls**; the loops never lose half their skill by
   h=32 (their periods are 12-81 tokens). No horizon selectivity anywhere.

6. **Continuation sharpness: none (fig4).** Kaplan-Glass small-r rc/r at h=8 (Theiler 50, PC-50):
   data **1.40-2.28**, i.e. near pairs' futures are *more* separated than the pairs themselves;
   data - MIAAFT is **positive in 15/16** (+0.13..+0.87, sleep +0.87). The single exception is
   renewable's period-12 loop (rc/r = 1.087 vs MIAAFT 1.552, diff -0.47) - the only genuine
   "continuation" in the dataset, and it is a 12-token text loop with 12 unique tokens. No
   trajectory, loop or not, shows the rc << r signature of a deterministic flow.

7. **Delay embeddings (fig5): the m=2 trap persists; at m>=4 the data is thinner than its IAAFT
   null but never plateaus.** At m=2, tau=1: 11/16 show the textbook slope 1.79-2.06 over 0.6-1.8
   decades - matched by IAAFT and by same-marginal white noise (the 2-D fill dimension) - and
   **15/16 are not ER-legal at N~990** (the run dives below C = 10/N); the one legal "plateau"
   (history_of_music, 0.62 dec @ slope 0.82) is its period-17 loop. (The other 5 m=2 slopes,
   0.82-1.43, are loop-flattened, i.e. repetition, not geometry.) At m>=4 (Theiler w=m-1, correct
   for shared coordinates) data D2(m) sits **below the IAAFT band in 104/112 (traj, m) cells** - a
   genuine thinner-than-second-order embedding structure - but 14/16 keep growing with m toward
   the fill dimension (interstellar 1.69 -> 4.58 vs IAAFT 1.78 -> 5.14): no attractor plateau. The
   only two saturators are the loops renewable (0.81 -> **0.37**) and sleep (1.30 -> **1.01**),
   whose IAAFT nulls *diverge* (to 2.0 and 4.0) because the loop's coherent phase structure is
   exactly what spectrum-preserving surrogates destroy. Below-band != chaos: it is repeated-text
   determinism (see bullet 8).

8. **What makes a trajectory predictable: text redundancy, nothing else (fig6).** Spearman across
   16 vs kNN NSE(h=1): #PCs for 90% variance rho = **-0.90**, token unigram entropy **-0.89**,
   repeated-bigram fraction **+0.87**, longest repeated n-gram **+0.86**, unique-token fraction
   **-0.86**, loop period +0.72; vs the h=8 gap: top-token share **+0.86**, PC participation
   **-0.75**, norm CV +0.70, loop distance -0.66 (all p <= 0.006, n=16, no correction - the
   bimodality makes these conservative). "Less predictable" = lexically diverse, high effective
   rank (n90 = 160-259, PR 21-36), non-repeating text. State geometry adds nothing once repetition
   is accounted for.

9. **Methods notes (for reproducibility and two bugs found and fixed during the run).** (i) An
   earlier phase-surrogate bug multiplied the wrong Fourier axis and returned exact copies of the
   data; fixed and verified (surrogate-data corr ~ 0.0-0.1, cross-periodogram preserved exactly).
   (ii) A multivariate-IAAFT rank-match was applying the inverse permutation (marginal/spectrum
   still exact, cross-PC structure destroyed); fixed via argsort-of-argsort, verified by unit test
   and by lag-1 cross-covariance error dropping from ~1.0 to 0.09-0.46. All numbers above are from
   the post-fix runs. MIAAFT fidelity is reported per trajectory in the JSONs; the phase surrogate
   (exact cross-spectrum) is the second-order null, MIAFT (exact marginals) the marginal null, and
   time-shuffle the statics null; conclusions require beating all three, which only the loops do.

## Best figures

- **`fig3_predictability.png`** - the key positive/negative split: NSE(h) for a clean example and
  the top loop example with all baselines, plus per-trajectory gap bars colored by class: the
  launch-style gap is absent in the non-loopy half.
- **`fig1_d2_scaling.png`** - three example C(eps) curves (saturation shelf, depletion run, loop
  knee) with the ER-legal band shaded and the naive-fit trap annotated; histogram: ER widths 0.00
  (non-loopy) vs 0.26-1.32 (loops), 0/16 valid.
- **`fig2_d2_controls.png`** - the time-shuffle exact identity, MIAFT/phase statics reproduction,
  naive-fit diagonal, DET vs surrogate classes.
- **`fig5_delay_embedding.png`** - D2(m) vs IAAFT bands (below-band everywhere, growing with m),
  the m=2 fill-dimension trap with surrogates and white noise on top.
- **`fig4_sharpness.png`**, **`fig6_properties.png`** - continuation anti-sharpness; predictability
  vs redundancy correlations.

## What this settles (and what it leaves open)

The estimator suite recovers real structure when it exists (d2b's Henon/Lorenz controls), so these
are data verdicts, not pipeline verdicts: **at 1000 greedy FP32 tokens there is no D2 scaling region
on any of 16 prompts, and the only reproducible predictability lead is the repetition-loop class
(8/16), which is degenerated text, not chaotic dynamics.** The two old suggestions are now
resolved: interstellar's clean null replicates on a fresh realization, and launch's +0.15-0.33 gap
does not replicate on any non-repetitive prompt (max +0.05). The residual scientific content worth
keeping: (i) the below-IAAFT delay-embedding effect (104/112 cells) shows the ordered stream is
systematically thinner than second-order surrogates even on non-repetitive prompts - consistent
with mild recurrence/copula structure of token-conditioned states, not with an attractor; (ii) if
a chaos-style predictability claim is ever to be pursued, it needs anti-repetition sampling
(penalty/temperature) at >= 10^5 tokens, with the MIAAFT-vs-phase double null used here as the
standard of proof.

---

## One-paragraph summary for the main agent

On the 16 new 1000-token greedy-FP32 trajectories: **no D2/RQA scaling exists anywhere** - 0/16
trajectories have any slope-stable ER-legal C(eps) interval (N=500/1000, Theiler 0/10/50); the
non-repetitive 8 have ER-legal width 0.000 decades (slopes 6.6 -> 4.0 across the 0.3-decade rise,
over the ER ceiling), the repetitive 8 have wide ER bands (up to 1.32 decades) that are loop knees
with no stable slope, and every naive r^2 >= 0.99 "D2" (0.64-6.93) sits in the depletion zone or
on a loop; time-shuffle reproduces C(eps) to the last pair (exact integer-count identity, as it
must - it is a point-cloud tautology), MIAFT/phase reproduce the curve shapes, and RQA DET is
bracketed by those nulls (data - max(phase, MIAFT) = +0.017 median). **The predictability lead
replicates only as repetition degeneration**: my loop screen finds 8/16 are period-12..81
quasi-loops (renewable 0.018 @ P=12 with 12 unique tokens; music 0.019 @ P=17; sleep 0.025 @ P=27;
cyber 0.035 @ P=53; the supplied screen's "0.135/none" numbers do not reproduce under this
definition and should be re-run) - those 8 have kNN NSE 0.67-1.00 at h=8 with gaps +0.27..+0.67
over the best of MIAFT/phase/shuffle (z up to 26) and mostly-linear ridge ceilings, while **the 8
non-repetitive prompts are clean nulls (gap vs MIAAFT at h=8: +0.018 median, +0.053 max; vs their
own ridge +0.002 median, +0.042 max), so the old launch-style gap (+0.33/+0.15) does NOT
replicate**; interstellar_travel re-run replicates the old clean null exactly. There is no
Kaplan-Glass continuation sharpness anywhere (data rc/r exceeds MIAFDT in 15/16; the sole
exception is renewable's 12-token loop), no prediction-horizon selectivity (non-loopy skill
half-life 1.5-2.3 tokens = the ridge timescale; loops never halve by h=32), the m=2 delay-embedding
textbook curves are the fill dimension (15/16 non-legal at N~990), and at m >= 4 the data is
thinner than its IAAFT null in 104/112 cells but keeps growing with m - the only two saturators
(renewable 0.81->0.37, sleep 1.30->1.01) are the period-12/27 loops. Predictability tracks text
redundancy with |Spearman rho| 0.65-0.90 (top-token share +0.86, n90 -0.90, token entropy -0.89):
**recommendation: exclude or regenerate the 8 loopy prompts with anti-repetition sampling before
any further dynamics work; treat the below-IAAFT embedding effect, not D2, as the thing worth
chasing at >= 10^5 tokens; and note that two surrogate-implementation bugs (a Fourier-axis
phase-randomization error and an inverse-permutation rank match) were found and fixed during this
run - early MIAFDT-looking numbers elsewhere should be re-checked against the fidelity metrics
now stored in the JSONs.**
