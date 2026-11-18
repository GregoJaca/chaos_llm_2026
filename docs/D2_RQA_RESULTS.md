# D2 and RQA: method and first results

Status: 2026-11-17. Data: `results_pecs` (12 prompts, first/middle/last hidden states, fp32), DeepSeek-1.5B.

## Method (corrected)

- **Centered Euclidean** distance on hidden states (subtract the mean).
- **Chunked** pairwise distances; no `N×N` matrix is stored. `N=2000–5000`.
- Epsilon grid = percentiles of off-diagonal distances (2–60%), 20–30 values.
- `C(ε)` with the diagonal excluded; Theiler window `W∈{0,10,50}` tested.
- D2 = slope of the **longest high-R² local-slope plateau**, not a fixed index slice.
- Convergence: D2 vs `N`; reliability bound `D2 ≤ 2 log10 N` (Eckmann–Ruelle).
- **Data policy:** per-prompt primary; vocbulary matrix as a null; no pooled-prompts single D2.
- RQA: target `RR≈2%`, line lengths with `l_min=2`, DET/LAM; Theiler on/off. Symmetric recurrence matrix implies LAM=DET, as expected.

## Critical data-validity correction

Three of the five long trajectories (`cyber`, `future`, `great`) are **near-exact repetition loops**:

| trajectory | period P | fidelity `median||x_t−x_{t+P}||/scale` | fraction recurring <5% | cycle-model variance |
|---|---:|---:|---:|---:|
| cyber | 53 | 1.8% | 90.6% | 92.1% |
| future | 74 | 1.8% | 85.8% | 87.4% |
| great | 69 | 1.7% | 85.6% | 84.9% |

These loops explain the low D2, high predictability and TwoNN≈6 on those prompts; a trivial `period + drift + jitter` synthetic reproduces all of it, while a statics-only phase-shuffled null predicts nothing (NSE≈−0.05). **They are repeated-text degeneration, not attractor geometry.** Exclude or regenerate them with anti-repetition settings before using D2.

The two **non-loop** trajectories (`interstellar`, `launch`) are the only honest tests.

## Results

| data | D2 (N≈2–3k) | Eckmann–Ruelle bound | verdict |
|---|---:|---:|---|
| first-layer | 13–27 | 6.6 | not reliable (duplicates) |
| last-layer loop prompts | 1.2–5.8 | 6.6 | repetition-loop artifact |
| last-layer non-loop (interstellar) | 5.3 | 6.2 | no plateau; finite-N inflation |
| last-layer non-loop (launch) | 5.1 | 7.0 | no plateau; one predictability lead |
| vocabulary (3k) | 5.04 | 7.0 | reference; not a trajectory |

- **No D2 plateau anywhere**: local slopes decline monotonically with ε (e.g. 9.0→5.3→1.1 for interstellar); small-ε slopes exceed the Eckmann–Ruelle ceiling; the “D2” value depends on the fit window.
- **Raw vs time-shuffled D2 is a tautology**: time-shuffling preserves the point cloud, and D2/TwoNN/MLE/PCA are point-cloud statistics. Max `|Δlog C|≤0.018` (≤0.0024 for C>1e-4); `interstellar`/`launch` are exactly identical. This control cannot test dynamics.
- Theiler windows 0→1000 change almost nothing; the correlation sum is dominated by loop copies (loopy prompts) or static cloud geometry.
- **Non-loop dynamics:** `interstellar` is a clean negative — best kNN NSE 0.329 equals its linear ceiling, and phase/IAFFT surrogates match longer horizons; no Kaplan–Glass continuation sharpness. `launch` shows a **suggestive** kNN NSE gap: 0.297 at h=8 vs −0.036±0.005 (phase), −0.034±0.011 (MIAAFT), +0.15 above its own ridge-PC50 ceiling; but there is **no continuation sharpness** and it is one trajectory. Treat as a lead, not a result.
- **Static geometry:** locally thin (TwoNN 3.3–6.7, MLE 2–4) but globally spread (90% variance needs 161/223 PCs for the non-loop prompts). That is inhomogeneous token-state geometry, not a low-dimensional attractor. TwoNN/MLE also have finite-N bias (iid-Gauss gives ~100), so only raw-vs-null contrasts are meaningful.
- RQA: DET 0.37–0.95, Lmax up to 1775; dominated by temporal correlation and loop copies.

## Interpretation

The old claim “last layer looks low-dimensional / chaotic, first layer looks random” is **not supported**. The low D2 on three prompts is repetition degeneration; the non-loop prompts show no D2 plateau, no dimension below finite-sample ceilings, and no dynamic signal beyond linear autocorrelation except one unconfirmed `launch` predictability lead. D2/RQA do not carry the main chaos claim.

## Files

- `src/chaos_llm/analysis/dimension.py`: `correlation_curve`, `fit_plateau`, `d2_curve`, `rqa`.
- `scripts/run_d2.py`, `scripts/d2_controls.py`, `scripts/d2_all_controls.py`, `scripts/d2_explore.py`, `scripts/d2_theiler.py`.
- `results/subagent_d2/REPORT.md`, `exp0`–`exp4` scripts/JSON.
- Figures: `d2_loops_scan.png`, `d2_scaling_subagent.png`, `d2_slopes_subagent.png`, `d2_synthloop_C.png`, `d2_synthloop_pred.png`, `d2_intrinsic_dim.png`, `d2_prediction.png`, `d2_continuation.png`, plus the earlier `d2_controls.png`, `d2_raw_vs_shuffled.png`, `d2_distribution.png`.

## What would change the conclusion

- Regenerate the three loopy prompts with anti-repetition settings and rerun D2/predictability.
- Replicate the `launch` kNN predictability gap on several non-loop trajectories, with continuation sharpness.
- A genuine C(ε) plateau over ≥1 decade, stable across N and Theiler, below the Eckmann–Ruelle ceiling.
- A statistic that separates ordered data from MIAAFT/phase surrogates **and** has sharp continuation; D2/TwoNN alone cannot, because they are point-cloud statistics.

Until then, treat D2 as a descriptive, not conclusive, diagnostic.
