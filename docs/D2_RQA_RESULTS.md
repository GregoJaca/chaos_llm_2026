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

## Results

| data | D2 (N≈2000) | Eckmann–Ruelle bound | verdict |
|---|---:|---:|---|
| first-layer trajectory | 13–27 | 6.6 | **not reliable** (above bound) |
| last-layer trajectory | 2.97–11.31 | 6.6 | some below, some above; no stable plateau |
| vocabulary matrix (5k rows) | 5.35 | 7.4 | reference; not a trajectory |

- No stable plateau in the local slope for first/last-layer trajectories. The slope rises to a peak and falls; fitting it gives a number but not a scaling regime.
- First-layer D2 is high-dimensional, as expected for embeddings.
- Last-layer D2 is lower, but it is not clearly below the vocabulary null and does not converge consistently.
- Theiler windows 0/10/50 do not change the conclusion.
- RQA: DET 0.37–0.95, Lmax up to 1775; Theiler 20 barely changes DET. This is dominated by temporal correlation and near-repeats, not evidence of deterministic chaos.

## Interpretation

The old claim “last layer looks low-dimensional / chaotic, first layer looks random” is **not supported** by a proper plateau/convergence analysis. The honest statement is: with the accessible `N` (2–5k) and these trajectories, we cannot estimate a robust low correlation dimension. This is a negative result, and it should be reported as such. It also means D2/RQA should not carry the main chaos claim.

## Files

- `src/chaos_llm/analysis/dimension.py`: `correlation_curve`, `fit_plateau`, `d2_curve`, `rqa`.
- `scripts/run_d2.py`: per-prompt D2, convergence, vocabulary null, RQA.
- `scripts/d2_theiler.py`: Theiler sensitivity.
- `results/d2/d2_rqa.json`, `results/d2/d2_theiler.json`.
- Figures: `results/figures/d2_curves.png`, `d2_vs_N.png`, `d2_distribution.png`, `d2_theiler.png`.

## What would change the conclusion

- Much longer trajectories (`N≫10^4`) and a genuine plateau over ≥1 decade.
- A surrogate test showing the data slope differs from phase-randomized/IAFFT surrogates.
- A local dimension estimator (TwoNN/GRIDE/EVT) that agrees with D2.
- A per-prompt paired comparison showing last-layer D2 systematically below first-layer and vocab.

Until then, treat D2 as a descriptive, not conclusive, diagnostic.
