# Findings, procedure, and transfer to other LLMs

This is the practical knowledge from the first DeepSeek-1.5B pass. Keep it with `DECISIONS.md` and `RESULTS_SUMMARY.md`.

## 1. What worked

1. **FP32 + in-batch `r=0`.** Every reliable result came from this. The old batch-1 baseline vs batch-N perturbed data was contaminated; the zero-perturbation control catches it.
2. **Teacher forcing for the smooth response.** One baseline forward + one forward per direction gives `δh_t`, `δm_t`, and thresholds without generation. This is the workhorse.
3. **Margins + signed local threshold.** `m_t`, `δm_t(u)`, `r_c⁻=m_t·r/max(−δm_t,0)`, `T=min{t:r≥r_c⁻}` predicted free-run flips across windows 100/200/500 and two prompts. The symmetric `|δm|` version is adversarial and over-predicts; use the signed version.
4. **Scaling collapse by `r`.** For `r≤0.3`, `δh=r·G(t,ℓ)` with `α=1.00±0.02` and collapse CV 2.0–3.3%; `√r` does not collapse. No time-dependent exponent is needed. Deviations at `r=1e-3` are FP32 floor; at `r≥0.5` the response is mildly superlinear.
5. **The regimes are window-driven.** Shallow layers go bitwise zero at t*=92/191/290 (window−1); deep layers rise to t≈17, surge pre-flush, then decay. w100 jumps ×1.73 at t=35 and w200 ×1.51 at t=135, with wider windows as byte-identical controls. Depth gain `A(ℓ)~ℓ^0.64` (×12 from L1 to L27), front-loaded.
6. **Directions: same shape, scalar gain.** Gains span ×1.13–18.97 at L27 (dir10 strongest), depth-stable; one mode explains 72% of log-variance, but the response is not rank-1 (direction-specific token oscillations).
7. **Margin spectrum vs window.** Window 50 has no fragile steps; 100 has one at t=123; 200/500 have several (t=44/55/…) and many margins <0.05. This is a real attention effect.
8. **Scale-normalized depth contraction.** Absolute depth growth is residual-stream scale drift (×30 block 0, ×1.05–1.18/layer). Normalized by the state scale, every block contracts; max relative exponent −0.034. FSLE ×206 absolute becomes ×0.46 relative; h_depth is gauge drift. Report absolute and relative.
9. **Non-normality / rotation.** `σ_max>1` while the product contracts; median rotation 83.6°; misalignment budget `(σ−1)cos²θ` predicts chained gains within ≲2×. This is the quantitative composition answer.
10. **Shuffled control for D2.** The low last-layer D2 (~5–7) is static geometry: time-shuffled data gives the same D2. Strong negative.
11. **State direction equivalence.** PC1 explains 94–97% of log-curve variance; median direction correlation 0.94–0.98; no clusters/special directions. `R(t,s)` is not a function of lag (collapse fails ×1.6). Layer-14 injection is 9–36× weaker and plateaus at ≈2.9e-4.
12. **Signed threshold limit.** Predictions 19–20/20 at `r≤0.1`, 17–18/20 at `r=0.3`, 15/20 at `r=0.5`, 9–12/20 at `r=1–2`; gain grows ~1.5× by `r=2`, so do not extrapolate beyond `r≈0.3`.

## 2. What did not work / was misleading

1. **Averaging first-flip time `T`.** It is a staircase pinned by near-ties. Use survival/hazard/threshold distributions.
2. **Prompt perturbations below ~1e-4.** FP32/RMSNorm ill-conditioning creates a response floor; the result stops scaling. The usable range is roughly `1e-3…0.1`; `r=1–2` is large but still coherent at short horizons.
3. **Depth-axis FSLE.** It is `lnρ/1` and flat in `r`: everything happens in layer 1. The depth spectrum is more informative than the FSLE.
4. **Depth expansion as “chaos.”** The absolute top exponents are positive, but this is residual-stream scale drift: normalize by the state norm and every block contracts (max relative exponent −0.034); the final RMSNorm projects it out. The depth FSLE ×206 absolute is ×0.46 relative. Do not sell absolute depth growth as sustained chaos.
5. **D2/RQA as evidence of chaos.** At N=2–5k there is no D2 plateau; small-ε slopes exceed finite-N ceilings; Theiler windows do not help. Three of five long trajectories are near-exact repetition loops (P=53/74/69), so their low D2 is a degeneration artifact. Raw vs time-shuffled D2 equality is a **tautology** (shuffling preserves the point cloud). RQA DET is dominated by temporal correlation/loop copies. Strong negative.
6. **One D2 lead, not a result.** The non-loop `launch` trajectory shows a kNN predictability gap +0.33 over phase/MIAAFT surrogates and +0.15 over its linear ceiling, but no continuation sharpness and only one trajectory. Treat as a lead; replicate on more non-loop prompts.
7. **Naive `⟨T⟩(r)` scaling law.** The plan’s log-vs-power-law test is not the right observable; measure margins and thresholds instead.
8. **600-token A1 in one batch.** Full-vocab FP32 logits + all layers OOM on T4. 500 tokens works with `batch-dirs=1` and cache clearing; otherwise reduce tokens or directions.
9. **Old `launch_sep`/`launch_aug` data for sensitivity.** Dense Gaussian per-coordinate perturbation, no seed, `repetition_penalty=1.1`; not comparable to simplex `r`. Use only for qualitative RP/dimension.
10. **Phi bf16 token-level runs.** `r=0` diverges due to bf16 batch shape; the data is invalid.
11. **Partial `r=0` control.** If there is no `r=0` row in the same batch, do not trust the run.

## 3. Procedure for a new LLM

### Calibration (cheap, 10–20 min)
1. Load the model in **FP32** if it fits; otherwise choose one precision and never mix. For a T4, ~1.5B FP32 fits; ~3.8B does not.
2. Pick 2–3 prompts and a window; generate a baseline with **two identical `r=0` rows in one batch**; verify bit-identical.
3. Compute the baseline `m_t` for 300–600 tokens; record the smallest margins and their steps. This tells you the useful `r` scale.
4. Choose `r` grid from the numerical floor to ~10× the smallest margin. For DeepSeek-1.5B, useful `r≈1e-3…0.3`; start there.
5. Test one `r` above the expected threshold and inspect generated text (coherence, repetition). Do not assume large `r` is garbage; DeepSeek stays coherent at r=2/300 tokens.

### A1 (response/scaling)
1. Use 20 directions (simplex or random unit), 8–12 `r` values, layers `1,2,3,7,14,21,27` (and 0 for embedding).
2. Save norms/cosines, `m_t`, `dm`, `r_c`, `pred_T`. Full vectors only for 3 directions.
3. Fit `log δh` vs `log r` in the linear range; report `α(t,ℓ)`. Try `δh/r` and `δh/r^α` collapse.
4. Plot 3 individual directions + median/IQR; raw and normalized side by side; label the separating parameter (`r`, layer, time).

### Threshold/flux tube
1. Compute `r_c(t,u)`; `r_*(u)=min_t r_c`; `T(r,u)=min{t:r≥r_c}`.
2. Validate with free-run at 2–3 `r` (batch includes `r=0`); compare predicted vs observed flips.
3. Report `r_*` distribution, controlling near-tie, and a confusion matrix; avoid direction-index-only plots.
4. Repeat for 2–3 windows; the margin spectrum changes, so the threshold changes.

### State / R(t,s)
1. Inject at input embedding at s=16/50/100/150, 32 directions, `r=1e-3`; optionally inject at a middle residual layer.
2. Compute pairwise direction correlations; cluster. Expect high correlation; report it.
3. Plot `R(t,s)` vs lag; do not force a collapse if near-ties occur at different lags.
4. Compare input vs middle-layer injection; residual injection is weaker.

### Depth/non-normality
1. Depth spectrum: k=8–16 tangent vectors, QR per layer, per token; save absolute and **state-scale-normalized** `λ_i`, `h_depth`, local gains.
2. Full-block `σ_max` by power iteration, `ρ` by ARPACK; MLP dense `J` with power/ARPACK (do not use dense eig on GPU; 1536² eig ≈23 s).
3. Principal angles between top singular subspaces; expect large rotation.
4. Report relative exponents and the gain budget; do not overclaim if the absolute growth is radial drift.

### D2/RQA
1. Center, chunked distances, percentile epsilon grid, local-slope plateau, N convergence.
2. **Always run a time-shuffled control and a phase-randomized surrogate.** If raw ≈ shuffled, D2 is static geometry.
3. Never save N×N matrices. Save only eps/C/slope/summary.
4. Treat D2/RQA as descriptive unless they beat controls.

## 4. Compute/resource notes

- One GPU job at a time; use background with notifications; no polling.
- FP32 DeepSeek-1.5B: ~6 GB weights; A1 batch 1 at 500 tokens fits with cache clearing; 600 tokens OOM on T4 with full-vocab logits.
- Teacher forcing is cheap; generation is the expensive part.
- Depth/non-normality: power/ARPACK, not dense eig; k=8–16.
- CPU analyses (D2, depth FSLE, plots) can run in parallel; local laptop with numpy/matplotlib is enough for copied tensors.
- For a new model, scale `r` with the embedding/activation scale and re-calibrate margins; do not copy DeepSeek’s `r`.

## 5. User feedback to keep in mind

- Prefer longer, deeper runs over many tiny results; keep the GPU busy but avoid OOM.
- Figures: always think about which parameter separates curves; try to make curves overlap by normalizing; show 3 individual + aggregate; annotate what gray lines/parameters mean.
- Direction dependence and aggregation are hard; quantify, don’t hand-wave.
- Use subagents (GLM 5.3 preferred) for parallel non-GPU analysis; give them context and let them explore.
- Write down positive and negative results; keep a procedure for reproducing on other LLMs.
- Share plots progressively; the user is the final judge.

## 6. Exploratory negatives

- **Attention entropy as a near-tie explanation:** computed for w50/100/200, layers 7/14/21/27; correlation with margin +0.32–0.38 at w100, ~0 at w200; near-tie values mixed. No clean relation.
- **Lag collapse of R(t,s):** fails at ×1.5–1.6 RMS under amplitude/lag normalization; curves differ in kind.
- **Direction clustering at r=1e-3:** no clusters, no special directions, no cross-s persistence.
- **D2 on repetition loops:** low values are degeneration; exclude/regenerate.
- **600-token A1 in one batch:** OOM; 500 works with cache clearing.
