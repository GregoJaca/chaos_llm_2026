# Findings, procedure, and transfer to other LLMs

Practical knowledge from the DeepSeek-1.5B pass. Keep with `DECISIONS.md` and `RESULTS_SUMMARY.md`.
Every claim below has a figure in `results/figures/` (see `INDEX.md`).

## 1. What worked

1. **FP32 + in-batch `r=0`.** Every reliable result came from this. Two identical `r=0` rows in the same batch must be bit-identical; if not, the run is invalid.
2. **Teacher forcing is an exact simulator up to the first flip.** One pass on the perturbed prompt, fed the unperturbed token sequence, reproduces (i) whether the free run flips, (ii) the exact step, (iii) the exact new token. Measured: 29/29 flips exact with 0 false positives and 0 misses in 140 cases. After the flip the prefixes differ, so teacher forcing says nothing about later tokens. This makes generation unnecessary for response/threshold work.
3. **Signed local threshold.** `m_t` (top1−top2 gap), `δm_t(u)`; a flip needs `δm_t ≤ −m_t`; hence `r_c⁻(t,u)=m_t·r/max(−δm_t,0)` and `T=min{t:r≥r_c⁻}`. **Measured at the radius being predicted** it is 318/320 (99.4%) correct over 19 experiments and 3 prompts (57/59 exact words, 0 false alarms, 0 misses). The sign-blind `|δm|` variant gives 21 false alarms on the same runs (89.1%) — that variant caused the earlier "many outliers" impression.
4. **The prediction is a survival forecast.** A flip is a point of no return: 0/75 flipped runs ever re-synchronise; mean token agreement afterwards is 2.7%; surviving fraction of the answer = first-flip step / 300. The 305 no-flip runs are 100% identical. So one cheap pass tells you whether the answer survives and how much of it.
5. **Linearity in `r`.** For `r ≤ 0.3`, `δh = r·G(t,ℓ)` (α=1.00±0.02, collapse CV 2–3%), and the same holds for single-token kicks (slope 0.99–1.00). Below `r≈3e-4` the FP32 floor dominates; above `r≈0.5` the response is superlinear and tie-dependent, so *extrapolating* a small-`r` threshold fails (16/20 at r=0.5, 6/20 at 1, 3/20 at 2).
6. **The window sets the shape, provably.** With one fixed token sequence and five windows (100/150/200/300/1000), each curve is bit-identical to the no-clipping control until exactly `t = window − prompt_len`, then departs: 35/85/135/235 for a 65-token prompt, 38/138 for a 62-token prompt. The generated text splits 4–6 tokens later. This is the cleanest causal control in the project.
7. **Two kinds of flips.** 17/27 observed flips happen at the response peak; 10/27 happen at a later close call (t=116/123/143) when the response has already fallen below 30% of its peak. A perturbation can sit latent for ~100 tokens.
8. **Single-token (current-state) kicks.** Linear in `r`; immediate size varies ×13 with position; half-life 1–7 tokens; a kick at the near-tie position s=16 changes a token 107 tokens later (validated autoregressively 4/4, plus 1/1 at s=50); only 4/24 directions at s=16 and 1/24 at s=50 change any token at r=0.1; a layer-14 kick is 9–36× weaker at the output than the same kick at the input.
9. **Directions are NOT equivalent.** Gains span ×7–12 at r=1e-3 and are stable across different continuations (rank corr 0.80–0.98). Shape differs beyond gain (6% of log-variance; split-half r=0.999). At the same close call, the signed margin change has opposite signs for different directions (7 shrink, 13 widen; only one crosses the flip line), so flips are stratified: 3/20 at r=0.5, all from the "hot" directions 10–12. No discrete clusters (k=1 wins, P(k>1)=0.10–0.18).
10. **Non-normality / rotation.** `σ_max>1` within a block while the same-position product contracts; median rotation 83.6°; the misalignment budget `(σ−1)cos²θ` predicts chained gains within ≲2×. This is a component-level statement; see the correction below.
11. **D2 shape artifacts are now explained.** The user's memory of "different, nicer" correlation-sum shapes is reproduced by five mechanisms: repetition loops, first-layer exact duplicates, m=2 delay embedding (fill dimension), first-PC saturation, and `cdist` self-pairs / linear ε grids. The estimator recovers Hénon (1.157) and Lorenz (1.984) cleanly, so the absence of scaling in LLM states is a property of the data.
12. **16 fresh 1000-token non-loop trajectories** were collected (2 batches, greedy, FP32, window 4096). The repetition screen flags none as loops (worst relative distance 0.135).

## 2. What did not work / was misleading

1. **"The depth expansion is radial scale drift; every block contracts."** Wrong as stated. The state's length grows ×452, but the perturbation is **86–87° from the state** (nearly perpendicular, not radial) and its relative size grows ×10 through the depth, shrinking only in the final band (×0.84). The same-position tangent product does contract, but it cannot see attention carrying the perturbation across positions. The "cosine ≈ 1" in `a1.npz` is a float32 floor artifact.
2. **Averaging first-flip time `T`.** It is a staircase pinned by near-ties. Use thresholds, survival curves and per-direction outcomes.
3. **Prompt perturbations below ~1e-4.** FP32/RMSNorm ill-conditioning creates a floor; the response stops scaling and `r_c` inflates as noise/`r`.
4. **The sign-blind `|δm|` threshold** (the version shipped in the first pass). It over-predicts and produced most of the apparent prediction errors.
5. **Extrapolating a small-`r` threshold to large `r`.** The gain grows ~1.55× by r=2 and becomes tie-dependent. Re-measure at the radius of interest.
6. **Depth-axis FSLE.** `lnρ/1`, flat in `r`: all growth is in layer 1. Use the depth spectrum instead.
7. **D2/RQA as evidence of chaos.** No valid scaling region exists (valid interval 0.00–0.04 decades wide); every quoted D2 is a window/tube/fill-dimension/duplicate artifact. Also, "raw ≈ time-shuffled D2" is a **tautology** for point-cloud statistics.
8. **"Directions are mostly equivalent."** True only for the shared backbone (token profile of ||δh||, corr 0.89–0.92) and for random current-state directions (≤2.1× gain spread). Prompt directions differ in gain, shape, and sign.
9. **Naive `⟨T⟩(r)` scaling law.** Wrong observable; use margins/thresholds.
10. **600-token A1 in one batch.** OOM on a 15 GB T4; 500 tokens works with `batch-dirs=1` and cache clearing. Batch >2 at 400 tokens also OOMs.
11. **Old `launch_sep`/`launch_aug` data.** Dense per-coordinate Gaussian, no seed, `repetition_penalty=1.1`: not comparable to simplex `r`.
12. **Phi bf16 token-level runs.** `r=0` diverges; invalid.
13. **Partial `r=0` control.** If no `r=0` row exists in the same batch, do not trust the run.

## 3. Procedure for a new LLM

### Calibration (cheap, 10–20 min)
1. Load in **FP32** if it fits; otherwise pick one precision and never mix.
2. Generate a baseline with **two identical `r=0` rows in one batch**; require bit-identical output.
3. Compute `m_t` for 300–600 tokens; record the smallest gaps and their steps: this sets the useful `r` scale.
4. Choose `r` from the FP32 floor to ~10× the smallest gap. For DeepSeek-1.5B: `1e-3…0.3` is the clean window.
5. Test one `r` above threshold and read the text. Large `r` is not automatically garbage.

### A1 (prompt response)
1. 20 directions, 8–12 `r` values, layers `0,1,2,3,7,14,21,27`.
2. Save `margins`, `dm`, `r_c`, `pred_T`, `dh_norm`, `cos`, and full vectors for 3 directions.
3. Check `δh/r` collapse; fit α only in the clean range.
4. **Window experiment:** generate one token sequence with a large window, then re-run the same sequence with 4–5 smaller windows. Departure must occur at `t = window − prompt_len`. This is the strongest control and costs one forward per window.

### Threshold
1. Compute `r_c⁻`; `r_* = min_t r_c⁻`; `T = min{t: r ≥ r_c⁻}`.
2. Validate with free runs at 2–3 radii that include flips and non-flips.
3. Report per-direction outcomes, a danger ladder (sorted thresholds, controlling word, sign blind vs signed), and the survival fraction.
4. Say explicitly whether a number is "measured at that radius" or "extrapolated".

### Teacher forcing vs autoregressive
1. One teacher-forced pass; compare predicted first changed token with a free run.
2. Expect exact agreement up to the first flip and nothing after it.
3. Use teacher forcing for response/threshold work; use free runs only for post-flip behaviour.

### State / single-token kicks
1. Inject at the embedding of positions spanning the sequence (8–200), 24 directions, 3–4 radii, measure 4 layers.
2. Report immediate size, half-life, depth ratio, and whether any token changes (teacher-forced check), then validate the few that do with a free run.
3. Expect: position matters more than size; flips only at close calls; latency can be ~100 tokens.

### Depth/non-normality
1. Depth spectrum k=8–16, QR per layer, per token; save absolute and state-normalised exponents.
2. `σ_max` by power iteration, `ρ` by ARPACK, principal angles by SVD.
3. **Also measure the response's angle to the state** using saved full vectors (not the stored cosine): if the angle is near 90°, the response is a direction change, and "radial drift" language is wrong.
4. Report the component-level and full-response statements separately.

### D2/RQA
1. Chunked distances, log ε grid, N convergence, Theiler.
2. Mark the valid window `[10/N, 2 log10 N]` on every plot; if it is empty, say "no scaling region".
3. Always run IAAFT/phase surrogates; never use time-shuffle as a dynamic control.
4. Screen trajectories for repetition loops before analysis.

## 4. Compute/resource notes

- One GPU job at a time; run queues with `nohup setsid bash queue.sh` so a dropped ssh does not kill the work. Wait on `pgrep` rather than sleeping manually.
- FP32 DeepSeek-1.5B: ~7.1 GB weights. Safe A1: batch 1–2, ≤500 tokens, cache clearing between batches. Batch ≥3 at 400 tokens OOMs on a 15 GB T4.
- Teacher forcing is ~20× cheaper than generation for the same number of directions.
- Depth/non-normality: power/ARPACK, never dense eig on GPU.
- CPU analyses (D2, depth, plots) run in parallel on a laptop with numpy/scipy/matplotlib.
- For a new model, rescale `r` with the embedding/activation scale and re-calibrate gaps; do not copy DeepSeek's `r`.

## 5. User feedback to keep in mind

- Prefer longer, deeper runs over many tiny results; keep the GPU busy but avoid OOM.
- Every claim needs a figure that shows the parameter dependence.
- Figures: separate y-axis per panel; define every axis; 3 individual examples + aggregate; say what separates the curves.
- Avoid jargon ("collapse", "radial scale drift") without a plain definition.
- Use subagents (GLM 5.3 preferred) for parallel non-GPU analysis and let them explore.
- Write down positive and negative results; keep a procedure for other LLMs.
- Share plots progressively; the user is the final judge.

## 6. Exploratory negatives

- **Attention entropy** has no clean relation to near-ties (+0.32–0.38 with margin at w100, ~0 at w200).
- **Lag collapse of `R(t,s)`** fails (×1.5–1.6 RMS) even with much more data.
- **Direction clustering** at r=1e-3: no clusters (k=1 wins).
- **D2 on repetition loops:** low values are degeneration.
- **600-token A1 in one batch:** OOM; 500 works.
