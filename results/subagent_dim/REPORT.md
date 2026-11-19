# Dimensionality analysis of greedy LLM generation dynamics

**Scope.** Does the geometry/dimensionality of the representations explain any part of the observed perturbation dynamics (linear-in-r regime up to r≈0.3, contracting per-token maps with discrete flips, model-dependent downstream kick gain)? Four models with full per-layer data: DeepSeek-R1-Distill-Qwen-1.5B (`deepseek`, D=1536, L=28, untied), Qwen3-4B-Instruct-2507 (`qwen3_4b`, D=2560, L=36, tied), Phi-4-mini-instruct (`phi4mini`, D=3072, L=32, tied), Qwen3.5-2B (`qwen3_5`, D=2048, L=24, tied, hybrid linear/full attention). All computations local CPU, float32, layer-by-layer, <1 GB RAM.

**Deliverables.** `numbers.json` (every number below), `figs/` (103 PNGs, dpi≥170, `bbox_inches='tight'`), `dimlib.py` + `s0_check.py`, `s1_vocab.py`, `s2_hidden.py`, `s2b_figs.py`, `s3_logits.py`, `s4_cross.py`, `s4b_radial.py`, `digest.py`, `run_all.py` (full reproduction, ~1 h CPU). Caches in `cache/` let `s2b/s4/s4b` be re-run without recomputation.

---

## 0. Data forensics (done first; changes interpretation)

* `GE/GW` are **D×D feature Grams** (EᵀE), not V×V row Grams: `tr(GE)/Σ‖e_v‖² = 1.0000` for all 4 models. Pairwise cosines between vocabulary **rows** therefore cannot be read off the saved Grams; we compute them exactly on the uniform 1024-row sample (all 523,776 pairs, SE≈2e-4) and validate the row-cone with the identity `s₁² ≈ V·‖μ‖²` (ratio 1.09–1.10 for deepseek/qwen3_4b/qwen3_5; phi4mini 2.86, i.e. its top direction is *not* the mean). Column-pair cosines from the Gram (20k pairs) are ≈0 for all models and reported separately.
* The **top acts index (L) is the post-final-RMSNorm state**, not the final block: `logits = W_U·h_L` reproduces the recorded logit row norms with corr = 1−10⁻⁹ and mean ratio = 1.0000 for all 4 models (`s0_check.py`). The raw final block is **not** in the data. Hence the apparent last-step "norm collapse" (deepseek ‖h‖: 513→105; qwen3_4b 622→172) and the anti-aligned last "write" (cos(Δ,h) = −0.98) are the final RMSNorm, not a block. All depth plots mark this index with a star and it is excluded from block-write statistics.
* a1 `layers[ℓ]` ↔ acts index ℓ (verified via `norm_h0` at both ends). a1 `dh_norm` at layer 0 of generated positions ≈ **0** (the kick enters through the prompt's last position, not through the generated tokens' own embeddings), so the depth gain is defined as `G_ℓ = ‖δh_ℓ‖/r`.
* a1 `cos` stores **cos(h+δh, h)** (values ≈1), not cos(δh,h). Section D3 decodes it (see below). For 3 of 4 models it is saved on a ~4.9e-4 grid (6–11 unique values, ~67% exactly 1.0) and is therefore saturated.
* Everything else checked out: tied flags match `GE==GW` exactly; `sE² = eig(GE)` to 1e-8; layer-0 acts = embeddings (`norms == normE[gen_ids]` exactly, deepseek); `logits.ids == gen_ids[0,:128]`; `sample_E[j] = E[j·(V//1024)]`.

## 1. Four-model comparison table

**Vocabulary matrices (E; W_U in parentheses when different).** PR uses λ=s²; eff-rank = smallest k with Σλ≥90%; row-cos = mean pairwise cosine between random vocab rows (uniform sample, exact pairs); cone = ‖μ‖/mean‖row‖; α = tail slope of s_i~i^{−α} over i∈[10, D/2] (R²≥0.99).

| model | D | V | tied | PR(λ=s²) | stable rank | eff-rank90 | λ₁/Σλ | row-cos | cone | α |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek | 1536 | 151,936 | no | 13.3 (W: 9.3) | 4 | 1172 (W: 1140) | 0.273 (W: 0.327) | +0.274 | 0.51 | 0.13 |
| qwen3_4b | 2560 | 151,936 | yes | 136.1 | 12 | 2083 | 0.083 | +0.081 | 0.28 | 0.13 |
| phi4mini | 3072 | 200,064 | yes | 4.1 | 2 | 1021 | 0.490 | +0.166 | 0.54 | 0.38 |
| qwen3_5 | 2048 | 248,320 | yes | 22.3 | 5 | 1622 | 0.211 | +0.193 | 0.44 | 0.12 |

s₁/s₂ = 7.1 / 5.0 / 3.1 / 7.8: one dominant direction (the shared mean, `s₁²/V‖μ‖²≈1.1`) plus a smooth, genuinely high-dimensional bulk (90% of vocab variance needs 1021–2083 dims — there is **no** sharp rank). qwen3_4b is the near-isotropic outlier (row-cos 0.08, PR 136); phi4mini is the cone-iest (λ₁/Σλ = 0.49, and 48.3% of vocab norm² sits in u₁, |proj u₁| vs row-norm r = +0.97; deepseek r = −0.26). Note **qwen3.5: tied with V=248,320×D=2048, so its vocabulary geometry is exactly its logit geometry** — everything in the table applies to the unembedding head too. DeepSeek is untied, so E-vs-W_U principal angles **cannot** be computed from separate Grams (no cross-Gram EᵀW_U available); skipped per protocol, and the spectra do differ (fig_A5_E_vs_W_deepseek.png). Figs: `fig_A1_spectra_grid.png`, `fig_A1b_top20_grid.png`, `fig_A2_cos_hist_grid.png`, `fig_A2b_mean_cos_bars.png`, `fig_A3_cone_bars.png`, `fig_A3b_topvar_bars.png`, `fig_A4_norm_hist_grid.png`, `fig_A4b_proj_vs_norm_<tag>.png`.

**Hidden states (pooled 3 prompts × 400 generated tokens, n=1200).** IDpp = median over the 3 prompts of Levina–Bickel MLE (k=20, neighbors **within** one 400-token trajectory); PR/eff-rank90 are of the **mean-removed** covariance; cos = mean pairwise cosine between the 1200 states; cone = ‖μ‖/mean‖h‖.

| model | IDpp blk1 | IDpp mid | IDpp last | PR blk1 | PR mid | PR last | eff-rank90 last | cos blk1 | cos last | cone last |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek | 2.8 | 7.2 | 8.0 | 51 | 41 | 37 | 196 | 0.235 | 0.556 | 0.74 |
| qwen3_4b | 3.0 | 9.3 | 9.0 | 82 | **1.4** | 43 | 294 | 0.434 | 0.653 | 0.81 |
| phi4mini | 4.2 | 11.3 | 10.6 | 15 | 31 | 43 | 395 | 0.547 | 0.435 | 0.66 |
| qwen3_5 | 3.2 | 10.0 | 10.2 | 46 | 59 | 49 | 280 | 0.416 | 0.223 | 0.47 |

Corroborating estimators at the last raw block — TWO-NN: 5.5 / 3.9 / 6.1 / 3.2; correlation dimension: 2.0 / 9.4 / 11.3 / 8.0. Estimator spread (factor ~2–4) is the honest uncertainty. The **pooled** (cross-prompt) MLE is 2.0 / 7.3 / 11.5 / 7.3 — for deepseek it collapses to ~2 while per-prompt is ~8, a 3-cluster mixing artifact (see "trivial/artifact"). Two models grow the cone with depth (deepseek cos 0.24→0.56, qwen3_4b 0.43→0.65), two shrink it (phi4mini 0.55→0.44, qwen3_5 0.42→0.22). qwen3_4b has a dramatic **mid-stack variance collapse**: PR = 1.4–2.5 at blocks 17–23 (eff-rank90 = 15–47) — one giant direction dominates the mean-removed covariance there.

## 2. Depth profiles (Section B)

* **Norm** (`fig_B0_norm_<tag>.png`): mean ‖h‖ grows ×430 (deepseek, 1.2→513), ×620 (qwen3_4b), ×20 (phi4mini), ×37 (qwen3_5) from embedding to last raw block.
* **PCA spectra** (`fig_B1_pca_spectrum_<tag>.png`, `fig_B1b_mean_vs_centered_<tag>.png`): bulk slope β flattens with depth (deepseek 2.4→1.5; phi4mini 1.4→1.03). The uncentered second moment adds exactly one huge top mode: **with-mean PR is ~10× smaller than centered PR** (fig_B6c: "forgetting to center makes states look rank-1–5"). `fig_B1c_measures_<tag>.png` (PR, eff-rank90, λ₁/Σλ, β vs depth), `fig_B6_measures_all.png` + `fig_B6b_measures_all2.png` (all models, same axes, min/max bands over 3 prompts).
* **Intrinsic dimension** (`fig_B2_mle_vs_k_<tag>.png`, `fig_B2b_id_vs_depth_<tag>.png`, `fig_B6e_perprompt_id_all.png`): single-trajectory ID rises from ~3 (block 1) to 8–12 and plateaus; d̂(k) drifts ≤1.4× from k=10→50 (no hard plateau). LB-MLE, TWO-NN and correlation dimension agree on the *shape* (rising then flat) though not the level.
* **Local (per-token) dimension** (`fig_B3_localdim_hist_<tag>.png`, `fig_B3b_localdim_time_<tag>.png`, `fig_B3c_localdim_heat_<tag>.png`): the local dimension is **not** constant along a trajectory — the per-position d̂ᵢ (k=20) has a wide distribution (e.g. deepseek last block: median 7, IQR ≈ [4, 12]) and moves with t, with occasional high-d̂ bursts.
* **Per-layer write** (`fig_B5_write_<tag>.png`, `fig_B5b_write_rank_all.png`): mid-depth ‖Δ‖/‖h‖ ≈ 0.3–0.7; the first block writes 28× / 7.8× / 1.2× / 3.0× the embedding norm (because ‖h₀‖ is tiny, not because the write is special). **Negative result**: the write is *not* lower-rank than the state it updates — mean write eff-rank90 / state eff-rank90 = 1.42 / 1.60 / 1.23 / 1.20. Writes are mostly tangential early (|cos(Δ,h)| < 0.2 for blocks 4–16 in all models) and turn radial late (deepseek 0.39–0.5, qwen3_4b 0.3–0.5 at blocks 24–35).
* **Post-norm state** (star markers): re-orthogonalizes — deepseek cos 0.556→0.360, qwen3_4b 0.653→0.177, qwen3_5 0.223→0.131, phi4mini 0.435→0.500; PR drops to 17–50.
* **B7 limitation (not faked)**: `acts.npz` covers **generated positions only**. `logits.npz/prompt_ids` identifies the prompt (16/15/13/15 tokens) but no prompt-position states exist, so prompt-vs-generated geometry is **not computable from this data**.

## 3. Logit space (Section C)

The 128×V logit matrix (prompt 1, generated positions) is **drastically low-rank**:

| model | logit PR | logit eff-rank90 (/128) | state eff-rank90 same 128 pos. | collapse ratio | centered-logit rank90 | top-2 gap median (q05) |
|---|---|---|---|---|---|---|
| deepseek | 3.0 | 24 | 41 | 0.59 | 33 | 2.07 (0.17) |
| qwen3_4b | 1.6 | 10 | 54 | 0.19 | 45 | 4.16 (0.24) |
| phi4mini | 1.0 | **1** | 61 | **0.02** | 3 | 2.31 (0.15) |
| qwen3_5 | 3.5 | 36 | 65 | 0.55 | 56 | 1.33 (0.11) |

phi4mini's logits over 128 positions are 80% **one** direction (rank₈₀ = 1). The unembedding collapses state dimensionality by 1.8×–45× at the same positions. Decision margins dip to q05 ≈ 0.1–0.2 logit units at the 5th percentile of positions (`fig_C2_gap_timeseries_<tag>.png`, `fig_C2b_gap_hist_<tag>.png`) — the flip-prone positions of the main project's margin-dip mechanism. Figs `fig_C1_logit_spectrum_<tag>.png`, `fig_C1b_rank_bars.png`, `fig_C2c_cumvar_<tag>.png`, `fig_C2d_cumvar_all.png`.

## 4. Section D — does geometry order the dynamics? (**n=4 models: anecdotal**)

A 4-point Spearman needs |ρ|=1 for p<0.05; everything below is an **ordering observation, not evidence**.

**D1 cross-model** (`fig_D1c_spearman_bars.png`, `fig_D1_kickgain_vs_id.png`, `fig_D1b_kickgain_vs_responserank.png`). Kick gain (s=16) low→high: qwen3_5 0.47 < deepseek 0.66 < qwen3_4b 5.93 < phi4mini 22.99.

* **Late-layer anisotropy matches it exactly**: mean pairwise cosine 0.24 < 0.41 < 0.48 < 0.53 and cone 0.49 < 0.64 < 0.69 < 0.73, both ρ = **+1.00**.
* Bulk spectrum slope anti-orders: β 1.61 > 1.28 ≈ 1.31 > 0.99, ρ = **−0.80** (flatter spectrum ↔ bigger kick).
* Per-trajectory ID: 8.1 / 8.9 / 11.7 / 9.9, ρ = +0.40 — **does not** order the kick gain (phi4mini is highest-ID *and* highest-gain, but deepseek and qwen3_5 are swapped).
* δh-response-field eff-rank90 (PCA of pooled `dh_last`, all radii/tokens): 102 / 629 / 104 / 81 — ρ = +0.80 but driven entirely by qwen3_4b; phi4mini (largest kick) has a *low-rank* response field.
* No geometry measure consistently orders the late rate λ (−0.011, +0.0004, +0.00003, +0.0006) or the direction spread (5.85, 1.15, 1.23, 1.93); best |ρ| = 0.8–1.0 scattered across unrelated measures — i.e. noise.

**D2 within-model** (`fig_D2_gainprofile_<tag>.png`, `fig_D2b_scatter_pr_gain_<tag>.png`, `fig_D2c_scatter_cos_gain_<tag>.png`, `fig_D2d_withinmodel_heatmap.png`). Depth gain grows ×100 (deepseek 0.03→3.17), ×239 (qwen3_4b →11.95), ×1775 (phi4mini →17.75), but only ×12 (qwen3_5 →0.62, sub-unity). Per-layer correlations with G_ℓ (blocks only):

* deepseek (n=27): mean-cos ρ=+0.88 (Pearson +0.96), PR ρ=−0.91, spectrum slope ρ=−0.98, per-prompt ID ρ=+0.93.
* qwen3_4b (n=35): slope ρ=−0.77, ID ρ=+0.68, cos ρ≈0.
* phi4mini (n=31): eff-rank90 ρ=+0.87, PR ρ=+0.82.
* qwen3_5 (n=6 sampled layers): PR ρ=−0.94, mean-cos ρ=−0.77 — **opposite sign to deepseek**.

So: strong within-model structure, but the signs are not consistent across models and the correlations largely ride shared monotone depth trends (deepseek's anisotropy and gain both rise with depth; qwen3_5's anisotropy *falls* while its gain rises). We **cannot** claim a geometry→dynamics mechanism from this.

**D3 radial vs tangential + response subspace** (`fig_D3_radial_c2_deepseek.png`, `fig_D3c_radial_vs_radius_deepseek.png`, `fig_D3d_dhfield_spectra.png`, `fig_D3e_subspace_overlap.png`). Decoding `cos(h+δh,h)`: 1−cos ≈ (1−c²)ρ²/2 with ρ=‖δh‖/‖h‖, c=cos(δh,h):

* **DeepSeek (only model with usable precision): c² = 0.006 median** — the response is **99.4% tangential**, at every depth (per-layer c² 0.003–0.009) and **at every radius r ∈ [0.001, 2.0]** (w100 sweep, 12 radii, median c² = 0.0056). Perturbations rotate the state, they do not rescale it — consistent with the observed persistence of the argmax at small r and with flips being driven by tangential motion across the decision boundary.
* qwen3_4b/phi4mini/qwen3_5: `cos` grid-saturated (see forensics) → radial fraction **unknown**, a data gap (not a null result).
* Response field (pooled δh at the last layer): eff-rank90 = 102 / 629 / 104 / 81 of D; PR 15.7 / 57.8 / 7.9 / 4.9; top-var 0.71 / 0.44 / 0.77 / 0.92 — the response is confined to a low-dim subspace (except qwen3_4b, which spreads to 629 dims).
* **That subspace is NOT the state subspace**: top-32 PC overlap (mean squared principal cosines) δh↔state = 0.070 / 0.068 / 0.071 / 0.046, vs prompt-A↔prompt-B control 0.185 / 0.112 / 0.226 / 0.118, vs random 0.021 / 0.012 / 0.010 / 0.016. The response overlaps the state subspace 3–7× chance but **~2× less than a second prompt's states do** — the perturbation excites directions that ordinary generation does not densely populate.

**D4 window sweep** (`fig_D4_window_qwen3_4b.png`, `fig_D4_window_phi4mini.png`, `fig_D4_window_qwen3_5.png`, `fig_D4b_predT_vs_window.png`; window4d, W∈{60,100,200,4096}, r=0.1, non-DeepSeek models). Baseline margins differ across W (`margins_equal=False`), so pred_T comparisons are baseline-shifted — reported with that caveat. Last-layer gain: qwen3_4b 23.0 (W60) → 7.6 (W4096), ratio 0.33; phi4mini 7.9 → 3.8, ratio 0.47; qwen3_5 0.56 → 0.58, ratio 1.03 (window-independent, consistent with its hybrid/linear attention and with its sub-unity gain). The amplification profile is a property of the computation at a given context window, not of the weights alone.

## 5. What is interesting (≤5 falsifiable statements)

1. **The perturbation response is ≥99% tangential, robustly.** For DeepSeek, c² = cos²(δh,h) = 0.006 (median over 3 radii × 20 directions × 300 tokens × 27 blocks), and stays ≤0.01 for r from 0.001 to 2.0 — i.e. the response never becomes radial even far outside the linear regime (r≈0.3). *Falsifiable: compute cos(δh,h) directly in float32+ for the other three models; prediction c² < 0.02 everywhere.*
2. **The unembedding head collapses dimensionality by 2–45×, and phi4mini's decision space is essentially 1-D.** Logit eff-rank90 = 24/10/1/36 of 128 positions vs post-norm state eff-rank90 = 41/54/61/65 at the same positions; 80% of phi4mini's centered logit variance is one direction. *Falsifiable: on ≥10 prompts × 1000 positions, phi4mini's centered-logit rank₈₀ stays ≤ 3.*
3. **Late-layer anisotropy orders the kick gain exactly (n=4, anecdotal).** Mean pairwise cosine at the last raw block 0.24/0.41/0.48/0.53 ↔ kick gain 0.47/0.66/5.93/22.99, ρ=+1.00; spectrum slope anti-orders (ρ=−0.80). Within DeepSeek the same pairing holds layer-by-layer (cos-vs-gain Pearson +0.96, n=27). *Falsifiable: add ≥6 more models; prediction: the cos↔gain Spearman stays ≥ +0.6 — if it does not, the 4-model match is a coincidence.*
4. **A single greedy trajectory is a ~10-dimensional object.** Per-prompt LB-MLE ID at the last raw block = 8.0/9.0/10.6/10.2 (k=20, n=400), inside a 37–49 PR / 196–395 eff-rank90 affine hull, inside D=1536–3072. The states of one generation form a thin, depth-thickening manifold (ID ≈3 at block 1). *Falsifiable: ID stays ≤15 for 2000-token generations and for other prompts.*
5. **The response field is low-rank and lies outside the state subspace.** Pooled δh eff-rank90 = 81–104 (3 models; qwen3_4b 629) vs state 196–395; its top-32 PCs overlap the state PCs at 0.046–0.071, i.e. 3–7× chance (0.01–0.02) but ≈2× *below* the prompt-vs-prompt control (0.11–0.23). *Falsifiable: with δh saved at ≥3 depths, the δh↔state overlap stays below the prompt↔prompt control at every depth.*

## 6. What is trivial or an artifact

* **The last "layer" is the final RMSNorm** (forensics): the norm collapse, cos(Δ,h)=−0.98, and the PR/anisotropy jump at the top index are normalization, not computation. Any "final block" claim from this data would be wrong.
* **Pooled ID estimators are broken by 3-cluster mixing**: pooled MLE gives ~2 for DeepSeek while per-prompt gives ~8; TWO-NN/corr-dim inherit the bias. Pooling prompts and then estimating local dimension is methodologically unsound for this data; per-prompt values are the interpretable ones.
* **The embedding layer's ID is meaningless** (discrete token set + duplicates): 393/1200 (deepseek) points had all 20 neighbors equidistant (repeated tokens); those are excluded and reported; remaining values (~0.1) are quantization, not dimension.
* **Forgetting to center**: with-mean PR is ~10× smaller than centered (states look rank-1–5). The famous "LLM representations are 2-D" style claims would be this artifact.
* **Within-model layer correlations ride depth trends**: deepseek (cos & gain both rise with depth, ρ=+0.88) vs qwen3_5 (cos falls, gain rises, ρ=−0.77). Strong per-model correlations are not cross-model evidence.
* **The `cos` key was not cos(δh,h)** (≈1 everywhere); used naively it would "show" radial fraction = 1. For 3 models it's grid-saturated and unusable.
* **Window-sweep pred_T/gain comparisons are baseline-shifted** (margins differ across W); only within-W gain profiles are clean.
* **qwen3_4b's mid-stack PR dip (1.4–2.5 at blocks 17–23)** may be a single giant outlier channel surviving float16 storage; needs float32 states to confirm as structure.
* **Vocab "eff-rank90 ≈ 1000–2000" is honest but dull**: the bulk is genuinely high-dimensional; only λ₁ (mean direction) is special.

## 7. What data to collect next, and why

1. **cos(δh,h) or raw δh vectors per layer in float32+ for all models** (currently only DeepSeek is decodable, and only via a second-order identity at the last layer). This is the single most decision-relevant missing number: tangentiality explains *why* margin dips, not norm growth, drive flips.
2. **Prompt-position states** (acts at prompt positions, not only generated): enables prompt-vs-generated geometry and the attention-sink/common-direction story we currently cannot test.
3. **float32 (not float16) state dumps for 2–3 layers** to check the qwen3_4b mid-stack variance collapse and the small-scale ID tail (float16 quantization floor ≈ 1e-3×‖h‖).
4. **The raw final-block state** (before the final norm) — currently missing; "write-to-logits" analyses are contaminated without it.
5. **≥10 prompts and ≥2000-token generations**: makes pooled estimators usable (n≫3 clusters), stabilizes per-prompt ID, and lifts the cross-model n=4 anecdote to a real correlation (with more models too).
6. **qwen3_5 a1 with all 24 layers** (currently 7 sampled): the within-model correlations for the hybrid-attention model are underpowered and sign-flipped; more layers would settle whether its geometry–dynamics coupling is genuinely inverted.
7. **δh at multiple depths** (not just the last layer) + the state mean vectors μ_ℓ per layer: turns bullet 5 into a depth-resolved statement and enables exact E-vs-W_U angles for DeepSeek via a cross-Gram.

## Figure index (103 files in `figs/`)

* A: `fig_A1*`, `fig_A2*`, `fig_A3*`, `fig_A4*` (12 files) — vocab spectra, anisotropy, cone, norms, projections.
* B: `fig_B0..B6*` (49 files) — norm/PCA/ID/local-dim/anisotropy/write depth profiles per model + all-model overlays.
* C: `fig_C1*`, `fig_C2*` (18 files) — logit spectra, margins, decision subspace.
* D: `fig_D1*`, `fig_D2*`, `fig_D3*`, `fig_D4*` (24 files) — cross-model ordering, within-model correlations, radial/tangential, response field, window sweep.

Every number quoted above is in `numbers.json` (sections: `check`, `A`, `B`, `C`, `D`, `D3_radial`); `digest.py` prints the full digest. All figures: labeled axes with symbols/units, legends, conclusion titles, min/max percentile bands for aggregated curves, dpi≥170.
