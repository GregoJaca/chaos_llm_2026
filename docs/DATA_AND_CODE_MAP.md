# Data and code map

Status: verified on 2026-11-17. Paths are on `x11` under `/u0/grego/LLM` (`~` = `/u0/grego`).
This is a factual map, not a plan. “Quality” refers to reproducibility and fitness for quantitative claims.

---

## 1. Code map

### 1.1 `chaos_llm_2026/src/chaos_llm/` (token-level divergence pipeline)

| File | What it does | Important detail |
|---|---|---|
| `run_experiment.py` | Main loop: baseline generation, then perturbed generation, per sliding window and magnitude. | **Baseline is batch 1; perturbed rows are batch N (16–64). No `r=0` row in the perturbed batch.** This is the source of the protocol defect. |
| `generation.py` | `model.generate` wrapper; optional manual top-k loop; `BaselineDivergenceCriteria`. | `adaptive_stop` stops at first mismatch; default off. |
| `perturbations.py` | Builds a regular simplex and a random subspace; adds a **sparse** perturbation. | The perturbation has `num_conditions` nonzeros: one ~`r`, the rest ~`-r/N`. The norm is `r`. |
| `modeling.py` | Model/tokenizer loading; `apply_attention_overrides`. | Sets `config.sliding_window` and `use_sliding_window`. Qwen2 applies the window through a 4D causal mask in transformers 4.50. |
| `analyze.py`, `analysis/` | Token-level divergence, agreement, logits, plots. | No hidden states are loaded or saved. |
| `output.py` | Saves `tokens.npz`, `config.json`, `texts.json`. | `tokens.npz`: `baseline_ids`, `perturbed_ids`, `perturbed_lengths`, `divergence_index`, `prompt_len`. |

The pipeline does **not** save hidden states or logits by default. `tokens.npz` is the only per-run data.

### 1.2 `TDK_LLM/runs_aug/` (layer-wise hidden states and Jacobians)

| File | What it does | Important detail |
|---|---|---|
| `run_perturbations.py` | Single-token prompts; simplex perturbations; saves hidden states at every layer. | `dtype = float16 if config dtype == "float16" else float32`; with `bfloat16` in the config it loads **float32**. Saved tensors are float32. |
| `src/perturbations.py` | `build_simplex`, `select_subspace_indices`, `generate_simplex_perturbations`. | Same simplex construction as `chaos_llm_2026`; radius = norm. |
| `src/generation_experiments.py` | `generate_and_save_hidden_states`; batches; `output_hidden_states=True`. | Files are `single_token_..._layer_{l}.pt`, shape `[n_conditions, 1, d]` for single-token prompts. |
| `run_jacobians.py`, `analyze_jacobians.py` | MLP Jacobians, spectral norm, Frobenius norm, mean-field metrics. | Results are JSON metrics, not raw Jacobians. |
| `run_jacobians_attn.py`, `analyze_jacobians_attn.py` | Attention Jacobian spectral norm via power iteration. | Results in `results_jacobians/*/`. |
| `run_normalization_jacobians.py`, `analyze_normalization_jacobians.py` | RMSNorm/LayerNorm Jacobian measurements. | Results in `results_normalization*/`. |
| `analyze_perturbations.py`, `plot_perturbations.py` | Layer propagation, power-law fits, extracted Jacobian gains. | Used for the paper’s Fig. 7. |

### 1.3 `~/LLM/` (token trajectories, legacy)

| File | What it does | Important detail |
|---|---|---|
| `launch_sep.py` | 100 perturbed prompts; generates fixed-length token trajectories; saves last-layer hidden states. | Uses `generate_perturbed_prompts`: **dense i.i.d. Gaussian** `torch.randn(...) * radius` per coordinate, not a simplex. No seed. `repetition_penalty=1.1`. Model loaded **without `torch_dtype`** → default model dtype (bf16 for DeepSeek). |
| `launch_aug.py` | Same generator; 4 or 16 conditions; saves hidden states at layers -1/14/22. | Same dense Gaussian and no-seed caveat. |
| `launch_pentek.py` | Temperature/radius runs; dense Gaussian; fp32; saves `trajectories.pkl`, distances, entropy, hypervolume, `results_summary.json`. | Results in `results_10_april/`. |
| `launch_pecs.py` | One trajectory per prompt; saves first/middle/last hidden states and pairwise distance matrices. | Results in `results_pecs/`. No ensemble. |
| `data_analysis/src/runner/lyapunov_lib/core.py` | Previous “Lyapunov” analysis. | `_compute_ftle_curves` computes `log(d(t)/d(0))/t` from a **distance curve**; it is not a tangent-map Lyapunov exponent. Saturation exclusion/midpoint alignment are heuristics. |

---

## 2. Token-level data (`chaos_llm_2026/outputs_*`)

All runs use one prompt: `interstellar_travel` (`prompts.txt`). All are **tokens only** (no hidden states). The baseline is batch 1; perturbed rows are batch N.

| Path | Model | Runs | Windows | Magnitudes | `n_cond` | dtype | `r=0` | Size | Quality |
|---|---|---:|---|---|---|---|---|---:|---|
| `outputs_0.0001` | DeepSeek | 20 | 1–2000 | 1e-4 | 1000/200 | bf16 | no | 85 MB | **Low**: bf16 + no `r=0` + near-tie. |
| `outputs_0.0004` | DeepSeek | 14 | 0–1000 | 4e-4 | 100 | fp16 | no | 5.4 MB | Medium-low: near-tie fragile at some windows. |
| `outputs_0.0006` | DeepSeek | 20 | 0–2000 | 6e-4 | 250 | fp16/bf16 | no | 55 MB | Medium-low; mixed dtype. |
| `outputs_OLD` | DeepSeek | 29 | 0–4096 | 5e-5…8e-4 | 100 | fp16 | no | 24 MB | **Best token-only set**: same 100 directions across `r`; still near-tie fragile. |
| `outputs_OLD_0.0001` | DeepSeek | 3 | 20/35/4096 | 1e-4 | 100 | fp16/bf16 | no | 1.4 MB | Low. |
| `outputs_microsoft` | Phi | 65 | 3–1000 | 0…5e-4 | 100–200 | bf16 | **yes** | 33 MB | **Not usable**: `r=0` diverges (bf16 batch effect). |
| `outputs_microsoft_0.00001` | Phi | 17 | 1–4000 | 1e-5 | 200 | bf16 | no | 12 MB | **Not usable**: bf16 + no `r=0`. |
| `outputs_microsoft_0.0001` | Phi | 12 | 10–1000 | 1e-4 | 1000 | bf16 | no | 38 MB | **Not usable**: bf16 + no `r=0`. |

Each run folder contains `config.json`, `tokens.npz`, `texts.json`.
`tokens.npz` keys: `baseline_ids` `[L]`, `perturbed_ids` `[N,L]`, `perturbed_lengths` `[N]`, `divergence_index` `[N]` (often `-1` in batched mode), `prompt_len` scalar.

**Use for:** pipeline tests, quick token-level exploratory plots, fixed-direction staircase audit (`outputs_OLD`).
**Do not use for:** quantitative claims about perturbation sensitivity, attention-window dependence, or chaos. Re-run with the baseline protocol in `DATA_AUDIT.md`.

---

## 3. Layer-wise hidden states (`TDK_LLM/runs_aug/results_perturbations*`)

Single-token prompts: `"The"`, `"House"`, `"she"`, `" understanding"`, `"id"`, `"sex"`, `"cat"`, `"a"`, `"win"`, `"fly"`. Perturbation is a simplex in a random subspace; radius = norm. **No `r=0` / unperturbed baseline is saved.** Model and dtype are not recorded per run (inferred from the file dtype and dataset name).

| Path | Model | Runs | Radii | `n_cond` | Shape per layer | Size | Quality |
|---|---|---:|---|---|---|---:|---|
| `results_perturbations` | DeepSeek | 80 | 1e-8…1.0 | 1000 | `[1000,1,1536]` | 14 GB | **Good for layer/depth analysis** (saved tensors fp32, simplex, seeds recorded). Model/dtype not recorded per run; script maps `bfloat16` → fp32. No baseline. |
| `results_perturbations_OLD` | DeepSeek | 555 | 1e-8…1.0 | 20/100/500/1000 | `[N,1,1536]` | 88 GB | Same; many legacy runs. |
| `results_perturbations_microsoft` | Phi | 250 | 1e-6…0.08 | 1000 | `[1000,1,3072]` | 95 GB | Good for layer/depth, Phi. |
| `results_perturbations_microsoft_excluded` | Phi | 10 | 0.1 | 1000 | `[1000,1,3072]` | 3.8 GB | Edge case. |
| `results_jacobians` | DeepSeek | — | — | — | JSON metrics | 148 MB | Existing MLP/attention/MFT metrics; raw Jacobians not stored. |
| `results_jacobians_microsoft` | Phi | — | — | — | JSON metrics | 146 MB | Same. |
| `results_normalization` | DeepSeek | — | — | — | JSON metrics | 56 MB | Norm Jacobian metrics. |
| `results_normalization_microsoft` | Phi | — | — | — | JSON metrics | 63 MB | Norm Jacobian metrics. |

Per-run `config.json` keys: `timestamp`, `setup`, `prompt_hash` or `prompt_idx`, `radius`, `n_conditions`, `subspace_mode`, `prompt_text`, `tokenized_strings`, `target_tokens_perturbed`, `max_new_tokens`, `seed`.
**Use for:** depth propagation, layer-wise gain, depth FSLE, depth Lyapunov spectrum, non-normality of MLP blocks, mean-field checks.
**Do not use for:** token trajectories, first-divergence times, attention-window dependence. There is no unperturbed baseline; use pairwise simplex distances, not baseline differences.

---

## 4. Token trajectories with hidden states

### 4.1 `~/LLM/launch_sep/` (DeepSeek, 100 conditions, last layer)

| Run | Conditions | Hidden shape | Size | Generation |
|---|---:|---|---:|---|
| `interstellar_propulsion_review_0_0.00035` | 100 | `[T,1536]`, `T≈3031` | 6.4 GB | T=0, top_p=1, top_k=1, **repetition_penalty=1.1**, max_new=3096 |
| `childhood_personality_development_0_0.00035` | 100 | `[T,1536]`, `T≈3039` | 1.8 GB | same |

- Perturbation: **dense i.i.d. Gaussian**, `torch.randn(n_conditions, P, D) * 0.00035` per coordinate. The total norm is ≈ `0.00035 * sqrt(P*D) ≈ 0.11`, not `0.00035`. It is **not** a simplex and not comparable to the token-level `r` values.
- No seed is set in `generate_perturbed_prompts`; directions are not reproducible.
- Model is loaded without `torch_dtype` → default weights dtype (bf16 for DeepSeek); hidden states are stored fp32 but computed in bf16.
- No `r=0` baseline; all 100 conditions are perturbed.
- Extra files: `results.json` (list of 100: prompt, model_configuration, generated_text, generated_tokens), `generated_text.json`, sentence embeddings (`sentence-transformers_all-mpnet-base-v2_traj{i}.pt`), `axis_lengths.pt`, `hypervolume.pt`, `sweep_results/`.
- **Use for:** qualitative RP/correlation-dimension on individual trajectories, layer/volume analyses.
- **Do not use for:** perturbation sensitivity, `r`-scaling, baseline comparisons, reproducibility claims.

### 4.2 `~/LLM/results_pecs/` (DeepSeek, one trajectory per prompt)

12 prompts (childhood, cybersecurity, future of work, great emu war, healthy habits, interstellar, personal finance, quantum consciousness, remote work, robot friendship, smartphones, transformer architecture). Each folder:

- `hidden_states_first.pt`, `hidden_states_middle.pt`, `hidden_states_last.pt`: shape `[1, T, 1536]`, fp32. `T` 873–4095.
- `cosine_sim_{first,middle,last}_{first,middle,last}.pt`, `l2_dist_{...}.pt`: pairwise distance matrices.
- `result.json`: prompt, `model_configuration` (`temperature=0.0`, `do_sample=false`, `max_length=4096`), generated text.
- `tokens.json`: list of generated token ids for the single trajectory (length varies by prompt).
- `smartphones_society_impact` additionally has `dimension_results.json`, `correlation_dimension.py`, `embedding_matrix_correltion_dimension.png`.

**Use for:** RP/correlation dimension per trajectory, layer comparison.
**Do not use for:** perturbation ensembles; there is one trajectory per prompt and no `r`.

### 4.3 `~/LLM/results_10_april/` (DeepSeek, legacy T/radius runs)

Six runs: `launch_pentek_{0,0.6}_{0.01,0.02,0.04}`. Each has `trajectories.pkl`, `distances.pkl`, `entropy_values.pkl`, `hypervolume.pkl`, `axis_lengths.pkl`, `trajectory_hidden_lengths.pkl`, `results_summary.json` (`initial_prompt`, `generated_texts`, `fractal_dimensions`, `lyapunov_exponents`, `config`), `plots/`, and some `cos_similarity_traj_*.pt`. `recurrence_plots_doc/` has 6 files. Total ~4 GB.

- Perturbation: dense Gaussian (`torch.randn * radius`), same as `launch_pentek.py`.
- Model loaded in fp32.
- **Use for:** legacy comparison of fractal dimension / “Lyapunov” estimates; RP examples.
- **Do not use for:** new quantitative claims without checking the config/seed and the estimator definitions.

### 4.4 `~/LLM/launch_aug/` (DeepSeek, T/radius runs)

12 runs. Each has `hidden_states_layer_-1.pt`, `hidden_states_layer_14.pt`, `hidden_states_layer_22.pt` (shape `[N, T, 1536]`, N=4 or 16, T≈4031–4039, fp32), `generated_text.json`, `results.json`, `axis_lengths.pt`, `hypervolume.pt`, sentence embeddings. `repetition_penalty`/perturbation scheme same dense Gaussian as `launch_sep` (no seed). Total ~2.5 GB.

**Use for:** layer-wise trajectory analysis at multiple layers; qualitative RP/dimension.
**Do not use for:** perturbation sensitivity or reproducibility.

---

## 5. Embedding matrices

- DeepSeek: `151936 × 1536`, Phi: `200064 × 3072`. Available from the model weights (`model.get_input_embeddings().weight`).
- Free point cloud for correlation-dimension null/reference. It is a discrete set, not a trajectory; state that in the interpretation.
- The output embedding matrix (`lm_head.weight`) is separate; DeepSeek has `tie_word_embeddings=false`.

---

## 6. Multi-use matrix

| Data / run | A1 response | A3 margins / A2 prediction | A4 FSLE | 3.4 thresholds | 5.2 tangent/Lyapunov | RP / D₂ | Controls |
|---|---|---|---|---|---|---|---|
| New baseline protocol (FP32, in-batch `r=0`) | yes | yes | yes | yes | yes (two-time response) | yes | reference |
| Teacher-forced forward passes (new) | **core** | **core** | yes | **core** | yes | — | — |
| `results_perturbations*` (depth hidden states) | — | — | depth-axis | — | depth spectrum | — | — |
| `results_jacobians*` | — | — | — | — | MLP/norm metrics | — | — |
| `launch_sep` / `launch_aug` trajectories | — | — | qualitative | — | — | yes (caveats) | — |
| `results_pecs` trajectories | — | — | — | — | — | yes (per prompt) | — |
| Embedding matrix | — | — | — | — | — | null/reference | — |
| Old token-only `outputs_*` | — | staircase audit | — | pilot | — | — | data-quality test |

**One new teacher-forced dataset per prompt/direction set serves A1, A3, A2 prediction, 3.4, and the two-time response in 5.2.** This is the main compute-saving lever.

---

## 7. Gotchas (read before using any data)

1. **Token-only data has no hidden states.** `chaos_llm_2026/outputs_*` cannot support RP/D₂ or response curves without re-running with hooks.
2. **Phi bf16 `r=0` diverges.** `outputs_microsoft*` token-level results are invalid. See `DATA_AUDIT.md`.
3. **DeepSeek fp16 near-tie at window 100, step 123.** The saved baseline token is the second-best in FP32. Do not treat the 123-step divergence as perturbation amplification.
4. **`launch_sep`, `launch_aug`, `results_10_april` use dense Gaussian perturbations**, not simplex. The directory “radius” is the per-coordinate standard deviation; the total norm is ≈ `radius * sqrt(P*D)`. They are not comparable to the token-level `r`.
5. **`launch_sep`/`launch_aug` have no seed and use `repetition_penalty=1.1`.** Directions are not reproducible; the generator is not pure greedy argmax.
6. **`launch_sep` model is loaded without `torch_dtype`**; computation is in the model’s default dtype (bf16), even though hidden states are stored fp32.
7. **`TDK/results_perturbations*` has no unperturbed baseline** and uses single-token prompts; it is for layer/depth analysis, not token trajectories.
8. **Old “Lyapunov” is an FTLE from distance curves** (`log(d(t)/d(0))/t`), not a tangent-map Lyapunov exponent. Do not conflate.
9. **`results_jacobians*` store metrics, not raw Jacobians.** Recompute the full `J` if you need eigenvalues/non-normality.
10. **Qwen2 sliding window:** in transformers 4.50 the 4D causal mask applies `config.sliding_window`; `max_window_layers` affects the kernel-level argument, not the mask. Verify before claiming layer-specific window behavior.
11. **`outputs_microsoft` has `r=0` only for some windows** (3–150, 1000), not all. Do not assume a control exists elsewhere.
12. **One prompt only** in the token-level data (`interstellar_travel`). Generalization claims require new prompts.

---

## 8. How to verify a dataset before using it

1. Read `config.json` / `results_summary.json`: model, dtype, window, batch, `n_conditions`, radii, seed, temperature, repetition penalty.
2. Check the file dtype and shape (`torch.load(..., mmap=True, weights_only=True)`).
3. For token-level data: recompute the baseline in FP32 on the same tokens and compare; check margins at disagreements.
4. For ensemble data: confirm there is an `r=0` row in the same batch, or treat the dataset as non-reproducible.
5. For hidden-state data: confirm the layer index, whether teacher-forced or free-running, and whether the token sequence is saved.
6. For reuse: align rows by condition index and seed; confirm the same directions across radii.
