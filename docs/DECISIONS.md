# Decisions

Status: agreed 2026-11-17. These are binding for the current revision unless changed explicitly.

## Scope

- **Model:** `DeepSeek-R1-Distill-Qwen-1.5B` only for now. Phi and other models are out of scope.
- **Precision:** **FP32 for all new science runs** (generation, teacher forcing, Jacobians). No FP16/BF16 in new experiments. If something does not fit, reduce batch/sequence/saved layers; do not lower precision. Splitting a job is allowed only when it is not very inefficient; use common sense.
- **No Tier 3 compute:** no attention-window sweeps, no multi-model replication, no Mamba/controls, no training, no 7B models.
- **Prompt:** `interstellar_travel` from `prompts.txt`. One prompt for the core study; additional prompts only if cheap and useful.
- **Default window:** 100 for the first A1/threshold study; 200 and 500 for comparison where cheap. The old window sweep is not repeated.

## Measurement protocol

- **In-batch `r=0` reference.** The unperturbed condition is row 0 of every batch; a duplicate `r=0` row is included. All comparisons are within the batch.
- **Fixed-length generation** (no EOS truncation) or explicit EOS handling.
- **Save metadata:** model path/revision, dtype, attention implementation, window, batch, seed, prompt, prompt length, torch/transformers/CUDA versions, GPU, git commit, timestamp.
- **Verify before use:** duplicate `r=0` bit-identical; solo vs batched reference reported.
- **Save all positions.** Prompt positions `0…P-1`; generated positions `t=0…T-1` (`t=0` is the first generated token). Default figures are generation-only; the prompt response is a separate panel.
- **A direction is site-specific.** A direction over the full prompt embedding is not comparable to a direction in one token/layer/step. Do not pool across sites without saying so.

## Experiments

- **A1:** prompt perturbation first, finite differences. Current-state perturbation later if time. Use 20–32 directions; radii for linear check `1e-4, 3e-4, 1e-3`; FSLE grid 16 radii (log-spaced) × 20 directions.
- **A4/FSLE:** time-axis from A1 data; depth-axis from existing `results_perturbations*` hidden states. `ρ` is a choice; compute `ρ ∈ {1.5, 2, 4}` and assess. Consider the distance metric saturation (cosine max ~1, so `ρ` must be below saturation).
- **Thresholds/`T(r,u)`:** measure `m_t` and `δm_t`; use the **signed** threshold `r_c⁻=m_t·r/max(−δm_t,0)` (a flip requires `δm_t≤−m_t`; the symmetric `|δm|` rule over-predicts). Predict `T(r,u)=min{t:r≥r_c⁻}`; validate with free-run at 2–3 radii. Do not extrapolate beyond `r≈0.3`.
- **D2/RQA:** correct methodology; exclude or regenerate repetition-loop trajectories (near-exact periods, high recurrence); do not use raw-vs-time-shuffled D2 as a dynamic control (it is a tautology for point-cloud statistics). Use no-plateau/finite-N ceilings and MIAAFT/phase predictability.
- **Depth map:** full layer tangent product per token; QR/Benettin depth spectrum; local layer gains; principal angles; `h_depth=Σ positive λ_i`. Report **absolute and state-scale-normalized** exponents; the absolute expansion is radial drift and the final RMSNorm projects it out. Use existing depth hidden states for FSLE, new JVP for spectrum.
- **Non-normality:** MLP dense Jacobian (analytic or autograd) → `σ_max`, `ρ`, `κ=σ_max/ρ`; full block at selected layers/tokens via matrix-free power iteration/Arnoldi. Plot layer-wise and token-wise, individual + aggregate.
- **D2/RQA:** correct methodology; per-prompt primary; ensemble per-trajectory secondary; vocabulary null; Theiler window with and without; chunked distances, no `N×N` disk. RQA corrected and kept as a minimal robustness panel.
- **Cycles/periodicity:** not studied.
- **RP visual claims:** not a main claim; keep only quantitative RQA robustness if useful.
- **Token-time full Lyapunov spectrum:** not computed initially; A1 `R(t,s)` is the cheaper substitute. Compute only if A1 shows exponential growth.

## Data and repo

- **Do not reuse** the old step-by-step inference code for new data. Use the shared protocol layer with KV cache.
- **Refactor `run_experiment.py`** to call the shared protocol, preserving CLI/config.
- **Git:** track code, configs, small JSON/CSV, docs; ignore tensors/data/models. Add a data manifest with x11 paths/checksums. Do not use Git LFS for 250 GB.
- **Commit and push** after each coherent unit of work.
- **No defensive over-engineering.** Correct algorithms, clean data structures, simple code. The repo is ours.

## Non-goals

- Proving deterministic chaos.
- Phi/bf16 results.
- Attention-window dependence.
- Periodicity.
- Contractive Transformer, logistic-map toy model, Mamba.
- Full token-level Lyapunov spectrum before evidence.
- SOTA benchmarking or downstream task performance.

## Extra, nearly-free analyses to include when the tensors are already loaded

- Baseline margin distribution `m_t`.
- Hidden-state norms by layer (radial scaling).
- Top-1/top-2 logit gap at near-ties.
- Attention entropy by layer/head (if cheap).
- Per-layer response gain at selected layers.
- Embedding-matrix D2 null.
- Public plot index with one-line descriptions.
