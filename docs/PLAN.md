# Implementation guide (living document)

This is a guide, not a straitjacket. Change course when data says so. Keep `DECISIONS.md` authoritative.

## Goal

Measure the **continuous response** and the **decision-boundary geometry** of DeepSeek-1.5B in FP32, and decide whether the observed sensitivity is exponential instability or threshold-driven (stable chaos). Produce figures that show 3 individual cases plus a well-defined aggregate for every analysis.

## Core quantities

- `m_t = z_t^(1) - z_t^(2)`: top1–top2 logit margin at generated step `t`.
- `δh_t^(ℓ)`: hidden-state response at layer `ℓ` (finite difference or JVP).
- `δz_t`, `δm_t`: logit response; margin response using baseline top1/top2 indices.
- `R(t,s) = ||∂h_t/∂h_s||₂`: two-time response.
- `λ(t,s) = log R(t,s)/(t-s)`.
- `r_c(t,u) = m_t / |δm_t(u)/r|`: local flip threshold along direction `u`.
- `r_*(u) = min_t r_c(t,u)`: flux-tube radius.
- `T(r,u) = min{t: r ≥ r_c(t,u)}`.
- Depth map `h_{ℓ+1}=F_ℓ(h_ℓ)`, `J_ℓ=∂F_ℓ/∂h`; product `M=J_{L-1}…J_0`; depth spectrum `λ_i=(1/L)log σ_i(M)`; `h_depth=Σ max(λ_i,0)`.
- Non-normality `κ=σ_max(J)/ρ(J)`.
- `D₂` from `C(ε)=2/(N(N-1)) Σ_{i<j} Θ(ε-||x_i-x_j||)`; local slope plateau.

## File layout

```
src/chaos_llm/protocol.py        # shared FP32 inference: model, in-batch r=0, generation, teacher forcing
src/chaos_llm/run_a1.py          # baseline, margins, A1 prompt finite differences, save arrays
src/chaos_llm/run_thresholds.py  # r_c, predicted T, free-run validation
src/chaos_llm/run_depth.py       # depth tangent product, QR spectrum, local gains, principal angles
src/chaos_llm/run_nonnorm.py     # MLP/full-block sigma_max, rho, kappa
src/chaos_llm/analysis/dimension.py  # chunked D2
src/chaos_llm/analysis/rqa.py       # corrected RQA
scripts/plot_*.py                # figures; 3 individual + aggregate
results/<run_name>/              # tokens, margins, response arrays, summary.json
results/figures/*.png            # user-facing figures
results/figures/INDEX.md         # one-line description per figure
```

## Phase 0 — hygiene and data reuse

1. Refactor `run_experiment.py` to call `protocol.py` (preserve CLI/config).
2. Write data manifest and `.gitignore` fixes; commit.
3. Reuse `TDK_LLM/runs_aug/results_perturbations*` for depth FSLE (CPU).
4. Reuse `results_pecs` / `launch_sep` for D2/RQA (after correctness fixes).

## Phase 1 — baseline, margins, A1 (cheapest GPU)

1. Generate FP32 baseline, window 100, 300 tokens, in-batch `r=0`; verify.
2. Compute `m_t`, top1/top2 tokens, hidden norms.
3. A1 prompt finite differences: 20 directions, `r ∈ {1e-4, 3e-4, 1e-3, ...}`; save `δh` norms at layers `[0,7,14,21,27]`, `δz` top-k, `δm`.
4. Plot: 3 individual response curves; aggregate median/IQR; `R(t,s)` for a few `s`; margins with near-ties marked.

## Phase 2 — thresholds and FSLE

1. From A1 arrays, compute `r_c(t,u)`, `r_*(u)`, predicted `T(r,u)`.
2. Free-run validation: 20 directions × `r ∈ {1e-4, 1e-3, 1e-2}`, 300 tokens, batch with `r=0`.
3. Time-axis FSLE: 16 radii × 20 directions (reuse the A1 forwards); compute `λ(δ)` for `ρ ∈ {1.5,2,4}`; show individual `d_r(t)` and aggregate.
4. Depth-axis FSLE from existing `results_perturbations*`.
5. Plots: survival `P(T>t)`; `T(r)` step functions (3 directions); `r_c` heatmap (t × direction index, sorted by `r_*`); ECDF/violin of `r_*`; margin-vs-gain scatter with iso-`r_c`.

## Phase 3 — depth spectrum and non-normality

1. Depth spectrum: for 8–20 token positions, propagate `k=8` tangent vectors through all 28 layers with layer-wise JVP + QR; save `λ_i`, `h_depth`, local gains.
2. Principal angles between consecutive top singular subspaces.
3. Non-normality: MLP dense `J` for selected layers/tokens → `σ_max`, `ρ`, `κ`; full block at selected layers/tokens via power iteration/Arnoldi.
4. Plots: individual token growth curves, aggregate spectrum, local gains, principal angles, `κ` heatmap, `σ_max` vs `ρ` scatter, product singular values.

## Phase 4 — D2/RQA

1. Correct the old bugs: chunked distances, centered Euclidean, percentile epsilon grid, Theiler on/off, local-slope plateau, N-convergence, bootstrap/surrogates (drop if unstable).
2. Data: vocabulary matrix (2–5k rows), `results_pecs` per-prompt trajectories, last-layer states from new baseline, perturbed ensemble (per trajectory).
3. RQA: target RR 1–5%, correct DET/LAM denominators, report RR sensitivity; keep as robustness only.
4. Plots: `C(ε)` + local slope; `D₂` vs N; `D₂` vs Theiler; first vs last layer vs vocab; per-prompt distribution; surrogate band.

## Compute and parallelization

- One GPU job at a time. Background with notification; no polling.
- CPU jobs (D2, depth FSLE, plots) can run locally or on x11 in parallel.
- Subagents can work on non-overlapping CPU analyses; no subagent on GPU.
- Expected: Phase 1 < 20 min; Phase 2 < 1–2 h; Phase 3 < 1 h; Phase 4 CPU hours.
- OOM: start batch 1–4 at seq 300–600; save selected layers; if memory high, reduce batch or tokens, never precision.

## Acceptance criteria

- Every divergence result has a passing `r=0` control.
- Every analysis has 3 individual examples + one labeled aggregate.
- No `N×N` matrices on disk.
- No single number where a distribution is the object.
- All raw arrays, scripts, and figure descriptions saved and committed.
- `results/figures/INDEX.md` lets the user see every figure with a one-line description.
