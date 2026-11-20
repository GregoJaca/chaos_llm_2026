# SPARK HANDOFF — large models on the DGX Spark (branch `spark-bigmodels`)

Concise operating doc for agents working on the large-model campaign. Read `docs/HANDOFF.md`
and `docs/FINDINGS_AND_PROCEDURE.md` in this repo for the science background; this file only
covers what is NEW for the Spark machine. Last updated 2026-11-20.

## 1. Machine and rules

- Host `ssh spark` (spark-d5eb, DGX Spark, GB10, 128 GB **unified** memory, ~273 GB/s).
- **Shared machine**: colleagues run two llama.cpp servers (~49 GB). **Never kill or disturb
  them.** ~63 GB are available to us. `big_engine.check_mem()` enforces a floor before every
  phase; keep it.
- **Bandwidth-bound**: every forward pass sweeps the weights (14B fp32 ≈ 59 GB → ~0.22 s).
  Therefore: batch many rows per forward; prefer full-sequence teacher-forced forwards
  (1 sweep for P+T tokens) over generation (T sweeps). Generation batch B amortizes
  bandwidth too: bigger B is free.
- **FP32 only** for science runs on this machine (weights fit: 14B → 59 GB). Never mix
  precisions. TF32 must stay OFF (it fakes margins).
- Repo on spark: `~/LLM/chaos_llm_2026`, branch `spark-bigmodels` (pull from origin).
  Models in `~/LLM/models/<name>`. Venv: `~/LLM/venv` (torch 2.14.1 cu130, transformers 4.57.6).
- Sync flow: edit in the Windows worktree `C:\Users\hjaca\research\chaos_spark_wt`
  (branch `spark-bigmodels`) → commit → `git push origin spark-bigmodels` → on spark
  `git pull`. **Never touch the main checkout `chaos_llm_2026`** (another agent works there).
- Long jobs: `nohup setsid bash script.sh > log 2>&1 < /dev/null &`. Never block ssh on a job.
- Windows→ssh quoting is fragile: **write scripts as local files, scp them** (files are LF
  already — never run `tr` on them; `tr` once ate every "r" in a script).

## 2. New engine code (all under `src/chaos_llm/`)

| file | what it does |
|---|---|
| `big_engine.py` | FP32 loader, memory guard, direction blocks, hook-based hidden-state capture, batched teacher-forced forward, batched free generation. The margin response uses the lm_head row-difference trick: `dm[b,t] = (h'-h0)·(W[top1_t]-W[top2_t])` — **never materialise [B,T,V] logits**. |
| `run_calib_big.py` | calibration: r=0 bit-identity, margin spectrum, r-linearity table, signed r_* preview. Decides the r window. |
| `run_a1_big.py` | A1 ensembles (1000+ directions, 1–2 radii), **same npz schema as `run_a1.py`** so old analysis scripts work. Resume-safe via `parts/`. |
| `run_free_big.py` | free-run validation, batched (chunk up to 32), same outputs as `run_thresholds.py`. |

Direction ensemble: `build_direction_blocks(K, P, D, block=20, seed)` — independent
20-direction simplex blocks on disjoint random coordinates; `simplex`/`subspace` are stored
so `index_copy_` reconstruction works exactly as before.

## 3. Protocol rules carried over (non-negotiable)

- In-batch `r=0` rows, bit-identical duplicates, one precision per model, fixed-length runs.
- Teacher forcing is the workhorse; free runs validate flips.
- Response is linear in `r` (validated per model by calibration): main ensembles run ONE
  radius; `r_c = m·r/|dm|` is r-independent via `dm/r`. Multi-radius sweeps only for the
  linearity check itself.
- Do not extrapolate thresholds to large r; validate at the radius of interest.
- fp32 floor: below some r the response stops scaling; calibrate before choosing r.
- Most action is in the first ~300 tokens; 300–500 token runs unless a specific
  experiment needs longer (trajectories, long-horizon λ).

## 4. Output layout (on spark, `results_big/`)

- `<model_tag>/calib/` — calibration npz + meta
- `<model_tag>/a1_<prompt>_w<window>/` — a1.npz (schema-compatible) + meta.json
- `<model_tag>/free_<...>/` — free_run.npz, summary.json
- window sweep dirs `a1_<prompt>_w<W>_fixed/` (teacher-forced on the big-window trajectory)
- trajectories `traj/<name>.npz` (for dimension/D2; GPU pre-computes distances, laptop
  runs the estimators)
Copy final npz/json (not parts) to the laptop for analysis; figures go to
`notebook_big/figures/` and are documented in `notebook_big/notebook_big.md`
(served on port 8898; the 1.5B notebook stays on 8899 — **no overlap**).

## 5. Model notes

- **DeepSeek-R1-Distill-Qwen-14B**: Qwen2 arch, 48 layers, d=5120, V=152064, native
  `sliding_window` + `max_window_layers=48` → full window program (flush law!) is replicable.
- **Qwen3.5-9B**: hybrid Gated-DeltaNet, transformers 5.x needed (separate venv planned),
  no sliding window → run the W-null test only.
- **phi-4** (14.7B, Phi3 arch) and **Qwen3-14B**: standard full program, window = null test
  if no native sliding window.

## 6. User-facing document rules

Every figure: define both axes with symbols, 3–4 curves max or split panels, percentile
bands for aggregates, free y-axes, conclusion in the title; say which model, which prompt,
teacher-forced or free-run, and the exact construction. Null results are first-class.
User likes many plots from multiple perspectives.
