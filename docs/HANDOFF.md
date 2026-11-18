# Handoff: DeepSeek-1.5B dynamical-systems project

Concise operating manual for the next agent. Read this first, then `notebook/notebook.md`
(the single reference document) and `docs/APPROVED.md` (what the user endorsed).

## 1. What this is

We treat greedy decoding of `DeepSeek-R1-Distill-Qwen-1.5B` as a dynamical system and study how
perturbations propagate. The paper's framing is **stable chaos / flux tubes**: trajectories shadow
each other exactly until a finite perturbation crosses a local decision boundary; no positive
Lyapunov exponent is detectable at these scales.

Goal: enough data and analysis for a publication. The user wants breadth (many plots), honest
nulls, percentile/error bands, and explanations of every axis.

## 2. Compute

- GPU host `x11` (ssh alias). One Tesla T4, 15 GB, **shared** — check `nvidia-smi` first.
- Repo on the host: `~/LLM/chaos_llm_2026` (same git repo as local).
- Local machine (where the user reads): `C:\Users\hjaca\research\chaos_llm_2026`.
- Model: `~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B`, FP32.
- **Run long jobs detached:**
  `ssh x11 "cd ~/LLM/chaos_llm_2026 && nohup setsid bash scripts/QUEUE.sh > results/QUEUE.log 2>&1 < /dev/null & disown"`
  Poll with `tail results/QUEUE.log` / `pgrep -af chaos_llm` — never block on the ssh call.
- Queue scripts (`scripts/queue*.sh`, `scripts/big_queue.sh`) run jobs sequentially and skip
  outputs that already exist (`if [ -f out/a1.npz ]`).
- Transfer results with `scp -q x11:~/LLM/.../results/X local/results/local_data/X`.

### Memory rules (learned the hard way)
- FP32 weights ≈ 7.1 GB; the T4 leaves ~7.5 GB.
- `run_a1` is safe with `--batch-dirs 1` and ≤500 tokens. **Batch 2 OOMs sporadically** at 300–400
  tokens (allocation failure at ~14.4 GB, looks like fragmentation). Use batch 1.
- 800+ token single runs OOM when computing full-vocab logits; cap at ~500 tokens.
- `run_state2` is safe with batch 1–2; `run_layer_probe` with 6 dirs.
- Call `torch.cuda.empty_cache()` between batches (already done in the scripts).

## 3. Repository map

```
src/chaos_llm/protocol.py        model loading, FP32, window, embeddings, generation
src/chaos_llm/run_a1.py          prompt perturbation: response, margins, thresholds
src/chaos_llm/run_thresholds.py  free-run validation, first-flip steps
src/chaos_llm/run_tf_predict.py  teacher-forced token prediction vs free run
src/chaos_llm/run_state2.py      single-position perturbation (multi-layer, multi-radius)
src/chaos_llm/run_state_freerun.py  autoregressive validation of a single-position kick
src/chaos_llm/run_layer_probe.py inject at layer L, read every layer
src/chaos_llm/run_after_flip.py  state difference with identical tokens after a flip
src/chaos_llm/collect_trajectories.py  1000–4000-token greedy trajectories
src/chaos_llm/run_attention.py   attention entropy (negative result)
scripts/queue*.sh, big_queue.sh  GPU job queues
scripts/notebook_fig_*.py        every figure in the notebook
notebook/notebook.md             THE document (definitions + every figure explained)
notebook/figures/                figures referenced by the notebook
docs/APPROVED.md                 endorsed results + user preferences
docs/RESULTS_SUMMARY.md          condensed numbers (older, partly superseded)
docs/FINDINGS_AND_PROCEDURE.md   procedure for a new model
docs/REVIEWER_MAPPING.md         reviewer objection -> evidence
docs/HANDOFF.md                  this file
results/subagent_*/REPORT.md     deep dives (stablechaos, dirs, d2c, state2, lambda, forms)
```

## 4. Data inventory (local copies under `results/local_data/`)

| path | contents |
|---|---|
| `w100_layers/a1.npz` | main A1: 12 radii, 20 dirs, 8 layers, 300 tokens |
| `w200/a1.npz`, `w500/a1.npz` | same prompt, other windows |
| `window_sweep/w*`, `window_sweep_cyber/w*` | fixed tokens, 6 windows, radii .03/.1/.3 |
| `flush_sweep/w*`, `flush_cyber/w*` | shallow layers, 500 tokens, for the flush law |
| `state_deep_emb`, `state_window/w*` | single-position perturbations, depth + window |
| `layer_probe*`, `layer_probe_multi/` | injection-layer probes |
| `fr_*.npz` | free-run flip records (radii .05…2) |
| `rate_prompts/`, `rate_w50/` | λ(t) across prompts at W=100 and W=50 |
| `long_last/` | 16 × 1000-token last-layer trajectories |
| `preclip.npz` | hidden states for W=50/100/200/4096, no perturbation |

On the host, `results/big/` holds the large 12–24 h batch: `flush/`, `regime/`, `state_w*`,
`probe_p*`, `chaos/`.

## 5. Established results (with the evidence figure)

1. **Stable chaos** — discrete flips on the lowest-margin steps (p<5e-6), parameter-free survival
   prediction, no growth (≤1.02), hard floor r≈0.13. `stablechaos_*.png`, Part 2.0.
2. **Window flush law** — `t*_1 = p_last + (W−P)`, `t*_l = t*_1 + (l−1)(W−1)`, exact to the token.
   `window_flush_scaling.png`.
3. **No pre-clip effect** — hidden states identical until `t = W−P`. `window_preclip_null.png`.
4. **Threshold prediction** — signed rule 318/320 (99.4%) at the measured radius; sign-blind 89.1%.
5. **Flips are terminal** — 0/75 re-synchronise; survival = first-flip/300.
6. **Latency** — a kick at s=16 changes the token 107 later; confirmed autoregressively 5/5.
7. **Directions** — magnitudes stable across texts (ρ 0.8–0.98), signs not (ρ≈−0.05).
8. **Depth** — no local amplification at an ordinary position; ×3 at a near-tie position.
9. **D2** — no scaling region; all apparent plateaus are artifacts (loops, duplicates, m=2 fill).
10. **λ(t)** — positive early (~+0.03 nats/token at t≈50), negative later; no sustained positive rate.

## 6. Traps that produced wrong results before

- **Median vs max over directions**: "first exact zero" of a *median* curve is a quantile, not a
  physical time. Use per-direction times, or a threshold crossing.
- **Confounded window comparisons**: comparing mean λ over a fixed t-interval across W mixes the
  clip position into the statistic. Compare at fixed `τ = t−(W−P)`, or only pre-clip.
- **`cos` in `a1.npz`** saturates at 1.0 (float32). For angles use the saved full vectors
  (`dh_last`) — the perturbation is 86–87° from the state.
- **Qwen2 windows only the last `max_window_layers`**; teacher-forced long-lag tails persist.
  Free runs are the ground truth for the window.
- **bf16**: r=0 diverges; never mix precision. Phi-4 bf16 data in the repo is invalid.
- **`a1.npz` in `local_data/` root** is the w200 run, not w100. Check `meta.json`.
- **npz corruption**: `a1_cyber_w100` and `a1_ds_w100` had damaged zip directories; recovered with
  `results/subagent_threshold2/manual_npz.py`.
- **Time-shuffled D2** is a tautology for point-cloud statistics; do not use it as a control.
- **Repetition loops** contaminate D2/predictability; screen trajectories first
  (`scripts/screen_loops.py`).

## 7. Current state and what to do next

- The 12–24 h batch (`results/big/`) is running or just finished: flush (4 prompts × 10 windows),
  regimes (6 prompts × 10 windows × 3 radii × 12 dirs), state-window, layer probes, stable-chaos
  free runs.
- Next: copy `results/big/` locally, run the intended analyses:
  1. flush law across prompts/radii, with prediction residuals;
  2. λ(t) pre-clip vs post-clip (fixed τ), direction-resolved with bootstrap CIs;
  3. regime boundary at W=P with the continuous scan;
  4. stable chaos across 4 more prompts.
- Then update `notebook/notebook.md` (prepend to Part 2) and `docs/APPROVED.md`.

## 8. Style rules (user requirements)

- Every figure: defined axes with symbols, 3–4 curves max, conclusion in the title, percentile
  bands when aggregating, free y-axes (never squash).
- Keep the single notebook current; do not scatter new md files for the user.
- Null results are first-class.
- Keep the GPU busy; prefer long runs; use subagents for CPU analysis.
