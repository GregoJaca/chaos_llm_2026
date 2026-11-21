# Brief: core-program analysis for a new model (phi-4 / Qwen3-14B)

You are an analysis agent in the git worktree C:\Users\hjaca\research\chaos_spark_wt
(branch spark-bigmodels). No GPU. python (numpy/scipy/matplotlib) on PATH.
Notebook style rules: free y-axes, every axis defined (symbol + short
definition), conclusion in the title, percentile bands for aggregates, no
jargon, full experimental setup per figure, facts first.

## Data (local under results_big/<tag>/, tag = phi4 or qwen314)

- `calib/` — margins spectrum, r-linearity table, r_* preview (8 dirs x radii).
- `a1_inter_w0/a1.npz` — the core: 1000 directions at the calibrated radius,
  300 tokens, ~14 saved states (0,1,2,3,5,8,12,16,20,24,28,32,36,39),
  24 dh_last vectors + h0_last/h0_mid baseline vectors, margins/top1/top2/dm.
- `free_inter_w0/` — free validation at 2 radii (0.1, 0.3), all 1000 dirs,
  complete gen_ids.
- `state_w0/state2.npz` — single-position kicks (10 positions x 24 dirs x 2 radii).
- `depth_w0/depth.npz` — depth tangent spectrum (8 tokens x 40 blocks x k=8).
- `nonnorm_w0/nonnorm.npz` — MLP/full-block Jacobian metrics + principal angles.
- `a1_childhood_w0/`, `a1_cyber_w0/` — 200-dir multiprompt generality runs.
- `traj/` — trajectories (for D2; the distance matrices arrive separately).
- Tokenizer for decoding flip words: assets/tok_<tag>/ (pull tokenizer.json
  from spark:~/LLM/models/<model-dir>/ if missing).

## The established reference numbers (use in every comparison)

| | 1.5B | DS-R1-14B | Qwen3.5-9B |
|---|---:|---:|---:|
| min margin (nats) | 2.02e-2 @123 | 9.73e-5 @21 | 7.27e-3 @267 |
| median margin | 4.78 | 3.10 | 2.46 |
| r* median | 1.56 | 0.116 | 0.707 |
| flips at r=0.1 (free) | 0/20 | 45.3% | 1.2% |
| fp32 floor (last state) | 1.0e-3 | 2.7e-3 | 6.1e-4 |
| clean r window | [1e-3,0.1] | [0.03,0.3] | [0.01,0.3] |
| depth: state-normalized verdict | contracts | contracts (61/64 chunks) | contracts, saturates |
| principal-angle median | 83.6 deg | 87.0 deg | 90.1 deg (response vs state) |
| prediction at r=0.1 | 19/20 exact | 98.2% acc | 12/12 exact |

## Tasks — `scripts_spark/figs_core_<tag>.py`, figures tag <tag>

1. `<tag>_calib.png` — margins + linearity (the Part-0-style calibration view).
2. `<tag>_core.png` — margins trace with the 5 smallest labelled; r_* ECDF
   (log-x, run-radius line, predicted flip fraction); controlling-step ladder
   (which ties control, top-3); gain distribution (G(u) ECDF).
3. `<tag>_free.png` — scorecard (exact/wrong/false-alarm/miss per radius) +
   survival curves + the decoded competing tokens at the top gates
   (tokenizer) + re-synchronisation verdict (0 re-sync expected).
4. `<tag>_layer_profile.png` — median ||dh^l||/r per state over t (heatmap)
   + depth growth summary (median gain at each state, states 0..39).
5. `<tag>_depth.png` — depth spectrum: lam_i(l) medians + h_depth per token;
   the state-normalized relative exponent verdict (contract or expand?),
   comparing to the reference table. Use the razor-tie token if it is among
   the 8 depth tokens and report whether it is a tangent hotspot like the
   14B's t=21.
6. `<tag>_nonnorm.png` — sigma_max/rho/kappa by block (MLP + full), principal
   angle histogram, versus the reference table.
7. `<tag>_state.png` — kicks: immediate response per position s, half-lives,
   the latency question (which later gate flips).
8. `<tag>_generality.png` — the childhood/cyber 200-dir runs: r_* ECDFs vs
   the interstellar run of the same model (do the gates move? margin spectra
   side by side).
9. `<tag>_numbers.json` with everything.

## Report

`notebook_big/sections/section_<tag>.md` per the user's template. Include the
cross-model comparison table (4 models now) updated with this model's numbers
in every relevant part. Answer explicitly: where does this model sit in the
fragility hierarchy and WHY (margin spectrum vs response gain, using the
r_* = m_min / (local sensitivity) decomposition from the DS-14B analysis).

Commit: "core analysis, <tag>". No push. Do not touch notebook_big.md or
running servers.
