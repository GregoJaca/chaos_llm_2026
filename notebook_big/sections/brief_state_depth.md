# Brief: state kicks + depth spectrum + non-normality, DS-R1-14B (true-window core)

You are an analysis agent in the git worktree C:\Users\hjaca\research\chaos_spark_wt
(branch spark-bigmodels). No GPU. python (numpy/scipy/matplotlib) on PATH.
Style: free y-axes, define every axis, conclusion in the title, percentile
bands, no jargon, full experimental setup per figure.

## Data (pull from spark when phase 2 is done; paths on spark under
~/LLM/chaos_llm_2026/results_big/ds14/, local copies under results_big/ds14/)

- `state_w100/state2.npz` — single-position kicks. Setup: DS-R1-14B FP32,
  window 100 (TRUE, verified), the true-w100 A1 baseline tokens; inject r=0.03
  and r=0.1 at ONE generated position s (a random unit vector in the 5120-d
  residual stream at the embedding output of that position, 24 random
  directions each), teacher-forced continuation; measures dh [2 radii, 10
  positions, 24 dirs, 5 states, 300 steps], cos, dm [2,10,24,300].
  positions s = 8,16,32,50,75,100,150,200,250,299; states ~ [0,12,24,36,47].
- `depth_w100/depth.npz` — depth tangent spectrum. Setup: forward-mode
  Jacobian-vector products: k=8 orthonormal tangent vectors at one generated
  position, propagated block by block (48 blocks), QR-renormalised each block
  (Benettin); lam [8 tokens, 48, k] = cumulative log-gain per block,
  local_gain [8,48], tokens = 0,21,50,100,150,200,250,299 (21 = the razor-tie).
- `nonnorm_w100/nonnorm.npz` — MLP-block dense Jacobian at 5 tokens x selected
  blocks (analytic SwiGLU formula): mlp_smax, mlp_rho, mlp_kappa = smax/rho;
  full-block sigma/rho/kappa via matrix-free power iteration + Arnoldi at 5
  blocks; angles: principal angle between the top singular directions of
  blocks 0,4,8,...,47.

## 1.5B reference numbers (from the T4 notebook, for comparison in text)

- Depth: absolute top-8 exponents positive, final lambda_i ~ 0.08-0.18 nats/block,
  h_depth ~ 0.6-1.4; but residual norm grows x30 in block 0 and x1.05-1.18
  afterward; NORMALIZED by state scale every block CONTRACTS
  (max relative exponent -0.034); a small +0.02 nats/layer aligned drift remains.
- Non-normality: full block 0 sigma_max ~95-195, rho ~11-39, kappa ~5-8;
  mid layers sigma ~2.7-3.6, kappa ~1.7-2.5; last block sigma ~6-9, kappa ~3-5.3;
  MLP alone at last block sigma ~58-93. Median principal angle 83.6 deg.

## Tasks — `scripts_spark/figs_state_depth.py` -> figures tag `ds14sd`

1. `ds14sd_state_kicks.png` — per position s: (a) immediate response size
   ||dh_s^47||/r (median over 24 dirs) vs s; (b) half-life: first t where the
   response falls below half its peak, per s; (c) the latency question: for
   kicks at s, does dm_t at LATER steps ever cross the flip line? count
   directions where min_t r_c^-(t) <= 1.0 (any radius would flip some token),
   per s. Justify each quantity in text.
2. `ds14sd_depth_spectrum.png` — (a) the depth spectrum lam_i(l) per block for
   the 8 tokens (median + IQR band, 8 curves max -> 2 panels); (b) cumulative
   h_depth per token; (c) local gain per block (median over tokens) with the
   state-norm growth on a twin axis if norm_h0 available from the A1 run —
   the point: absolute growth vs relative contraction (recompute the relative
   version by dividing local gain by the baseline state norm ratio between
   consecutive states, from the A1 run's norm_h0 at states 0..47 if present,
   else use the saved per-block gains only and say so).
3. `ds14sd_nonnorm.png` — (a) mlp sigma_max, rho, kappa vs block (5 tokens,
   median + range); (b) full-block sigma vs rho scatter colored by block with
   the kappa=1 line; (c) principal angles histogram with the 1.5B's 83.6-deg
   median marked.
4. `notebook_big/figures/ds14sd_numbers.json` — all key numbers.

## Report

`notebook_big/sections/section_state_depth.md` per the user's template.
Explicit comparisons to the 1.5B numbers above in every part, with a summary
table. If the 14B contradicts a 1.5B conclusion, say it loudly.

Commit: "state kicks + depth + non-normality, DS-14B". No push. Don't touch
notebook_big.md or running servers.
