# Brief: the window regime on the 14B — true W=100 vs full attention

You are an analysis agent in the git worktree C:\Users\hjaca\research\chaos_spark_wt
(branch spark-bigmodels). No GPU. python (numpy/scipy/matplotlib) on PATH.

## Data (local)

- `results_big/ds14/a1_inter_full/a1.npz` — DS-R1-14B, FP32, prompt
  interstellar_travel, P=65, T=300, 1000 directions at r=0.1, FULL ATTENTION
  (window setting was inactive in transformers 4.5x without a layer_types patch).
- `results_big/ds14/a1_inter_w100/a1.npz` + meta — same everything, TRUE
  sliding window 100 (verified: states bit-identical to full attention until
  t=35, depart exactly at t*=W-P; a fresh 1000-direction run at r=0.1 with the
  same direction-block construction and the same seed).
- `results_big/ds14/a1_inter_w4096/a1.npz` — 500 tokens, 32 dirs, window 4096
  (no clipping within the horizon; the phase-2 fixed-token source).
- `results_big/ds14/calib_w100/` + log — true-window calibration (margins over
  400 tokens, r-linearity, r_* preview).

## What to measure (all per the notebook style rules: free y-axes, defined
axes, conclusion in title, percentile bands, no jargon)

Write `scripts_spark/figs_w100_vs_full.py`, run it, figures tag `ds14reg`:

1. `ds14reg_margins.png` — margin traces m_t, full vs true-w100 (two panels,
   free y): same early razor ties? different late decisions? Mark t=35 (the
   first window-masked position). Table: the 8 smallest margins of each with
   their steps; count below 0.05/0.1/0.5.
2. `ds14reg_response_shape.png` — median over directions of ||dh_t^47||/r for
   full vs w100 (two curves, one panel, free y) with t=35 marked: does the
   true-window response show the flush signature (the 1.5B showed a x1.73
   jump of the response across t=35 at window 100)? Quantify: median ratio
   of ||dh/r|| at t in [36,45] vs [25,34], both runs.
3. `ds14reg_rstar.png` — r_* ECDFs full vs w100 (log-x) + predicted flip
   fraction at r=0.1 for each; controlling-step distributions (the two
   dominant ties are t=21/19 in both?).
4. `ds14reg_direction_gain.png` — G(u) distributions full vs w100 overlaid
   ECDF (log-x). The window cuts the prompt's direct influence after t=35,
   so late-t response should be smaller; quantify median G and the late-time
   fraction (median over t in [150,300) of ||dh^47||/r vs the same in [0,50)).
5. `ds14reg_depth_gain.png` — the per-state gain profile (median over t and
   directions of ||dh^l||/r vs state l, both runs): does the window change
   the depth growth of the response?
6. `notebook_big/figures/ds14reg_numbers.json`.

## Report

`notebook_big/sections/section_w100_vs_full.md` per the user's template:
plain summary; explicit quantities with math + why; per-figure full setup,
axes, numbers, what-it-does-not-show. The 1.5B reference facts for text
comparison: window 100 vs 200/500 changed its min margin 0.0201 -> 0.0031 and
the controlling step moved; the flush law t* = p_last + (W-P) held exactly.

Commit: "window regime: true w100 vs full attention, DS-14B". No push.
Do not touch notebook_big.md or running servers.
