# Brief: Qwen3.5-9B analysis — margins, thresholds, and the hybrid-attention layer profile

You are an analysis agent in the git worktree C:\Users\hjaca\research\chaos_spark_wt
(branch spark-bigmodels). No GPU. python (numpy/scipy/matplotlib) on PATH.
Style: free y-axes, define every axis, conclusion in the title, percentile
bands, no jargon, full setup per figure.

## Data (local copies under results_big/qwen35/ after scp from spark)

- `calib/calib.npz` + `calib.log` — calibration: margins, r-linearity table
  (find the clean r window; the fp32 floor may sit elsewhere than the 14B's
  because the architecture mixes linear-attention blocks whose recurrent state
  rounds differently), 8 dirs x 9 radii, states [0, 8, 16, 24, 31].
- `a1_inter_w0/a1.npz` + meta — the core: K=1000 directions at the calibrated
  radius, T=300, ALL 32 block states saved (layers 0..31), 24 dh_last vectors,
  h0_last/h0_mid baseline vectors, margins/top1/top2, dm, dh_norm.
- `free_inter_w0/free_run.npz` + summary — free validation at 2-3 radii.
- Architecture facts (from the model config, verified): 32 blocks, hidden 4096,
  vocab 248320 (padded), layer pattern period 4: THREE Gated-DeltaNet
  (linear attention) blocks then ONE full-attention block; block indices with
  `full_attention` in config.layer_types are 3,7,11,15,19,23,27,31. Full-attention
  blocks: 16 heads (4 KV), head_dim 256. Linear blocks: 16 QK heads x 128,
  32 V heads x 128, conv kernel 4. No sliding window anywhere (documented null).
- Tokenizer for decoding flip words: assets/tok_qwen35/ (pull tokenizer.json
  from spark:~/LLM/models/Qwen3.5-9B/ if not present).

## Tasks

Write `scripts_spark/figs_qwen35.py` and run it -> `notebook_big/figures/`, tag `q35`.

1. `q35_calib.png` — margins spectrum + the r-linearity table as a plot
   (dh/r vs r per state, slope-1 guides, floor marked): the Qwen3.5 version of
   the 14B's Part 0.
2. `q35_layer_profile.png` — THE new measurement. Median over 1000 directions
   of ||dh_t^l||/r for each block l=0..31 (heatmap l x t, log color; or 32 thin
   curves in 4 grouped panels by l%4 position in the period). Mark the
   full-attention blocks. Questions to answer with numbers: (a) does the
   response grow monotonically like a standard transformer, or do the
   linear-attention blocks compress it (their recurrent state has fixed size,
   so old information decays structurally); (b) is there a visible signature
   at the full-attention blocks (spikes in the profile)?
3. `q35_margins_rstar.png` — margins trace + 5 smallest; r_* ECDF; predicted
   flip fraction at the run radius; the danger ladder of controlling steps.
   Compare in text: 9B hybrid vs DS-14B vs 1.5B (numbers from the notebook
   Parts 1-3: 1.5B min margin 2.0e-2, r_* p50 1.56; 14B min 9.7e-5, r_* p50 0.116).
4. `q35_free_validation.png` — scorecard + survival per radius (same format as
   the DS-14B free section).
5. `q35_gain_structure.png` — gain distribution + the 24 response-vector
   pairwise cosines + the response-vs-baseline-state angle USING the saved
   h0_last (now available!): cos(dh, h0) per saved direction per step, histogram;
   the 1.5B claimed responses sit ~87 degrees from the state — test it here.
6. `notebook_big/figures/q35_numbers.json`.

## Report

`notebook_big/sections/section_qwen35.md` per the user's template (plain
summary; explicit math + why per quantity; per-figure setup/axes/numbers;
facts first). Include a cross-model table: 1.5B / DS-14B / Qwen3.5-9B with
margins, r_*, gains, floor, clean window, flip fractions.

Commit: "Qwen3.5-9B core analysis". No push. Don't touch notebook_big.md or
running servers.
