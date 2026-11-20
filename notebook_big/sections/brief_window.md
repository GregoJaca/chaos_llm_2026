# Brief: window flush program analysis, DS-R1-14B (true windows)

You are an analysis agent in the git worktree C:\Users\hjaca\research\chaos_spark_wt
(branch spark-bigmodels). No GPU needed; python (numpy/scipy/matplotlib) on PATH.
Style rules: free y-axes, define every axis, conclusion in the title, percentile
bands, no jargon. Every figure gets the full experimental setup in the text.

## The experiment (what produced this data)

All runs: DS-R1-Distill-Qwen-14B, FP32, teacher-forced prompt perturbations
(unit direction x r=0.1 added to the 65-token prompt `interstellar_travel`),
greedy baseline, batched forward passes, sliding window ACTIVE (verified by a
bitwise state-departure test at t*=W-P).

The window mechanics, stated plainly: with a sliding window of width $W$, a
generated position $P+t$ attends only to positions $\max(0, P+t-W+1)\dots P+t$.
A prompt-embedding perturbation can therefore only influence position $P+t$
while some perturbed prompt position is still inside the window. For a
perturbed position $p$ (0-indexed), its last visible generated step is
$t=p+(W-P)$; at $t=p+(W-P)+1$ the response from that position is gone. For a
direction whose perturbed positions have maximum $p_{\rm last}(u)$, the
response at the FIRST blocks must therefore become **bitwise zero** at
$t^* = p_{\rm last}(u) + (W-P)$ — before that, at least one perturbed position
is visible; after, none is. Deeper blocks keep a nonzero response longer,
because position $P+t$'s block-$\ell$ state reads block-$(\ell-1)$ states of
previous positions inside the window: the expected cascade is
$t^*_\ell = t^*_1 + (\ell-1)(W-1)$ (a chain: to reach block $\ell$ at $t$, the
information must have travelled one window span per block).

Files (each an a1-style npz with `dh_norm [1,K,16,300 or 500]`, `simplex`,
`subspace`, `margins`, `layers`):
- `results_big/ds14/a1_inter_w4096/a1.npz` — no-clipping reference, 500 tokens,
  K=32 (single block? check `dirs_per_block` in meta).
- `results_big/ds14/a1_inter_w{50,100,150,200,300,500}_fixed/a1.npz` — same
  tokens (fixed-tokens-from the w4096 run), K=240 (12 blocks of 20), 500 tokens.
  NOTE: each direction's own p_last(u) differs (each block uses its own 20
  random prompt positions) — compute p_last per direction:
  for direction u, active coords = subspace[simplex[u] != 0]; positions =
  coords // D (D=5120); p_last(u) = positions.max().
- `results_big/ds14/a1_inter_w100_pos{0-22,22-44,44-65}/a1.npz` — support scan,
  240 dirs confined to prompt positions [0,22), [22,44), [44,65), window 100,
  300 tokens, same fixed tokens.
- `results_big/ds14/a1_{short_prompt,long_prompt}_w{50,100,200,300,500}_fixed/a1.npz`
  — prompt-length scan, P~6 and P~250 (check each meta.json for the true P),
  120 dirs, 300 tokens.
- `results_big/ds14/a1_inter_w100_cascade/a1.npz` — 1500 tokens, 8 dirs,
  layers 0..14, window 100.
- 1.5B reference curves if useful: `C:\Users\hjaca\research\chaos_llm_2026\results\local_data\flush_sweep\w*\a1.npz` (READ-ONLY).

## Tasks — `scripts_spark/figs_window.py` -> `notebook_big/figures/`, tag `ds14win`

1. `ds14win_departure.png` — THE law test. For each W-run and each direction:
   measured death time = first t where dh_norm[1] (state 1 = block 0 output) is
   <= 0 (use exact zero; fp32 makes pre-death states nonzero), predicted =
   p_last(u) + (W-P). Scatter measured vs predicted, one color per W, with the
   identity line. Annotate: n directions, fraction on the diagonal, max
   deviation (in tokens). Repeat for state 2 and 3 (cascade start).
2. `ds14win_cascade.png` — death time per state index l (0..15) for the W=100
   and W=200 runs: median death time vs l with the predicted lines
   t*_l = t*_1 + (l-1)(W-1) (use each direction's own t*_1; median over dirs).
   Include the 1500-token cascade run (states 0..14, better statistics) as a
   second panel.
3. `ds14win_support.png` — support scan: median death time (state 1) for the
   three position-restricted runs vs the p_last+35 prediction; each direction's
   p_last comes from its restricted coords. This is the controlled version of
   the law: same W, same P, only the support moves.
4. `ds14win_plen.png` — prompt-length scan: for short_prompt and long_prompt,
   measured death time (state 1) vs predicted p_last+(W-P) per (W, direction).
   Annotate the P<W, P~W, P>W regimes. State plainly what happens when P>W
   (the prompt itself does not fit the window: the model attends only to the
   last W-1 prompt positions + current).
5. `ds14win_response_shape.png` — median dh_norm/r at the last state over t for
   each W (50..500, w4096 reference), same tokens: the flush surge/kink at
   t=35=W-P for W=100 etc. Mark the predicted kink t* per W with vertical lines.
6. `ds14win_margins_by_w.png` — the margin spectrum of the SAME fixed tokens
   under each W (margins arrays differ per W): median margin and min margin vs
   W; plus the full margin traces overlaid (light) with the t=35 line. Justify:
   the window changes the decision landscape itself, not just the propagation.

Write `notebook_big/figures/ds14win_numbers.json`: per (W, state) measured vs
predicted death stats; per-run p_last distribution; kink positions; margin
minima per W.

## Report

`notebook_big/sections/section_window.md` per the user's template: plain
summary first; the window mechanics paragraph above reproduced (it defines
p_last, t*, and the cascade, with explicit expressions); every figure with
setup/axes/numbers/what-it-does-not-show. Compare explicitly with the 1.5B law
(numbers: on the 1.5B the law held exactly to the token for 5 windows).
Facts first; conclusions only where direct.

Commit: "window flush analysis DS-14B". No push. Don't touch notebook_big.md
or the 8898 server.
