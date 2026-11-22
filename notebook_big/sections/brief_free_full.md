# Brief: free-run validation analysis, DS-R1-14B (full-attention run)

You are an analysis agent in the git worktree C:\Users\hjaca\research\chaos_spark_wt
(branch spark-bigmodels). No GPU needed. python (numpy/scipy/matplotlib) on PATH;
if `tokenizers` is missing, `pip install tokenizers`.

## Data

- `results_big/ds14/a1_inter_full/a1.npz` — the 1000-direction teacher-forced run
  (DS-R1-Distill-Qwen-14B, FP32, prompt interstellar_travel, P=65, T=300, r=0.1,
  FULL ATTENTION — the window setting was inactive; see the notebook's
  correction note). Keys: margins, top1, top2, dm [1,1000,300], dh_norm,
  radii=[0.1], simplex [1000,1000], subspace [1000].
- `results_big/ds14/free_inter_full/free_run.npz` — the free-run validation:
  obs [3,1000] first-mismatch step (-1 = never flips within 300 tokens),
  agreement [3,1000] mean token agreement over the whole run,
  radii = [0.05, 0.1, 0.3], dirs (all 1000),
  gen_ids = object array, one entry per radius: [1000, T] generated ids
  (rows are the directions in order; the in-batch r=0 reference of each chunk
  is NOT included), pred (engine's symmetric prediction — recompute the signed
  one yourself).
- `results_big/ds14/free_inter_full/summary.json`.
- Tokenizer: `assets/tok_ds14/tokenizer.json` (+tokenizer_config.json). Use the
  `tokenizers` library: `Tokenizer.from_file(...)` then `decode([id])`.

## Quantities (define all of these in your report)

- $r_c^-(t,u)=m_t\,r/\max(-\delta m_t(u),0)$, $r_*(u)=\min_t$, $T_{\rm pred}(u)=\min\{t: r\ge r_c^-(t,u)\}$ (signed rule; recompute from dm).
- For radius $\rho\in\{0.05,0.1,0.3\}$: a direction "flips" if obs[ri,u] >= 0. Classification per direction: **exact** (obs == T_pred at that rho, both >=0), **wrong step** (obs >=0, pred >=0, obs != pred), **false alarm** (pred >= 0, obs == -1), **miss** (pred == -1, obs >= 0), **both none**.
- NOTE: T_pred depends on rho: compute T_pred(rho, u) = min{t: rho >= r_c^-(t,u)}; at rho=0.05 many directions have no predicted flip.
- Survival curves: S(t) = fraction of directions with first flip > t. Predicted: from r_c^-(u) per rho. Observed: from obs.
- Latency classes among flipped runs: flip step == the controlling tie t_*(u) ("immediate") vs flip step > t_*(u) ("delayed"). Report the distribution of obs - t_*.
- Re-synchronisation: for flipped runs, token agreement on the suffix AFTER the flip... you only have full-run agreement (mean over all T) — better: compute per-run agreement after flip directly from gen_ids vs the r=0 reference. The reference row: gen_ids rows are perturbed runs; the baseline ids are a1's gen_ids. Compute agreement_after = mean(gen_ids[u, obs+1:] == a1_gen[obs+1:]) and max_agreement_after (does it ever return to 100%?).
- What flips at t=21: decode a1's top1[21], top2[21] and the flipped runs' gen_ids[u,21] for a few u. Report the two competing tokens as text.
- The sign-blind comparison: using |dm| (engine's stored r_c) count predictions per rho and the empirical false-alarm rate vs the signed rule.

## Deliverables

1. `scripts_spark/figs_free_full.py` + run it -> `notebook_big/figures/`:
   - `ds14free_scorecard.png` — per radius: stacked bars exact/wrong-step/false-alarm/miss/none, counts annotated.
   - `ds14free_survival.png` — observed vs predicted survival curves, one panel per radius (3 curves max per panel: observed, predicted-signed).
   - `ds14free_latency.png` — histogram of observed flip steps per radius (log-y) with the t=21/19 ties marked; scatter obs vs predicted flip step (jittered, hexbin if needed) with the diagonal.
   - `ds14free_resync.png` — distribution of agreement-after-flip (ECDF) + fraction of runs that ever re-sync to the baseline; per radius.
   - `ds14free_flipword.png` — small panel: the tokens competing at t=21 (top1 vs top2 as decoded strings) and counts of which one the flipped runs took.
2. `notebook_big/figures/ds14free_numbers.json` — all counts/fractions.
3. `notebook_big/sections/section_free_full.md` — user's template EXACTLY: 3-5 sentence plain summary; every quantity with explicit math + meaning + why; every figure with full experimental setup, axes, curves, concrete numbers; facts first; explicit "what it does not show". State clearly in the setup that this validation is against the FULL-ATTENTION A1 (the window-inactive run) and that a true-window version is being generated.

Commit with message "free-run validation analysis (full-attention DS14)". No push. Do not touch notebook_big.md. Do not touch the python server on port 8898.
