# Threshold prediction, round 2: clear figures, full evaluation, and the value story

**Scope.** Re-analysis of the DeepSeek-R1-Distill-Qwen-1.5B threshold data in
`results/local_data/`, answering four complaints about
`results/subagent_threshold/REPORT.md`: (1) unclear what is predicted and why
it matters, (2) predicted-vs-observed needs more data and honest outlier
handling, (3) the control map is unreadable, (4) the breakdown at large r
needs a mechanism figure. Everything below is recomputed from raw arrays by
self-contained scripts in this folder; nothing is carried over from round 1
except the data. Data pulled in beyond round 1: the raw free-run records
`results/a1_ds_w100/free_run{,_large}/free_run.npz`,
`results/a1_ds_w200_large/free_run`, `results/a1_ds_w500_large/free_run`
(sequences, not just summaries - these supersede the summary jsons, verified
identical), the teacher-forced cyber run `results/a1_cyber_w100/a1.npz`
(recovered - see bullet 10), the childhood-prompt free run
`fr_childhood.npz`, and the tiny-radius twin run `results/a1_ds_w100/a1.npz`
(13 radii, 1e-6..0.01). Total: **19 free-run experiments x 20 directions =
380 evaluated direction-runs, all 75 observed flips with raw sequences**
(round 1 had ~7 experiments, no cyber/childhood scoring, sequences only at
r >= 0.5).

## The prediction in plain words (for a non-expert)

The model writes its answer one word at a time, and at every word it keeps a
scoreboard of candidate words. The difference between the top two scores is
the *decision gap*: how comfortably the model chose the next word. Most words
are decided by a comfortable gap (about 5 logit units in this data), but
occasionally the model hesitates - at word 123 of the propulsion review the
gap was only 0.02, a 240x closer call than typical. Now nudge the prompt's
internal representation slightly - like tapping the table while someone
writes. The nudge shifts every decision gap a little. One extra pass through
the model (teacher-forced along the unperturbed text) records exactly how
much each gap shifts; a word can only change where that shift is *negative*
(a shrinking gap), and the smallest nudge that erases the gap at word t is
gap divided by shrink-per-unit-nudge - the *threshold* r_c(t). The
prediction: for a nudge of size r, the first changed word is the first t
whose threshold falls below r, and if no threshold does, **nothing in the
300-word answer changes at all**. Why that is worth predicting: we verified
it against 380 real generations across three prompts, and because a flip is
a point of no return - of 75 flipped runs, **none ever came back into sync**
(the two texts stay ~97% different to the end), so the fraction of the
answer that survives is simply (first-flip position)/300. One cheap pass
through the model therefore tells you whether a perturbed prompt still
produces the identical answer, and if not, exactly how many words of it
survive. Honest limits: below nudges ~3e-4 the measured shifts drown in
float32 round-off; above ~0.5 the response stops being proportional to the
nudge, so a threshold *measured at a tiny nudge* cannot be extrapolated
(only 2 of 16 flip steps exact at r=2) - but re-measuring the shift at the
nudge you actually care about restores the rule to near-perfection.

## Robust findings (each number recomputed here; scripts in this folder)

1. **The threshold rule, applied with the sign and at the right radius, is
   almost exactly right: 318/320 evaluable direction-runs (99.4%).** Over all
   19 free-run experiments (propulsion w100 at r=1e-4..2, w200/w500 at
   0.1/0.3, cyber at 0.01/0.1/0.3, childhood at 0.01/0.1/0.3): 57/59 flipping
   runs predicted to the *exact word*, 261/261 "no flip anywhere" calls
   correct, **0 false alarms, 0 missed flips**, 2 wrong steps. The observed
   flip *counts* match at every single radius, e.g. w100: obs 0,0,0,0,0,1,3,8,16
   vs predicted 0,0,0,0,0,1,3,8,16 for r=1e-4..2 (fig2, fig5b).
2. **The prediction is sharp exactly at the borderline, not fuzzy.** 70 of
   the 320 evaluable calls sit within a factor 2 of their direction's
   threshold (r within [r_*/2, 2 r_*]); **all 70 are called correctly**,
   flip and no-flip alike. The user's earlier "fuzzy in a +-30% borderline
   band" was an artifact of the sign-blind rule, not of the physics.
3. **What was wrong before was the rule as shipped, not the idea.** The
   delivered `pred_T` uses |dm| (sign-blind) and, for the three small w100
   radii, was even scored against the wrong teacher-forced row (r=0.01) - it
   says "no flip at r=0.3" while d10 flipped at word 16 (correct under the
   signed rule). Delivered totals across 380 runs: 58 exact + 279 no-flip
   correct (88.7%), but **26 false alarms, 1 missed flip, 16 wrong steps**
   (15 predicted vs 8 observed at r=1; 19 vs 16 at r=2; 7 vs 3 at r=0.5).
   `free_w100_large.json` and `free_w100_layers_large.json` are
   byte-identical duplicates.
4. **The value: a flip is a point of no return, so the prediction is a
   survival forecast.** From the raw generated sequences (all 75 flipped
   runs, 3 prompts): 0/75 ever re-synchronized (no run matched the reference
   again over its last 50 words); mean token agreement after the flip is
   **2.7%**; the whole-answer survival fraction hugs T/300 (fig3b). The 305
   runs with no flip are 100% identical. So "predicting T" = "predicting
   how much of the answer survives": at r=0.5, 3/20 directions lose 58-99%
   of the text; the other 17 lose nothing.
5. **Danger ladder replaces the control map** (fig4). Propulsion prompt
   (w100): the tie at word 123 (gap 0.020) controls 8/20 directions, word
   14 (gap 0.087) controls 6/20, word 143 (gap 0.020) 3/20; thresholds span
   r_* = 0.128 (d10) to 3.62 (d3). Cybersecurity prompt: fragility is spread
   over ties at words 17 (5/20), 39 and 22 (3/20 each), r_* = 0.13 (d1) to
   10.9 (d5) - no single dominant tie. The sign-blind |shift| rule "cries
   wolf" on 9/20 w100 directions (worst 5.5x too fragile) and **14/20 cyber
   directions** (median threshold halved) - drawn as open circles on the
   ladder.
6. **w200 and w500 are one experiment, not two: identical flip lists and
   identical reference texts.** At r=0.1 both flip exactly {d1@44, d10@55,
   d11@44}; at r=0.3 the same 11 directions flip, and d10 moves from word 55
   to word 16 - the controlling tie re-ranks *already at r=0.3*, and the
   same-radius rule still calls it exactly (11/11 steps exact).
7. **When it breaks, it is the extrapolation that breaks, not the rule**
   (fig5). Response is cleanly linear only for r in ~[3e-3, 0.3]: pooled
   gain |dm|/r is flat at ~0.0085 there. Below r ~ 3e-4 the measured shift
   saturates at |dm| ~ 1e-5 (float32 round-off of O(10) logits) and the
   apparent gain inflates as noise/r - yet the 60 tiny-nudge runs
   (r=1e-4, 1e-3, 1e-2) are still all called correctly (nothing is close
   to flipping: the smallest w100 threshold is 0.128). Above r ~ 0.5 the
   response turns superlinear (median gain 1.55x by r=2) and
   **tie-dependent**: the word-2 tie doubles (0.0148 -> 0.0313), the
   word-123 tie stays flat, the word-16 tie collapses then recovers.
   Consequence for prediction: linear extrapolation of the r=0.001
   threshold field is fully correct for r <= 0.3 (20/20 per radius;
   w200/w500 19-17/20 at 0.1/0.3), then decays 16/20 (r=0.5), 6/20 (r=1),
   3/20 (r=2, only 2/16 flip steps exact, worst error 129 words:
   predicted word 14, real flip at word 143).
8. **The only two same-radius misses have one mechanism: flips at the very
   first word are invisible to teacher-forcing.** d10 at r=1 (observed word
   0, predicted 22) and d1 at r=2 (observed 0, predicted 2). Both are deep,
   6.8-7.1x past threshold - large nudges deform the perturbed *prompt*
   itself, so the perturbed run's first step is already uncertain while the
   teacher-forced pass still follows the unperturbed context (fig6, panel 1).
9. **Childhood prompt (geometry not stored locally) scored with the
   delivered rule only:** 13/16 flips exact, 3 off by 48 words (predicted
   125, observed 173), 5 false alarms - consistent with the sign-blind
   over-prediction seen everywhere else.
10. **Data recovered / audited.** `a1_cyber_w100/a1.npz` and
    `a1_ds_w100/a1.npz` have damaged zip central directories; all members
    were recovered byte-complete via `manual_npz.py` (only the huge
    `dh_last` in `a1_ds_w100_large` is lost, unused here). w100_layers and
    a1_ds_w100 are *exact* twins (identical gen_ids, margins, and dm at the
    shared radius 1e-3), which licenses scoring the small-radii free runs
    against the w100_layers geometry. All `obs` values were re-derived from
    the raw `gen_ids` sequences and match the stored arrays exactly;
    reference rows equal the runs' unperturbed generations.

## Figures (all in this folder; all claims above have one)

| file | answers the complaint | verdict |
|---|---|---|
| `fig1_what_is_predicted.png` | "I don't understand the threshold prediction" | **keep - open with this**: real gap trace, a fragile and a safe nudged gap, overtake line, worked r_c example (0.020/(0.041/0.5)=0.25), and the observed flip at exactly the predicted word |
| `fig2_prediction_scorecard.png` | "predicted vs observed with more data; handle no-flip; outliers?" | **keep - the evaluation**: (a) all 75 flipping runs, colored by how deep past threshold, y=x line, misses annotated; (b) all 380 runs as outcome strips, signed vs delivered rule |
| `fig3_cost_of_a_flip.png` | "what is the value?" | **keep - the value**: agreement after the flip (3 examples + pooled over all 75 flips; 0/75 re-sync) and survival = T/300 |
| `fig4_danger_ladder.png` | "control map looks like noise" | **keep - replaces the control map**: one row per direction, sorted fragility, threshold + controlling word + sign-blind correction, observed radii overlaid |
| `fig5_when_it_breaks.png` | "when it breaks should teach us" | **keep - the mechanism**: gain vs r over 6 decades (noise floor / linear window / superlinear), and observed vs 3 prediction variants per radius |
| `fig6_three_directions.png` | (intuition bridge, 3 individual cases) | **keep as backup slide**: fragile d10, near-tie d12, resistant d13 - the threshold tube reshapes with r and the flip follows |

Retire from round 1: `fig3_control_map.png` (replaced by fig4),
`fig5_pred_vs_obs.png` (replaced by fig2), `fig6_large_r_linearity.png`
(replaced by fig5), `fig7_cases.png` (replaced by fig6). Round-1
`fig1_margin_gain_scatter`, `fig2_threshold_ecdf`, `fig4_window_spectrum`
remain complementary (geometry / distribution / window story).

## How to rerun

```bash
python s0_audit2.py             # audit2.txt: verifies every convention used
python s1_explore_divergence.py # divergence_explore.txt: flip cost per run
python s2_eval.py               # eval_points.json + eval_summary.txt: scorecard
python s3_numbers.py            # numbers4figs.json: fig-level numbers
python s4_qa.py                 # re-verifies every number quoted above
python s5_w200_w500.py          # w200 == w500 identical flip lists (bullet 6)
python s6_wolf.py               # sign-blind "cries wolf" counts (bullet 5)
python s7_extra_free_runs.py    # audit of the raw free_run folders (bullet 10)
python f1_explainer.py; python f2_scorecard.py; python f3_value.py
python f4_ladder.py;  python f5_breakdown.py; python f6_cases.py
```

`common2.py` holds the data model (signed threshold, free-run batch mapping,
divergence metrics); `manual_npz.py` reads the two npz files with damaged zip
directories.

## Caveats (do not overclaim)

- The "exact-radius" rule uses one teacher-forced pass *at the radius being
  predicted*. That is a cheap, local measurement - but it is a measurement at
  that radius; the radius-free *extrapolated* variant is the one that fails
  for r >= 0.5 (bullet 7). Always say which variant a number refers to.
- "No flip" (`obs = -1`) means identical within the 300 generated tokens.
- w200 and w500 obs are literally the same experiment (shared trajectory to
  t~136; identical flip lists) - count them once in any aggregate.
- Childhood scoring uses the delivered rule only (no local geometry).
- The two remaining misses (bullet 8) are both "flip at word 0" cases; we
  have no local data to test r between 0.3 and 0.5 or above 2.
