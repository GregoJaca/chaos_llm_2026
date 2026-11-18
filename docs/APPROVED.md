# Results the user explicitly endorsed

Keep this list when deciding what to protect, extend, and feature in the paper.
Each item was called good / interesting / important in review feedback.

1. **Stable chaos / flux tubes** — "some of the most interesting work so far".
   Discrete flips at margin dips, parameter-free survival prediction, no growth, hard floor.
2. **Window flush** — "brilliant and super interesting. This is one of the most important
   results." Now upgraded to an exact law: `t*_1 = p_last + (W - P)`, `t*_l = t*_1 + (l-1)(W-1)`.
3. **Layer probe** — "interesting", asked for more positions and statistics.
4. **Threshold gain scaling** — "interesting"; asked for percentile bands and to check whether the
   shape is a U (it is: concave, R²=0.977 for a parabola in log r).
5. **Latent single-position perturbation** (kick at s=16 acts 107 tokens later) — "interesting,
   get more data".
6. **Directions are not equivalent** — "nice".
7. **Window regimes** — "this is what I want to study in depth"; preliminary, now extended to
   prompt *and* state perturbations and 6 prompts.
8. **Two kinds of flips** (immediate vs delayed) — accepted as an explanation of the user's
   "latent then flip" experience.
9. **Linear response in r** — "well established, don't pursue further". Do not spend more time.

User preferences stated explicitly:
- Free y-axes; never squash curves into a corner.
- Fewer curves per plot; split when in doubt.
- Show percentiles / error bands for aggregated data, but do not overdo it.
- Define every axis; use technical terms, not English sentences, in labels.
- One unified document (`notebook/notebook.md`), not scattered md files.
- Null results are valuable and must be included.
- Keep GPU busy; longer runs are better; no idle GPU.
- Use subagents for parallel analysis.
- Prefer md over LaTeX.
