p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()

old = """**Panel (a).** $\\|\\delta h_{t,\\ell}\\|/\\|\\delta h_{1,\\ell}\\|$ on a log axis. Every layer decays; the
deeper layers decay later and less, and the curves cross: at $t<30$ the deep layers are still
filling up (the perturbation is arriving), after $t\\approx60$ the shallow ones have died."""
new = """**Panel (a).** $\\|\\delta h_{t,\\ell}\\|/\\|\\delta h_{1,\\ell}\\|$ on a log axis. Every layer decays; the
deeper layers decay later and less, and the curves cross: at $t<30$ the initial transient still
dominates, after $t\\approx60$ the shallow layers are dead and only the deep ones remain."""
assert old in s
s = s.replace(old, new)

part7 = r"""

---

# Part 7. What to do with the next 24 GPU-hours

Four proposals, each with the question, the data it needs, the cost, and the figure that already
supports it. The recommendation is at the end.

## P1 (recommended). The geometry of a flip: from phenomenology to a predictive theory

**Question.** A flip is a crossing of the decision hyperplane
$(w_a-w_b)\cdot h_t=0$. We have the first-order rule (318/320) and the margin coordinate $u_t$. Can
we turn this into a *quantitative, predictive* theory of when greedy generation changes its mind and
how the two trajectories re-converge afterwards?

**Why it is the strongest line.** It is the only place where we have a law that is both non-trivial
and falsifiable, and it is the mechanism behind every "chaos" claim. It also produces the number a
referee will ask for: the **recovery (mixing) time** after a flip.

**The three measurements.**
1. **Boundary map.** For one prompt and each generated position $t$, the set of perturbation
   directions that flip the token is a half-space; its boundary is the hyperplane
   $(w_a-w_b)\cdot\delta h_t=-m_t$. Measure the distance to the boundary in units of $r$ for many
   directions → a distribution of "how close the model was to changing its mind" at every token.
2. **Prediction test at large $r$.** Teacher-forced $\delta m$ at $r=0.3\ldots2.0$ predicts the first
   flip; run the same directions autoregressively and compare flip times (we have the pipeline; it
   is 318/320 at small $r$, the large-$r$ regime is untested).
3. **Recovery.** After a flip, measure $\|\delta h\|$ and the token agreement as a function of lag;
   the time to re-converge is the "memory" of the flip.

**Data.** 3 prompts × 20 directions × 6 radii, teacher-forced with `dm` and margins (cheap, already
supported), plus the same directions autoregressively (the expensive part). ~2–4 GPU-hours.

**Evidence already in hand.** `figures/margin_coordinate.png` (the coordinate and the linear approach
to the boundary, $r^*\approx0.54$) and the 318/320 signed rule.

## P2. An architecture scan: what sets the amplification factor?

**Question.** Across four models the single-kick downstream gain spans **0.5× to 23×** and the late
rate spans $-0.011$ to $+0.0006$. Is there an architectural variable that orders them?

**Candidate variables.** attention type (dense / sliding / linear), relative depth of the local
layers, the growth of the residual-stream norm, and the intrinsic dimension of the state (the
subagent is computing the last one now).

**Data.** The standard suite (A1 all layers, 3 prompts, state, kicks, trajectories) on two more
models chosen to break the degeneracy: **Qwen2.5-3B** (dense + real 32k window, same family as
DeepSeek) and **gemma-3-4b** (5:1 local 1k / global). ~1.5 GPU-hours each plus download.

**Evidence.** `figures/probe_models.png`: the factor is flat in injection depth, so it is a property
of the architecture/position, not of where you inject.

## P3. Window spectroscopy: reading the context without ablating it

**Question.** Which prompt tokens does the model actually use, at each generated position? The flush
law gives an *exact clock*: at token $t$ with window $W$, the prompt tokens older than $W-t$ are
invisible. Sweeping $W$ therefore removes one token at a time, in a controlled, reversible way, and
the difference between $W$ and $W+1$ isolates a single prompt position. This is a causal probe with
no ablation and no retraining.

**Data.** One prompt, $W=60\ldots400$ in steps of 1 (341 runs) at 2–3 layers, one radius, 4
directions. At ~15 s per run that is ~1.5 h. The result is a **contribution map**: $\Delta$response
versus (prompt position, generated position).

**Why it is novel.** Nobody uses the sliding window as a causal instrument. It exploits exactly the
mechanism we have characterised (the exact $W-P$ law), and it needs no new science to be correct.

**Evidence.** `figures/window4d_models.png` (panel c: the law is exact, residuals 0.00) and
`figures/window_first_departure.png`.

## P4. Long-horizon convergence of the flip rate

**Question.** Is the per-token flip hazard stationary over 4000 tokens, and does it depend on $r$ as
a threshold model predicts? We have 12 runs at 4000 tokens; the hazard confidence interval is wide.

**Data.** 40 runs × 4000 tokens × 10 directions at $r=0.1,0.3$. ~6 GPU-hours. This is the run that
would justify a "72-hour job" later: it is embarrassingly parallel and purely confirmatory.

## Recommendation

**Go all-in on P1, run P3 alongside it, and start P2's downloads now.** P1 is the theory; P3 is a new
instrument that costs almost nothing and produces a figure nobody has; P2 is the breadth that makes
the paper about models rather than about one model. P4 is confirmatory and can wait for the 72-hour
slot.

Budget: P1 4 h, P3 2 h, P2 4 h (two models × suite), figures/analysis 4 h, slack 10 h.
"""
s = s.rstrip() + part7
open(p, "w", encoding="utf-8").write(s)
print("appended", len(part7))
