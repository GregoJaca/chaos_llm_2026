p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
new = r"""

---

# Part 12. What is trivial, what is real, and where the next 10× of compute should go

This part exists to make an allocation decision. Every row is a claim we have made, its status after
the checks in Parts 9–11, and what it would take to turn it into a result.

## 12.1 Scorecard

Status codes: **D** = definitional/tautological (true by construction), **A** = artefact of the
measurement (ties, fp16, window, subspace), **R** = real but descriptive (a measured fact with no
mechanism yet), **M** = real and mechanistic (we know why it happens).

| # | claim | status | evidence | what is missing |
|---|---|---|---|---|
| 1 | response is linear in $r$ for $r\le0.3$ | R (standard) | 300× range, all models | nothing; it is a validation, not a result |
| 2 | late finite-time rate is negative (no sustained growth) | R | 0/31 configurations positive | clean re-measurement without ties and without a window (see #3) |
| 3 | a positive growth phase exists at $t\approx60$ | **A** | it exists only when $W=100$ truncates the prompt; absent for $W\ge150$ | drop, or re-frame as a window effect |
| 4 | flips occur at discrete margin dips | R | 318/320 signed rule at $r\le0.1$ | statistics with ties removed; fails at $r\ge0.5$ on fragile prompts |
| 5 | flip probability vs $r$ | **A** (contaminated) | 512 dirs: 3.5% at $r=0.1$ → 0.4% after removing the tie token | re-measure with fp32 margins |
| 6 | $\lambda_1(\ell)$ decays exponentially | **D** | it is $\log\sigma_1/(\ell+1)$ with $\sigma_1$ nearly flat | withdrawn |
| 7 | every block amplifies 3–6×, directions rotate, product grows ~20 | **M** | full-dim $\sigma_{\max}$; naive product $10^{17}$–$10^{24}$ vs measured ~20 | extend to more positions/models (cheap); identify *which* directions rotate |
| 8 | cross-position transport carries ~89% of the depth growth | **M** | same-position chain ×1.54 vs real ×53 | the transport kernel itself (running now) |
| 9 | models differ: contract vs amplify (kick gain 0.5–23×) | R | 5 models, 2 positions, 20 dirs | more prompts per model; is it prompt-stable? |
| 10 | window flush law, exact | R (mask property) | 0 residual over 240 combinations, 3 architectures | methods-section material; no more compute |
| 11 | $r_*$ distribution is a log-logistic law | **A** | not universal over 20 model-prompt pairs | withdrawn; keep only the model-level scale |
| 12 | FSLE is flat in $r$ | **D** | it is linearity | withdrawn |
| 13 | state is a ~10-dim manifold; logits are rank 1–36 | R | 4 models | not needed unless the paper is about geometry |
| 14 | response is 99.4% tangential | R | DeepSeek, all depths/radii | confirm on fp16 models with fp32 cosines |
| 15 | hot spots: DeepSeek s=16, Phi s≈40 | R | 22 positions, 5 models | are they prompt-specific? (cheap: 10 prompts) |
| 16 | long-horizon stability to 4000 tokens | R | 12 runs | systematic: 5 models × 3 prompts |

**What this table says.** The mechanistic core is #7, #8 and #9. Everything else is either a
validation, a definition, or an artefact. The next compute should go to #7–#9 and to making #2/#5
clean, not to more curve fitting.

## 12.2 Four programmes, with costs (T4-hours, measured from the runs in this notebook)

| programme | question | data | cost |
|---|---|---|---|
| **A. Transport kernel** (running) | which prompt position feeds which generated token, and how does that decay with distance and depth? Is the kernel the same across prompts/models? | all prompt positions × 8 dirs × 300 tokens, 5 models × 8 prompts | 3–4 h |
| **B. Sensitivity law** | P(flip) vs $r$, prompt, model, with ties removed: sharp threshold or smooth curve? | 5 models × 10 prompts × 512 dirs × 6 radii | 20–24 h |
| **C. Breadth** | do the model differences follow the architecture? | 4 more models (gemma-3-4b, SmolLM3-3B, Qwen2.5-3B done, +1) × suite + flips + transport | 4 h/model |
| **D. Long horizon** | is the regime stationary to 4000–8000 tokens? | 5 models × 3 prompts × 8 dirs × 4000 | 8–12 h |

Measured costs: 512 directions × 300 tokens ≈ 4 min; one prompt-position sweep (65 positions × 8
dirs) ≈ 3.5 min; a 4000-token trajectory ≈ 4 min; a full model suite (A1 all layers, state, chaos,
window) ≈ 2 h.

## 12.3 Recommended allocation of ~70 h (10× what we have used)

| item | hours | why |
|---|---|---|
| A. transport kernel, 8 prompts × 5 models | 6 | the mechanism; nothing else explains the depth growth |
| B. sensitivity law, 10 prompts × 5 models, fp32 margins | 24 | the paper's central quantitative claim |
| C. breadth: 3 more models (gemma-3-4b, SmolLM3-3B, one dense 3B) | 12 | n=5 is too small for a model claim |
| D. long horizon, 5 models × 3 prompts | 10 | stationarity of the regime |
| E. per-block rotation: 10 positions × 5 models | 6 | completes #7 |
| slack / repeats / failed launches | 12 | 20% overhead is realistic on this machine |

**What I would cut.** Anything involving the depth-map $\lambda_i$ curves, the FSLE, the $r_*$
distribution as a law, and window spectroscopy. They are either definitions or artefacts, and more
compute will not change that.

## 12.4 The decisions I need from you

1. **Headline.** Is the paper (a) the mechanism (transport + rotation), (b) the sensitivity law, or
   (c) the model comparison? The allocation above assumes all three with (b) largest; if you want a
   single spine, I would make it (a) + (b) and treat (c) as supporting.
2. **Models.** 5 (current), 7 (+gemma-3-4b, +SmolLM3-3B) or 9? Each costs ~4 h and ~8 GB of disk.
3. **Horizon.** Is 4000 tokens enough to claim stationarity, or do you need 8000–16000? Cost scales
   linearly with tokens.
4. **Margins in fp32.** May I compute the logits in fp32 for the fp16 models? It removes the tie
   artefact (0.4% vs 3.5% at $r=0.1$) but makes the new margins not strictly comparable with the old
   runs, so the key experiments would be re-run.
5. **Prompt set.** Should I freeze a canonical 10-prompt set and re-run all models on it? Cleanest
   for cross-model claims; means re-running the five models (~10 h).
"""
s = s.rstrip() + new
open(p, "w", encoding="utf-8").write(s)
print("appended")
