# Radial scale drift: what it means concretely, and is it actually right?

**Task.** The user asked what "depth expansion is radial scale drift" means in plain words, and whether it is true.
This folder contains a full re-analysis of the saved data (`radial_analysis.py`, everything recomputed from
`results/local_data/`), six figures, and `stats.json` with every number quoted below.

**The data used.** One 300-token teacher-forced run of DeepSeek-R1-Distill-Qwen-1.5B (FP32, attention window 100,
"interstellar travel" prompt). "Teacher-forced" = the model is fed its own previously recorded tokens instead of
re-generating, so every measurement is against a fixed baseline. A tiny nudge (radius r, 20 directions) was added to
the *prompt embeddings*; the saved arrays give, at 8 layers [0,1,2,3,7,14,21,27]: the state length `||h||`, the nudge
length `||dh||`, the cosine between perturbed and baseline state, the logit-margin change `dm`, and — for 3
directions — the full nudge *vector* at the deepest saved layer. The baseline state vectors of the same run exist in
`margins/w100.npz` (verified: identical token IDs, norms match to 1e-7, stored cosines match recomputation to 2e-7).
Because a 100-token attention window is used, the prompt nudge reaches generated positions in hops: directly at
layer 1 for tokens t≤98, one hop later for t≈99–198, two for t≥199. "Layer 27" in all files is the input to the final
block (HF `hidden_states[27]`); the final block, the RMSNorm output, and the unembedding weights are **not** saved.
Two other saved measurements are used where noted: the "JVP depth spectrum" (the model's local stretching factors
for infinitesimal nudges, measured layer-by-layer with automatic differentiation, `depth/depth.npz`) and the "FSLE
depth curves" (how fast a cloud of nudged copies spreads, layer by layer, in older runs, `depth/fsle_depth.json`).

---

## The concrete definition (2 sentences, no jargon)

Each layer of the model *adds* its contribution to a running per-token vector (the "residual stream"), so that
vector's **length grows from ≈1.2 to ≈545 (×452; ×29 inside the first block alone) as you go from layer 0 to layer 27**.
"Radial scale drift" was our name for the idea that a nudge's raw size inflates with depth only because the
**yardstick — the state's length — is inflating**, and that the output is blind to this because the network divides
the length away (RMSNorm) right before converting the vector into next-token scores.

## What the data says, bullet by bullet

- **The yardstick inflation is real.** Median `||h||` by layer: 1.21, 35.4, 43.6, 51.8, 76.0, 107.1, 199.4, 545.3
  (block-0 gain ×29.3, total ×452). This part of the claim is solid — but it is a statement about the **state**, not
  about the nudge.

- **FAILS: the nudge is not "radial" (not a length change).** Splitting the saved nudge vectors against the true
  baseline state: the angle between nudge and state is **86.1–86.4° (median) at every radius r=1e-3…2**
  (10–90% range 80.9–89.3°). The along-state part is only **6.3–6.9% of the nudge's length** (a direction-less random
  vector in 1536-D would have 2.6%; so there is a small systematic radial bias, but the nudge is a *direction* change,
  not a length change). The measured state deflection is tiny — at r=0.01, ≈0.0015°. **Best figure: `fig2_angle.png`.**

- **Why the docs thought it was radial: a float32 artifact.** The saved cosine between perturbed and baseline *states*
  reads ≈1.0 at r≤0.1 — but any nudge that is small next to the state gives cos≈1; below an angle of ≈0.028° the
  float32 cosine saturates at 1.0 and **cannot distinguish radial from perpendicular**. Where the cosine is resolvable
  (r≥0.3), the measured state deflection matches the fully-perpendicular prediction at every layer (e.g. r=2: measured
  0.478° vs predicted 0.481°). The project's own r=2 cosine analysis (`tangential_fraction ≈ 0.86–1.0` in
  subagent_nonnorm's summary.json) already showed this; it contradicted the "radial" wording and was not carried
  through. **`fig2_angle.png` panel (b) shows the floor.**

- **FAILS (in this dataset): the nudge does not merely ride the yardstick, and dividing by the state length does
  NOT shrink it.** Per token (r=0.01), the nudge's growth relative to the state's growth is **×5.6 over layers 3→7**
  (the arrival transient), **×1.46 over 7→14**, **×1.33 over 14→21** — 87–92% of tokens are above 1 — and only
  **×0.84 over 21→27** (84% of tokens below 1). Same numbers at r=0.03 and r=0.1 (±0.03). So the relative nudge
  `||dh||/||h||` **grows ≈×10 across the depth and contracts only in the final layers**. **Best figures:
  `fig1_state_nudge_ratio.png` (one glance: (a) and (b) grow, (c) grows too — the opposite of the claimed fall),
  `fig4_stability.png`.**

- **Partly right elsewhere: the "two numbers" reproduce.** From the JVP depth spectrum (the model's layer-by-layer
  stretching factors for infinitesimal nudges, `depth.npz`): absolute sum of
  positive top-8 exponents `h_depth` = **+0.947 nats/layer mean (range +0.644…+1.358)** (a "nat" is a natural-log
  unit: +0.95 nats/layer ≈ growth ×2.6 per layer); after removing the per-layer
  state-length growth every exponent is negative — sum **−0.42…−1.15 (median −0.836) nats/layer**, maximum single
  relative exponent **−0.039** (docs cite −0.034; same conclusion). From the old ensemble spread curves (FSLE): absolute
  separation growth **×206–215 net** (reached by layer ~2: ×240–262, then flat-to-declining), which becomes **×0.45–0.48
  relative** — *net*: the relative separation first **grows ×6.7 by layer 2**, then falls ×15 as the state lengthens
  while the spread does not. **Best figure: `fig5_exponents.png`** (also the honest note that the new teacher-forced
  data does *not* show the mid-depth contraction the FSLE/JVP numbers suggest).

- **The output-blindness mechanism is right, but must be restated.** RMSNorm is exactly scale-invariant
  (`c·h → c·h`, any c>0, normalizes to the same vector), so the state's ×452 length inflation — and the 5–8% along-state
  part of the nudge — are invisible to the logits *by construction*. What survives is the perpendicular part, divided by
  the state length. Measured: the margin change tracks the perpendicular part `b` linearly over 3+ decades
  (log-log slope 1.03; Spearman 0.58–0.64 vs 0.44–0.48 for the along-state part), with an effective gain
  **|dm| ≈ 4.1–4.4 × (b/||h||)**, stable across radii. At r=0.01 the relative nudge is 2.7e-5 and |dm| ≈ 1e-4 — the
  logits are protected by *scale-invariance*, not because a big radial piece gets deleted. **Best figure:
  `fig3_logits.png`, `fig6_schematic.png`.**

- **Where the numbers are trustworthy.** The response is linear in r for r=1e-3…0.3 (median scaling within ±10%); the
  smallest saved vectors are real signal (r=1e-3 vs r=1e-2 nudge vectors: median cos 0.906). At **r=1e-3 individual
  entries sit at the additive FP32 floor** (~4e-6 at layer 1 rising to ~1.7e-4 at layer 27): rank-correlation with the
  r=1e-2 pattern is 0.99 but medians are inflated ×1.07–1.58 — use medians only, or r≥3e-3.

- **Net verdict.** Right: the state's length inflates ×452; absolute nudge sizes are inflated by it; the output is
  length-blind, so only the relative direction change matters; JVP/FSLE exponents flip from positive to negative once
  the yardstick's growth is removed. Wrong: the nudge itself is **perpendicular (≈86–87°), not radial**, and its
  relative size **grows ≈×10 through mid-depth**, shrinking only in the last block band (×0.84). The sentence to
  publish is: *"the state's length drifts ×450 with depth and the output renormalizes length away, so logits see only
  the tiny, slowly-varying direction change — not the inflated raw sizes"* — not "the perturbation is radial and is
  projected out."

## What remains uncertain

1. The deepest saved state is one block *before* the final RMSNorm; the final block could still rotate the nudge, and
   the effective gain ≈4 (which folds in the final block, RMSNorm, and unembedding) is inferred, since none of those
   are in the data.
2. Mid-depth disagreement between datasets: the old FSLE ensemble (flat spread ⇒ relative contraction after layer 2)
   vs the new teacher-forced run (relative growth +0.04–0.05 nats/layer in layers 7–21). Different runs, prompts,
   injection, and attention setups — a same-run re-measurement is needed before publishing either as "the" depth law.
3. Why the response is ~perpendicular to the state (mechanism unknown from this data); the 3 saved direction-responses
   are also only weakly parallel to each other (pairwise cos ≈ 0.2–0.3), so the response is not a single dominant mode.
4. r=1e-3 per-entry values are floor-contaminated (above); cosines below r≈0.3 are uninformative (float32 floor
   0.028°), so per-layer angles cannot be measured in the deep linear regime from `cos` alone.

## Files

- `radial_analysis.py` — the single reproducible script (recomputes everything, prints the log, saves `stats.json`,
  `analysis_log.txt`, and the six figures). `inspect_data.py`, `explore2.py`, `explore3.py`, `qa_figures.py` are the
  intermediate verification steps.
- Figures: `fig1_state_nudge_ratio.png`, `fig2_angle.png`, `fig3_logits.png`, `fig4_stability.png`,
  `fig5_exponents.png`, `fig6_schematic.png` — each panel is titled with its conclusion and every axis is defined in words.

---

## Summary for the main agent

The user asked for a concrete, plain statement of "depth expansion is radial scale drift" and a check of its truth.
The check, run on the saved w100 teacher-forced data (including the stored last-layer nudge vectors and the matching
baseline states in `margins/w100.npz`), **confirms the scale story but refutes the "radial" wording and the "relative
response shrinks" part**: the state's length does inflate ×452 with depth (×29 in block 0), the JVP and FSLE absolute
exponents do flip negative once the state-length growth is removed (h_depth +0.95 → relative −0.84 nats/layer; FSLE
×206 → ×0.46 net), and the output really is length-blind (RMSNorm is exactly scale-invariant; measured margin changes
scale linearly with the perpendicular part, |dm| ≈ 4 × relative nudge). But the nudge itself is **perpendicular to the
state (86–87° at every radius 1e-3…2; along-state share 5–8%, vs 2.6% for a random direction) — a direction change,
not a length change — and relative to the state's length it grows ×5.6 (layers 3–7 arrival), ×1.46 and ×1.33
(layers 7–21), shrinking only in the final band (×0.84, 84% of tokens)**; the cos≈1 that motivated "radial" is a
float32 floor artifact (angles <0.028° read as exactly 0), and the project's own r=2 cosine analysis already said
"tangential". Suggested replacement wording: the *state's length* drifts ×450 with depth and the final normalization
divides length away, so logits see only the tiny relative direction change; the nudge is not a radial length-change,
and in the current w100 data its relative size *grows* through mid-depth — only the old ensemble data and the final
layers show relative contraction, and that mid-depth discrepancy (old FSLE flat vs new data +0.04–0.05 nats/layer)
is the one open question worth resolving before publication.
