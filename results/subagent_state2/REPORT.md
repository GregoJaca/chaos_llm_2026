# state2 deep-perturbation analysis — REPORT

Date: 2026-11-18. Data: `results/local_data/state_deep_emb/state2.npz` (input-embedding injection,
4 radii × 8 steps × 24 dirs × 4 layers × 300 tokens), `state_freerun/sf.npz` (autoregressive, r=0.1,
s=16/50), `state_deep_L14/state2.npz` (layer-14 injection, r∈{0.01,0.03}, 16 dirs), cross-checked
against the old `state.npz` (r=1e-3, 32 dirs, layer 27). All teacher-forced unless marked
"autoregressive". Convention fixed by the data: `dh[s,t]` is the hidden-state difference at
*position* t; `dm[t]`/`margins[t]` index the *token* they decide (read from logits at t−1), so
`dm[t]` responds for t ≥ s+1 and flips a token iff `dm[t] < −margins[t]`. In freerun, `pred[t]=−1`
for t ≤ s (fed, not generated).

**Indexing note:** the earlier claim "s=16 gap = 0.0696" does not match this dataset:
`margins[16] = 0.120` (7th smallest of 300; median margin 4.78). The value 0.0696 is
`margins[2]`. s=16 is still a genuine near-tie (2nd percentile of margins), and everything else
about the established picture reproduces (×14–14.4 head variation, 4/24 and 1/24 flips at
t=123/143, exact autoregressive agreement). Also: `dh` contains small pre-injection artifacts
(single entry 9.1e-3 at dir 0 / L27 / t=19, others ≤ 8e-4, identical in both injection files —
≤0.3% of typical responses); all analyses mask t < s.

## Bullets

1. **Decay law (fig1): no single exponential ever wins.** Fits on log-response vs lag τ=t−s
   (τ ∈ [0,99], mean of 24 dirs, r=0.1, all 8 s × 4 layers): **exp+floor wins 23/32 by AIC,
   26/32 by BIC**; two-timescale 6/3; power law 3/3; pure 1E never. Fast timescale
   **T = 1.6–11.9 tokens**, then a **floor at 8–18% of the immediate response** that does not
   measurably decay within 100 tokens. Example fits: s=16 (T=1.6, floor 16%), s=75 (T=5.8,
   floor 15%), s=150 (T=11.9, floor 17%). Full table: `decay_fits.csv`.

2. **Separability (fig2): response ≈ a(s)·b(τ) to first order, and the earlier "not a function
   of lag" claim is confirmed and sharpened.** SVD of the centered log-response field (8 s × 100
   lags, L27): **first singular value = 68% of variance** (52% at L7 — depth makes the field *more*
   separable). The non-separable 32% is structured, not noise: (i) the first ≤4 lags, where the
   retained fraction at τ=10 spans **9.9× across s** (s=32 keeps 6.5%, s=50 keeps 64%), and
   (ii) echo stripes at fixed t. Collapse test: retained fraction at τ=50 spans 0.035–0.331
   (**9.5×**) — shapes are not functions of lag alone.

3. **Echoes (figs 3,4,7): the tail is carried by narrow "recall" bumps at content positions,
   not by smooth decay.** At t≈63, 86, 105, 113, 124 (s-dependent), dh jumps 3–8× above the
   local envelope; s=8's echo (5.6 at t=105) exceeds its own immediate response (2.6). Echoes
   are **exactly as linear in r as the head** (dh(0.1)/dh(0.003) = 33.1–35.0 vs 33.3 expected),
   appear already at layer 7, and are amplified **8–28× by layers 21→27** while immediate heads
   gain only ×1–4.2. So the deep network is mainly an *echo amplifier*.

4. **Where the memory lives (fig3): same directions, sideways to the state, coupled to logits
   at every lag.** The directions that respond biggest at τ=0 also carry the tail (Spearman
   +0.40…+0.89 for 7/8 s; exception s=150, whose head itself oscillates). 1−cos shows the
   perturbed state stays a near-copy of the baseline (cos > 0.9997, i.e. the response is a small
   sideways displacement, never a rotation). The margin response |Δm| tracks ‖Δh‖ at all lags
   (Spearman +0.48 near the head rising to +0.65 in the tail, n≈42k pairs). Hidden bumps and
   decision fragility are different things: dh(t) only mildly anticorrelates with margins(t)
   (−0.48); the big echo at t=113 sits on a large margin (2.6) and flips nothing.

5. **Depth (fig4): the ordinary response is finished early; depth adds the kick and the echo.**
   At τ=0, layers 7/14/21 hold 57–104% of the final-layer response for ordinary s, but only
   **24/25/42% for s=16** — the near-tie kick is built late (L21→L27). Spread across s grows
   **3.8× (L7) → 14.4× (L27)**; in the tail the L7/L27 share drops to 0.13–0.21 everywhere.
   Baseline state norms estimated from cos (valid because the response is ⊥ to the state):
   ‖h‖ ≈ 66–89 (L7), 96–112 (L14), 181–216 (L21), 470–617 (L27). Relative to ‖h‖, the s=16 kick
   stays ~flat with depth (4.3%→2.7%) while ordinary positions shrink ×3–10 — confirming and
   quantifying the earlier depth claim.

6. **Flips (figs 5,7): the mechanism is an exact threshold crossing, and it predicts the
   autoregressive runs perfectly.** A token changes iff the signed margin response crosses the
   baseline gap, `dm[t] < −margins[t]`. This rule predicts the freerun flips **5/5 with zero false
   positives out of 48 runs** — dirs 12,13,17,23 at s=16→t=123 and dir 13 at s=50→t=143 — which
   are precisely the **two smallest margins in the whole 300 tokens** (0.0202 and 0.0204),
   latencies 107 and 93 tokens after injection. Closest non-flipping directions stop **12%/14%
   short** of the threshold. After a flip the continuation never re-syncs with the reference
   (2–3% token agreement). 43/48 runs never change a token.

7. **Immediate response does NOT predict flips (fig5).** AUC of head ‖Δh‖(τ=0) for flip vs
   no-flip directions: **0.46 at s=16 (CI 0.09–0.88, ranks 1,9,22,21 of 24), 0.83 at s=50 (n=1,
   CI 0.65–0.96), 0.69 pooled (CI 0.47–0.89)** — indistinguishable from chance to weak. What
   predicts a flip is the margin response at the fragile position itself (Spearman with head
   size only −0.04 at s=16): it is a readout-alignment question, not a response-size question.

8. **Injection depth (fig6): "later injection is weaker but flatter" is half wrong.** At matched
   r=0.03, layer-14 injection is **11–38× weaker** at L27 (head ratios 0.026–0.091) — the weakness
   is created by blocks 0–13, which amplify an input-embedding kick to 9–32× r by L14 (a direct
   L14 kick stays exactly r at L14) while blocks 14–27 treat the two similarly (s=16: ×4.0 emb vs
   ×3.3 direct; s=50: ×1.1 vs ×0.9). But it
   is **not flatter**: tail/head ratio is 0.5–1.0, and the fitted floor is *smaller* (4–16% vs
   8–19%). Blocks 0–13 create most of the persistent tail too. The layer-14 margin response
   (max |Δm| = 0.002) is 10× below the smallest margin — structurally unable to flip a token at
   this r.

9. **The field in one picture (fig7):** ‖Δh(s,t)‖ at L27, r=0.1: head values vary ×14
   (0.89–12.83) along the causal boundary; vertical echo stripes; a flip-fraction field that is
   nonzero only at (16→123) and (50→143); aggregate median decay **T = 4.7 tokens with an 11%
   floor**. Head size is only partially explained by the margin at s (s=16: margin 0.12 → head
   12.8, but s=32: margin 9.8 → head 3.1, larger than s=50's 0.96 with margin 7.2).

10. **Cross-checks:** old r=1e-3 data reproduces the tail pattern (fraction of head at t=299:
    s=16 3%, s=50 18%, s=100 6%, s=150 11% — same ordering as the new data); linearity in r holds
    everywhere including at echo peaks (ratios 3.33 ± 10–15%, echo ratios 33.1–35.0 vs 33.3).

**Best figures:** `fig1_decay.png` (model comparison), `fig5_directions_latency.png` (exact
flip prediction + honest effect sizes), `fig6_injectdepth.png` (falsifies "flatter"),
`fig7_response_field.png` (the overview).

## What remains uncertain

- **n is tiny for flip statistics** (4 + 1 events): every effect size on "which directions flip"
  has a CI that touches or crosses chance; only the threshold mechanism itself (exact 5/5) is solid.
- **Floor vs slow decay:** within a 300-token sequence I cannot distinguish a true constant floor
  from a ~300-token slow component (two-timescale fits put T_slow ≈ 300 for s=100; power law wins
  for s=200). Needs a longer generation.
- **Echo mechanism not identified:** the bumps are consistent with attention recalling the
  perturbed position, but which heads/circuits produce t≈105/113/124 needs attention-pattern or
  activation-patching work, not norm data.
- **Base-state norms are estimates:** they assume the response is orthogonal to h (true for random
  directions, consistent IQR 1–3% across dirs) and are only available at the 8 sampled injection
  positions; cos is float32, so 1−cos below ~1e-6 is quantization garbage (this limits the angle
  analysis exactly where responses are weakest, e.g. s=75/100 tails).
- **The autoregressive confirmation covers only s∈{16,50} at r=0.1** — the dm<margins rule is
  verified there, assumed elsewhere.
- **s=150 is an oddball** (head oscillates: dh(τ=3) > dh(τ=0); direction persistence ≈ 0) — one
  position, no explanation yet.
- **Mild nonlinearity in r** (±10–15% at the 1st/99th percentile of the 0.1/0.03 ratio) — the
  linear-response picture is excellent but not exact at r=0.1.

## Summary for the main agent

The deep state2 data turns the earlier qualitative picture into a mechanism with numbers, and it
falsifies one claim. The hidden-state response to an input-embedding kick is never a single
exponential: everywhere it is a fast drop (T ≈ 2–12 tokens) plus a persistent floor of 8–18% of the
immediate response (exp+floor wins 23/32 by AIC, 26/32 by BIC), and the field is roughly separable
a(s)·b(τ) (68% rank-1) with the non-separable part being (i) s-dependent head-relaxation shapes
(×10 spread at τ=10, so "not a function of lag" is confirmed) and (ii) narrow, content-position
"echo" bumps that are exactly as linear in r as the head, present at layer 7, and amplified 8–28×
by the late layers — the deep network is mainly an echo amplifier, and the near-tie kick (s=16)
is also built late (only 24% of it exists at L7, its relative size stays ~flat with depth while
ordinary responses shrink ×3–10). The memory is carried by the same directions that responded
immediately (Spearman +0.4–0.9) and stays coupled to the logits at every lag (Spearman 0.48→0.65),
but big responses do not flip tokens by themselves: flips happen exactly where the *baseline
margin* is tiny — the rule `dm[t] < −margins[t]` predicts every autoregressive divergence 5/5
with zero false positives, landing at t=123 and t=143, the two smallest margins in the whole
sequence, 93–107 tokens after injection — and the immediate-response magnitude does not predict
which direction flips (AUC 0.46 at s=16, 0.69 pooled, CIs touching chance). Comparing
input-embedding vs layer-14 injection at matched r: the "weaker but flatter" claim is half wrong
— later injection is 11–38× weaker (blocks 0–13 do the amplifying) but retains *less*, not more
(floor 4–16% vs 8–19%), and its margin response is 10× too small to ever flip a token at this
radius. Key open questions: which circuits make the echoes, whether the floor is a floor or a
~300-token tail (needs longer sequences), and s=150's oscillating head.
