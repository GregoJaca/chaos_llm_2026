# Results summary (DeepSeek-1.5B, FP32)

Window 100 unless stated. Prompt `interstellar_travel` (65 tokens), 300 generated tokens. All A1 runs use in-batch `r=0`; duplicate `r=0` rows are bit-identical.

## 0. Corrections to earlier claims (read first)

1. **"The depth expansion is only radial scale drift; every block contracts."** Wrong for the prompt-perturbation response. The state length grows ×452, but the perturbation is **nearly perpendicular to the state (86–87°)**, and its *relative* size grows ×10 across the depth (shrinking only in the final band, ×0.84). The same-position tangent product does contract, but it cannot see the cross-position spreading that produces the response. See section 4.
2. **Teacher forcing is an exact simulator up to the first flip.** One teacher-forced pass on the perturbed prompt reproduces whether the free run flips, the exact step, and the exact new token (29/29 flips, 0 false positives in 140 cases; 111/111 no-flip cases). After the flip the two runs have different prefixes and comparison stops being meaningful. This is a stronger and simpler validation than the linear-threshold scorecard.
3. **Window causality is now demonstrated with identical inputs.** With one fixed token sequence and five windows (100/150/200/300/1000), each curve is bit-identical to the 1000-token control until exactly `t = window − prompt_len`, then departs (35/85/135/235 for windows 100/150/200/300). See `a1_window_fixed_inputs.png`.
4. **Flips are of two kinds.** 17/27 observed flips happen at the response peak ("immediate"); 10/27 happen at a later close call after the response has already faded to <30% of its peak ("delayed", t=116/123/143). A single-token kick at s=16 changes no token immediately but changes the token at t=123, i.e. 107 tokens later. See `flip_is_a_close_call.png`, `state_current_depth.png`.

## 1. A1 response: one universal curve, window-driven regimes

- **Exact linearity in r.** For `r≤0.3`, `δh(t,ℓ;r) = r·G(t,ℓ)`. Collapse CV of `δh/r` is **2.0–3.3%** (median; p90≤11%) at every layer/time/direction. Fitted exponent **α=1.00±0.02**; no time- or layer-dependent exponent is needed. The earlier α≈0.95–0.98 came from including the FP32 floor/large-r tails.
- **The regimes are set by the sliding window.** Shallow layers (L1–3) go bitwise zero at `t*=92,191,290` (spacing 99=window−1: perturbed KV evicted one window later per layer). Deep layers rise to a peak (L27: +0.0225/token, t≈17), then a pre-flush surge (×1.9–2.5 at t≈70–90), then tail decay (−0.011…−0.013/token).
- **Causal window test:** w100 response jumps **×1.73 at t=35** (context exceeds 100 and the oldest prompt token is dropped) while w200/w500 stay byte-identical; w200 jumps **×1.51 at t=135** with w500 as control. Free-run token streams diverge only after clipping (t≈40/142). This is a clean natural experiment.
- **Depth gain:** `A(ℓ)~ℓ^0.64` (R²=0.95; 0.155 at L1 → 1.875 at L27, ×12); front-loaded (1.9×/layer L1→2, ~1.04 mid, 1.12 L21→27). Depth also sets rise time and memory.
- **Directions share the shape and differ by a scalar gain:** gain 1.13–18.97 at L27 (dir10 ≈10× median), max/min 12.9–16.8 at every layer; 72% of log-variance in one mode, but pairwise correlation 0.58 (t<85) vs 0.88 (t<250) — not rank-1; direction-specific token oscillations exist.
- **Large r:** no saturation/new regime; shape survives (CV 13% for r 0.1–2). Deep-layer exponent grows from 0.96 (L1) to 1.45 (L27); per-direction large-r exponent spans 0.47–2.0, uncorrelated with linear gain. Max response at r=2 is ~0.7× base norm, cos≈0.81; bounded.
- **Flip onset:** ~5% of directions at r≈0.2, 35% at r=0.5, 95% at r≥1.5; all observed flips occur at base margins ≤0.33.
- **Caveat:** Qwen2 windows only the last `max_window_layers` (21–27); early layers are full attention, so teacher-forced response keeps a long-lag tail. The free-run generation is the ground truth for the windowed protocol.
- **Numerical floor:** below `r≈1e-4`, response is FP32/RMSNorm ill-conditioning; usable range `1e-3…0.1`, large `r` tested to 2.

## 2. Margins and first-flip threshold: real window effect

Baseline margin spectrum (600 tokens, FP32):

| window | min margin | step | n(margin<0.05) |
|---:|---:|---:|---:|
| 50 | 0.167 | 13 | 0 |
| 100 | 0.0201 | 123 | 2 |
| 200 | 0.0031 | 44 | 8 |
| 500 | 0.0031 | 44 | 11 |
| 1000 | 0.0031 | 44 | 12 |
| 4096 | 0.0031 | 44 | 12 |

The window changes the **margin spectrum**, not just numerics. A smaller margin means a smaller perturbation crosses it.

**Threshold prediction.**
- `m_t = z_t^(1)-z_t^(2)`, `δm_t(u)`; a flip requires `δm_t ≤ −m_t`.
- The physically correct local threshold is **signed**:
  `r_c⁻(t,u) = m_t·r / max(−δm_t,0)`, and `T(r,u)=min{t:r≥r_c⁻}`.
  The symmetric `|δm|` version is adversarial and over-predicts.
- **Two variants, keep them apart.** *Exact-radius*: measure `δm` with one teacher-forced pass at the kick size you want to predict. *Extrapolated*: use `δm` measured at a small radius and scale linearly. The first is nearly exact; the second is only valid up to `r≈0.3`.
- **Exact-radius scorecard: 318/320 direction-runs correct (99.4%)** over 19 free-run experiments and 3 prompts: 57/59 flipping runs predicted to the exact word, 261/261 "no flip anywhere" correct, **0 false alarms, 0 missed flips**, 2 wrong steps. All 70 calls within a factor 2 of the threshold are correct.
- **Intermediate radii (new run, 0.05/0.1/0.2/0.3): 80/80 exact.** 0/20 flips at 0.05 and 0.1; 1/20 at 0.2 and at 0.3 (direction 10, word 16), predicted exactly. The 0.05 case uses the response measured at 0.03 (a 1.7× extrapolation) and is still correct because no direction is near its threshold.
- The previously reported worse score came from the **sign-blind rule** as shipped: on the same 320 runs it gives 45 exact, 240 no-flip correct, 13 wrong steps, **21 false alarms, 1 miss** (89.1%). That is the version that produced "too many outliers".
- The 2 remaining misses are both flips at **word 0**: at very large kicks the perturbed prompt itself is deformed, so the first decision is already different before teacher forcing can see it.
- **Value of the prediction:** a flip is a point of no return. Of 75 flipped runs, **0 ever re-synchronise**; mean token agreement after the flip is 2.7%; the surviving fraction of the 300-word answer equals first-flip position / 300. The 305 no-flip runs are 100% identical. So one teacher-forced pass forecasts how much of the answer survives.
- **When extrapolation breaks (r≥0.5):** the margin response turns superlinear (median gain ×1.55 by r=2) and becomes tie-dependent; steps are then predicted correctly in only 16/20 (r=0.5), 6/20 (r=1), 3/20 (r=2, 2/16 steps exact, worst 129 words). Below `r≈3e-4` the shift is at the float32 round-off floor. Clean linear window: `3e-3…0.3`.
- Window 200/500: the t=44 near-tie (m=0.0031) controls most directions; window 100: t=123 (m=0.020). **w200 and w500 are effectively one experiment** (identical flip lists and reference text).
- Window 50: min margin 0.167; no flips at `r=0.1/0.3`; 1/20 at `r=0.5`.
- Large `r`: 3/20 (r=0.5), 8/20 (r=1), 16/20 (r=2) flip; text stays coherent at 300 tokens (distinct-token ratio ~0.26–0.30).
- **Prompts:** childhood 0/20, 4/20, 12/20 at r=0.01/0.1/0.3 (ties 125/173); cyber 0/20, 0/20, 3/20 (ties 39/51). Same picture, different tie locations and threshold scale.

This is direct evidence for decision-boundary crossing.

## 3. Current-state (single-token) perturbation

New deep run: 8 injection steps (8/16/32/50/75/100/150/200), 24 directions, 4 radii (3e-3/1e-2/3e-2/1e-1), 4 measured layers (7/14/21/27), injection at the input embedding; plus a layer-14 injection run.

- **Linear in r at every position:** fitted slope 0.99–1.00 between r=3e-3 and r=1e-1.
- **Where the kick lands matters much more than its size:** the immediate response at `t=s` varies ×13 across positions (0.26 at s=75 to 3.49 at s=16). `s=16` is a near-tie position of this prompt.
- **The response is fast but not gone:** median half-life 1–7 tokens; a long low tail remains (decay continues to the end of the window).
- **Depth:** the kick is amplified into the last layers (×3–4 from L7 to L27 for s=16). Relative to the model’s own state length, the effect shrinks with depth for ordinary positions (s=50/100/200: ×0.05–0.09 from L7 to L27) but stays flat for the near-tie position s=16 (×0.6) — a sensitive position keeps its relative influence.
- **Effect on the text:** at r=0.1 a kick changes a later top-1 token for only 4/24 directions at s=16 and 1/24 at s=50; at r≤0.03 no token changes at all. A single-token kick is a very weak control.
- **Latency:** the kicks that do act sit latent. All four s=16 directions that change a token change it at t=123, i.e. **107 tokens after the kick**; the s=50 direction changes it at t=143 (93 tokens later). Nothing happens at the kick itself.
- **Injection depth:** a residual kick at layer 14 is **9–36× weaker** at layer 27 than the same-size kick at the input, and its tail is flatter (persistent ≈3e-2 footprint).
- **Directions:** the 24 Gaussian directions are equivalent in the same sense as before (spread of immediate response ×1.2–1.5 at fixed s); no clusters.

Earlier (r=1e-3, 32 directions, 4 steps) results that still hold: PC1 explains 94–97% of log-curve variance; median pairwise correlation 0.94–0.98; `R(t,s)` is not a function of lag (rescaled curves differ ×1.6); bumps are not pinned to margin near-ties.

## 3b. Are perturbation directions equivalent?

**No — and the way they differ is informative.**

- **A single scalar gain does not describe a direction.** Gains span ×7–12 even at r=1e-3 (this is real, not noise: rank correlation 0.80–0.98 across three different continuations, i.e. a direction that is strong stays strong on a different text). A stable "hot trio" (directions 10/11/12) is hot in all three continuations.
- **Shape differs beyond the gain:** 6% of log-variance, split-half reliability r=0.999, significant for 20/20 directions. The old "PC1 explains 94–97%" measures the shared backbone (the token profile of ||δh||, pairwise corr 0.89–0.92), not equivalence; after removing each direction's gain, PC1 is only 0.24.
- **The decisive difference is the sign of the margin shift.** Signed `δm` curves are essentially uncorrelated across directions (r≈0.05); 16/20 directions have a significant sign bias; at the same close call (t=123) different directions push the decision in *opposite* directions (response spans −0.073…+0.253). This is why flips are direction-stratified: at r=0.5 exactly the hot trio flips (3/20), at r=2 16/20.
- **No discrete clusters:** k=1 wins a parametric-bootstrap model comparison (P(k>1)=0.10–0.18). The structure is a continuum plus reproducible outliers, not groups.
- **Random current-state directions are much more equivalent** than prompt directions (gain spread ≤2.1×; rank correlation across injection steps ≈0.04), so "directions are equivalent" was closer to true for the old state data than for the prompt data.

Practical rule: average over directions only when reporting the shared backbone; never assume a single direction's gain or sign applies to another.

## 4. Depth: what actually grows, and what does not

Measured from the same A1 run (prompt perturbation, r=0.1, median over 20 directions and 300 positions):

| layer | state length `|h|` | perturbation `|δh|` | `|δh|/|h|` |
|---:|---:|---:|---:|
| 2 | 43.6 | 4.1e-4 | 9.3e-6 |
| 3 | 51.8 | 1.3e-3 | 2.5e-5 |
| 7 | 76.0 | 1.3e-2 | 1.8e-4 |
| 14 | 107.1 | 2.7e-2 | 2.5e-4 |
| 21 | 199.4 | 6.6e-2 | 3.3e-4 |
| 27 | 545.3 | 1.5e-1 | 2.7e-4 |

- The state length grows **×452** from the embedding to the last layer (×29 inside block 0). The raw perturbation grows **×365** from layer 2 to 27; the *relative* perturbation grows **×10** across the depth and shrinks only in the final band (×0.84 from L21 to L27).
- **The perturbation is not radial.** Measured with the saved full vectors, the angle between the perturbation and the state is **86–87° at every radius** (r=1e-3…2), i.e. almost perpendicular: it changes the *direction* of the state. The "cosine ≈ 1" stored in `a1.npz` is a **float32 floor artifact** (any angle below ~0.03° is stored as exactly 1.0); do not use it to claim radiality.
- **Why the output is still sensitive:** the network normalises the state length before the logits, so a pure length change would be invisible. Consistent with this, the measured margin change tracks the perpendicular part linearly: `|δm| ≈ 4 × (perpendicular length / |h|)`.
- **Same-position tangent product vs full response:** a fixed token position's depth map has aligned gain ≈1.0–1.1 per block while the state scale grows ≈1.1–1.2 per block, so that product contracts relative to scale (max normalised exponent −0.034). The full prompt-perturbation response grows because attention carries the perturbation to other positions. The two objects disagree; report both, and do not use the same-position product to describe the response.
- **Open question:** the old ensemble FSLE shows a flat mid-depth spread (implying relative contraction), while the new teacher-forced data shows +0.04–0.05 nats/layer relative growth. A same-run re-measurement is needed before calling either "the" depth law.

## 5. Non-normality: composition answer

Full block, 6 tokens × 5 layers:
- Layer 0: `σ_max≈95–195`, `ρ≈11–39`, `κ≈5–8`.
- Layers 7–21: `σ_max≈2.7–3.6`, `ρ≈1.3–2.1`, `κ≈1.7–2.5`.
- Layer 27: `σ_max≈6.0–9.1`, `ρ≈1.7–2.0`, `κ≈3.0–5.3`.
- Token 143, layer 14: `σ_max=9.2`, `ρ=1.57`, `κ=5.9` — a genuine non-normal transient spike at a 0.020 near-tie; does not propagate (h_depth mid-pack). Flagged anecdote.
- MLP alone (layer 27): `σ_max≈58–93`, `ρ≈17–24`, `κ≈1.8–3.9`.
- Principal angles: median **83.6°**, IQR 78.7–87.0°, no layer trend; `|cos|` 5× random but `cos²≈0.012`.
- Misalignment budget `(σ−1)cos²θ` with measured σ and angles predicts measured chained gains within ≲2×. The measured product is ~13 orders below the naive σ_max-chain and ~5 below the ρ-chain.
- Interpretation: within one block a component can amplify, but the block composition is strongly non-normal and the amplification direction rotates, so the same-position product contracts. Across the whole network the prompt perturbation still grows, because attention carries it to other positions (see section 4). “Components amplify, composition may not” holds for the fixed-position product, not for the full response.

## 6. D2 / RQA: conclusive negative, and why the curves "looked different" before

- **No scaling region exists on any non-loop trajectory.** The whole rise of `C(ε)` spans 0.16–0.19 decades with local slope 9→1.5; the part that is simultaneously below the finite-N ceiling `2 log10 N` and above the depletion floor `10/N` is **0.00–0.04 decades wide**. A plateau is geometrically impossible for this point cloud.
- **What does *not* change the shape:** centering vs not (Euclidean distances are translation-invariant, max |Δlog C| = 0.00), N (512→4096 only moves the depletion edge), ε normalization, z-scoring, cosine (which makes it steeper).
- **The five "good-looking" shapes that were probably seen before, all reproduced exactly:**
  1. **Repetition loops** — cyber's best 1-decade fit: D2=1.11, r²=0.999 (a loop tube). Repeated text, not chaos.
  2. **First layer** — 352,035 exact duplicate pairs (4.2% of all pairs; 36 states repeated up to 305×) give a flat floor `C=0.042` at every ε.
  3. **Delay embedding with m=2** — textbook: slope 1.95 over 0.96 decades, r²=0.999, "converging" D2(N)=1.89→1.95 — but **white noise at m=2 gives the identical curve**, as do phase-randomised, IAAFT and time-shuffled controls. It is the fill dimension of any 2-D cloud.
  4. **First PC "saturation" at D2≈2.3** — IAAFT nulls land at 1.7–2.4 with the same pattern (effective rank of a red, heavy-tailed marginal).
  5. Artifacts of the `cdist` self-pair floor and a linear ε grid (staircase).
- **The detector is not the problem:** at the same N it measures Hénon (D2=1.157, flat over 2.57 decades) and Lorenz (D2=1.984, r²=1.000) cleanly.
- Theiler windows change nothing for the raw cloud except removing loop-copy pairs below C<1e-4 — exactly the repeated-point mechanism.
- Non-loop dynamics: `interstellar` is a clean negative (kNN NSE equals its linear ceiling); `launch` has a suggestive +0.15 NSE over its linear ceiling but no continuation sharpness (one trajectory).
- **Dataset issue:** three of the first five 1000-token trajectories are near-exact loops (P=53/74/69; 86–91% of states recur within 5%). They are excluded from dynamics; 16 fresh 1000-token trajectories were collected and screened.

Conclusion: D2/RQA carry no evidence of low-dimensional chaos here. Every quoted D2 value is a window/tube/fill-dimension/duplicate artifact. Settling it would need ~10⁵-step non-repetitive trajectories.
- **Raw vs time-shuffled D2 is a tautology:** time-shuffling preserves the point cloud, and D2/TwoNN/MLE/PCA are point-cloud statistics. The earlier “raw≈shuffled” control cannot test dynamics.
- Static geometry: locally thin (TwoNN 3.3–6.7, MLE 2–4) but globally spread (90% variance needs 161/223 PCs) → inhomogeneous token-state cloud, not an attractor. TwoNN/MLE have strong finite-N bias (iid-Gauss ~100).
- RQA DET 0.37–0.95, Lmax up to 1775; dominated by temporal correlation and loop copies.

## 7. What is different from before

| Before | Now |
|---|---|
| Batch-1 baseline vs batch-N perturbed, low precision | In-batch `r=0`, FP32, verified |
| Average `T(r)` | Margin/response threshold prediction validated across windows |
| Distance-curve FTLE | Teacher-forced response, FSLE, depth spectrum |
| Per-component Jacobian norms | Product singular values, `ρ`, `κ`, principal angles |
| D2 with no controls | Chunked D2 + shuffled/phase/vocab controls; static-geometry negative |
| “Looks chaotic” | Transient response + decision-boundary crossing |

## 8. Limitations

- One model; three prompts (interstellar, childhood, cybersecurity); windows 50–500; 300 tokens (margins 600; A1 500 for one prompt).
- Depth spectrum top-8/16 only; `h_depth` lower bound. Scale-normalized conclusion is robust across 8 tokens.
- Full-block non-normality at 5 layers × 6 tokens; MLP same.
- State injection at input embedding and one middle layer; r-scan only 1e-3…3e-2.
- D2 N=2–3k; the negative is about this range.
- 600-token A1 hit the full-vocab logits memory ceiling; 500 tokens works with cache clearing.
- Teacher-forced forwards use Qwen2’s `max_window_layers` behavior; early layers are full attention, so long-lag tails persist. Free-run generation is the windowed ground truth.

## 9. Files

- Figures: `results/figures/INDEX.md`
- Measurements/formulas: `docs/MEASUREMENTS.md`
- D2/RQA method: `docs/D2_RQA_RESULTS.md`
- Decisions: `docs/DECISIONS.md`
- Data/code map: `docs/DATA_AND_CODE_MAP.md`
