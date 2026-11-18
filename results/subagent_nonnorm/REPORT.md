# Non-normality & depth composition — re-analysis (subagent_nonnorm)

**Scope.** CPU-only re-analysis of the existing non-normality/depth data of DeepSeek-R1-Distill-Qwen-1.5B (FP32, window 100, prompt `interstellar_travel`, 28 blocks, d=1536):

- `results/local_data/depth/depth.npz` — QR/Benettin tangent spectrum of the **diagonal-block product** (same-position block Jacobians), 8 tokens × 28 blocks × top-8;
- `results/local_data/nonnorm.npz` — per-block `σ_max`, `ρ`, `κ=σ/ρ` (MLP at blocks 0/7/14/21/27; full block at 0/14/27) and principal angles between consecutive top amplification directions, 6 tokens;
- `results/local_data/depth/fsle_depth.json` — depth-separation curves (OLD perturbation runs), 4 radii;
- `results/local_data/w100_layers/a1.npz` — **context only**: baseline state norms `norm_h0` (8 sampled layers × 300 positions), used for scale normalization, plus a fixed-position depth profile of the teacher-forced response (the t-axis response regime analysis belongs to the A1 subagent and is not duplicated here).

Everything below is reproducible with `analyze.py` → `figures.py` (numpy/scipy/matplotlib only). All logs/numbers: `analysis_log.txt`, `summary.json`.

---

## 1. Headline: the depth "expansion" is the residual-stream scale drift, not an instability

The residual-stream norm grows monotonically with depth: ‖h‖ ≈ 1.2 at the embedding, **×30.3 (median; range ×19–36) inside block 0**, then ×1.05–1.18 per block, reaching **×441–591 over the full depth** (per-token positions). This is a designed property of pre-norm residual streams (each block *adds* content; the final RMSNorm divides it out before logits).

Measured *absolute* gains along the QR-propagated basis: ×10.3–25.4 in block 0 (median ×17.4), then ≈1.0. That is the current "first layer expands, then constant" story. **Divided by the local scale ratio, every block contracts**:

| blocks | aligned gain | scale ratio | **relative gain** |
|---|---|---|---|
| 0 | ×17.4 | ×30.3 | **×0.55** (range 0.48–0.72) |
| 1–6 | 1.017 | 1.100 | **0.910** |
| 7–13 | 0.997 | 1.056 | **0.944** |
| 14–20 | 1.013 | 1.091 | **0.931** |
| 21–27 | 1.054 | 1.184 | **0.886** |

Consequently, subtracting the per-layer log scale rate from the finite-time depth spectrum makes **every exponent negative**: the maximum relative exponent over all 8 tokens × 28 depths × top-8 directions is **−0.034 nats/layer**. Since Benettin exponents are ordered, directions 9–1536 are smaller still (with the usual QR-convergence caveat). The product's top singular value grows ×10.3–166 absolute but **×0.02–0.33 in scale-normalized units** — a ×9–×50 *relative contraction*. Same for the depth FSLE: total growth ×203–215 absolute becomes **×0.45–0.48 relative** (Fig. 2d). A random prompt perturbation (×206–262) and the state scale (×531) agree to within a factor ≈2.5: the perturbation rides the radial drift.

This reframes the reviewer answer (R1.1/R2.2): *the positive finite-time depth exponents (λ₁ ≈ 0.08–0.18, h_depth ≈ 0.64–1.36) are the entropy of a monotone radial drift that (i) every trajectory undergoes — it is not sensitive dependence — and (ii) the final RMSNorm projects out before the logits (scale invariance, up to the norm's ε). The output-relevant dynamics live on the sphere ‖h‖ = const, and there every top-8 exponent is negative.*

Supporting measurement: the teacher-forced response is **almost entirely tangential** — at r=2 (where the cosine is numerically resolvable), ≥93% of ‖δh‖ is perpendicular to h at every probed layer ≥2. Radial growth is a gauge direction; angular deviation of the state is ~10⁻³ rad at the last layer and does not compound exponentially (Fig. 8b).

## 2. Transient vs sustained: the layer-0 jump is 82% of the absolute story — but it is a scale-lift, and what remains is marginal

- One-block entropy spike: h(block 0) = **18.5–25.8 nats** (top-8 sum) in a single block; the whole-depth average h_depth is 0.64–1.36. Transient share of the top-1 log-gain, S(0)/S(27): **0.63–0.98 (median 0.82)**.
- Depth FSLE: the separation reaches (slightly overshoots) its final value by recorded layer 2 (share 1.13–1.19) and the threshold time is τ=1 at ρ=1.5/2/4 for *all* radii → λ = ln ρ, i.e. a one-layer transient, r-independent.
- **But "then constant" is subtly wrong:** after block 0 the aligned subspace still drifts upward at **+0.018 nats/layer median** (top-8 mean over blocks 1–27; range −0.002…+0.057; top-1 median +0.021), with real layer structure — e.g. token 0 has a +0.18-nat bump at block 19, token 50 a −0.13 dip at block 25 (Fig. 1d, Fig. 3a). This sustained drift is *smaller than the scale growth at every depth band* (0.05–0.17 nats/layer), so relative to the stream it is a contraction of −0.04…−0.125 nats/layer (median −0.08).

**Answer to "is the layer-0 jump the whole story?":** for absolute gains, yes (63–98%); dynamically, no — block 0 is a scale-lift (relative ×0.55), the tail has genuine, token-dependent, signed structure at the ±0.05-nat level, and all of it is absorbed by the stream-scale growth.

## 3. Non-normality: components amplify, the block tames them, and the direction rotates

- **Full block:** block 0: σ_max = 94.6–194.8, ρ = 11.6–38.6, κ = 5.0–8.2. Blocks 14: σ_max ≈ 3.10–3.35, ρ ≈ 1.26–1.62, κ ≈ 1.9–2.5. Block 27: σ_max ≈ 6.0–9.1, ρ ≈ 1.7–2.0, κ ≈ 3.0–5.4.
- **MLP alone** (same tokens/layers): σ_max median rises 11.5 → 10.8 → 12.6 → 18.2 → **71.3** (block 27; range 58–93), with ρ up to 24.4 and κ 1.8–3.9. The MLP is *linearly unstable on its own* (ρ≫1) at every sampled layer. The full residual block damps σ by **≈9×** and ρ by **≈13×** at block 27; chained naively, the late MLP alone would give σ_max²⁷ ~ 10⁵⁰ (ρ-chain ~10³⁷).
- **Late-layer expansion is systematic and compensated:** both MLP (18→71) and full-block (3.2→7.9) σ_max rise in late layers, and the aligned gain band 21–27 rises to 1.054 — but the scale ratio rises faster (1.184), so the *relative* gain is deepest in late layers (0.886). "Late-layer MLP expansion vs full-block damping" resolves as: components amplify more late, the stream grows even faster.
- **Token-143 / block-14 outlier, quantified:** σ_max = 9.21 vs 3.10–3.35 for the other five tokens, with a *normal* ρ = 1.57 but κ = 5.87 (vs 1.9–2.5). A genuinely non-normal, token-specific transient-growth spike (3× σ at unchanged eigenvalues). Its margin is 0.020 (a near-tie, t=123/143 are the two smallest margins in the run) — a candidate link between forking tokens and one-step non-normal amplification. **It does not propagate:** h_depth(t=143) = 1.01, mid-pack; treat as a flagged anecdote until scanned over more tokens.

## 4. Principal angles and the concrete cancellation mechanism

- Angles between consecutive top amplification (right-singular) directions: **median 83.6°, IQR 78.7°–87.0°**, min 54.5° (t=16, pair at L25), max 90.0°; **no layer trend** (+0.036°/block, p=0.66). They are *more aligned than a random pair* (|cos| median 0.111 vs 0.020 expected in ℝ¹⁵³⁶) — a 5× overlap — but still nearly orthogonal.
- **Mechanism, quantified:** a rank-one amplifier with σ_max and pairwise angle θ chains only (σ−1)cos²θ per block. With the measured cos²θ = 0.0124 (median) and the measured full-block σ_max (3.2 at L14, 7.9 at L27), the predicted sustained log-gain is **+0.028 / +0.082 nats/layer**, vs measured aligned gains **+0.013 / +0.052** (medians; means 0.028/0.044). The model overpredicts by ≲2×, as expected — it assumes every block amplifies at its σ_max while the QR basis settles onto the *typical* amplifying directions. The angle distribution alone reproduces the measured per-layer gain distribution (Fig. 7b).
- Gain budget (nats/layer, medians): component log σ_max = 1.18 (L14) / 2.07 (L27); misalignment leaves 0.01–0.05; the stream scale takes 0.09–0.17. For block 0: component 4.89, aligned 2.79, scale 3.41 — the "jump" is 60% of the component bound and *below* the scale.
- Chained bounds at depth 28 (Fig. 5b): naive σ_max-chain (block-0 measured, ×3.1 for the remaining 27 — the *tamest* measured block) ≈ **2.5×10¹⁵**; ρ-chain ≈ **5.5×10⁶**; measured aligned product ≈ **×27.6** (range 10.3–166); FSLE ×206; scale ×531. The measured composition is ~13 orders below the component bound and ~5 orders below even the eigenvalue bound — the product behaves as if driven by neither σ_max nor ρ, exactly the "successive amplification directions decorrelate" geometry (Trefethen et al. 1993).

## 5. Expansion entropy

- h(ℓ) = Σᵢ max(λᵢ(ℓ),0) (top-8 lower bound) spikes to 18.5–25.8 nats at block 0 and decays ≈ A/(ℓ+1) (Fig. 6a) — the signature of a one-block transient plus a small residual.
- **Scale-normalized expansion entropy is identically zero**: the largest relative exponent is negative at every depth for every token (Fig. 6b); Σᵢλᵢ^rel(28) = **−0.42…−1.15 nats/layer** (median −0.84) vs h_depth = +0.64…+1.36 (Fig. 6c). Under Hunt–Ott's definition restricted to the output-relevant (post-RMSNorm) directions, the depth map has *no* expanding direction.
- Token variation of h_depth (0.64–1.36, 2.1×) is largely set by the block-0 gain (Spearman 0.79, p=0.02, N=8); its (exploratory) trend with margin is negative but insignificant (−0.45, p=0.26). Near-tie tokens (123, 143; margins 0.020) sit mid-range.

## 6. Context: the cross-position channel (where the flip-relevant response comes from)

The diagonal-block spectrum misses attention-mediated propagation from the prompt to downstream generated positions. From the A1 run (r=0.03, linear regime, fixed output position): the response *appears* at the first generated position already at 0.16·r after block 0 and grows through depth to **O(r)** at the last layer (t=0: ×12 from L2→L27; t=16: →4.4·r) — this is the channel that matters for margins/flips. But in scale-normalized (angular) units it stays at 10⁻⁴–3×10⁻⁴ of ‖h‖ and is *marginal and token-dependent* (t=0 declines ×0.5; mid tokens grow ×2–10 from a tiny start; no systematic exponential). So the "components amplify, composition does not" conclusion holds in both channels; the amplification-to-parity is an attention-cone accumulation, not an instability.

## 7. What is robust / transient / new

**Robust (across all probed tokens/radii):**
1. All relative (scale-normalized) exponents negative — 8 tokens × 28 depths × 8 directions, max −0.034; consistent with the FSLE curves (×0.45–0.48 relative) and the tangential response structure.
2. One-layer transient: FSLE τ=1 at every ρ and r; h(ℓ) ∝ 1/(ℓ+1); transient share 0.63–0.98.
3. Near-orthogonal rotation: median angle 83.6° (IQR 78.7–87.0), no layer dependence; |cos| 5× random but cos² ≈ 0.012.
4. Late-depth component rise (MLP σ 18→71, full 3.2→7.9) with compensating scale growth.

**Just a one-layer transient (do not build claims on it):** the entire ×206–262 FSLE growth; the h(0) ≈ 22-nat spike; σ₁(ℓ) curves' early elevation. These are scale-matching between the embedding scale (‖h₀‖≈1.2) and the residual-stream scale.

**Genuinely new (not in the current results docs):**
1. The scale decomposition itself: the positive depth spectrum is the entropy of the radial drift, and the final RMSNorm makes that direction output-irrelevant. This is the conclusive, quantitative version of "components amplify, composition may not."
2. Separation of the sustained +0.02-nat/layer aligned drift from the block-0 transient (previously conflated into "constant").
3. The misalignment budget: (σ−1)cos²θ from *measured* angles and σ_max reproduces the measured aligned gains within a factor ≲2 — a concrete, predictive cancellation mechanism.
4. Late-depth compensation: component amplification rises with depth and the stream-scale growth rises faster (relative gain deepest at 0.886 in blocks 21–27).
5. The token-143/block-14 non-normal spike at a near-tie margin — a candidate forking-token event (anecdote, flagged).

**Best figures:** `fig02_scale_normalization.png` (the headline answer-figure), `fig05_gain_budget.png` (the quantified composition budget), `fig07_toy_mechanism.png` (mechanism), `fig04_nonnorm.png` (σ vs ρ with κ guides + angles), `fig03_gain_heatmaps.png` (layer/token structure at a glance).

## 8. Theory hooks (for the writing)

- **Trefethen–Trefethen–Reddy–Driscoll (1993) / Farrell–Ioannou:** the model's blocks realize the canonical non-normal geometry — σ_max ≫ ρ (κ up to 8) with amplification directions rotating ~84° per step; the product is governed by neither eigenvalues nor singular values.
- **Poole et al. (2016):** the +0.02-nat/layer sustained aligned drift is the trained-network counterpart of their transient-chaos log-singular-value drift; here it is absorbed by the scale growth (their χ₁ budget should be defined *on the post-RMSNorm sphere* to predict our relative exponents ≈ −0.08/layer; this is the natural refinement of the λ_depth ≈ ½ln χ₁ formula in the revision plan).
- **Hunt–Ott expansion entropy:** h_depth as currently computed is a lower bound *contaminated by the gauge drift*; the relative version is identically zero-expanding. Suggest reporting both and defining the depth map's entropy post-RMSNorm.
- **Noci et al. (2022) / signal propagation:** the ×1.1/layer norm growth and the tangential response staying at ~10⁻⁴ relative is a "radial collapse" statement that complements their rank-collapse discussion.
- **FSLE (Aurell et al.) / stable chaos:** λ = ln ρ with τ=1 layer, r-independent, is precisely the transient signature; it supports the reframing "sensitivity without instability" — unpredictability must come from margin crossing, not from the smooth depth map.

## 9. Limitations (do not overclaim)

- Top-8 of 1536 directions (h is a lower bound); the "all relative exponents negative" claim extends to all directions only via the Benettin ordering argument, with QR convergence caveats.
- The depth product is the **diagonal** (same-position) block product; cross-position propagation is probed only through the A1 prompt-perturbation response at 8 sampled layers. A full cross-position Jacobian product remains the main missing measurement.
- Scale normalization uses log-interpolated norms at the 8 sampled layers (smooth monotone profile; interpolation error ≪ the 0.05–0.17-nat/layer effects). The FSLE OLD-run provenance (recorded position; d(0)=√2·r normalization) is partially undocumented — used qualitatively only.
- N=8 depth tokens, 6 non-normality tokens, one model, one prompt (window 100). All token-level correlations are exploratory.
- The tangential/radial split is measurable only at large r (fp32 cosine); quoted at r=2.
- Token-143/block-14 is a single event; do not claim a forking-token–non-normality link without scanning more near-tie tokens.

## 10. Files

- `analyze.py` → `derived.npz`, `summary.json`, `analysis_log.txt` (all numbers above).
- `figures.py` → `fig01`–`fig08` (PNG, 150 dpi).
- `qa_figures.py` — blind QA (ink coverage per panel).
- Figures: 01 depth growth · 02 scale normalization (headline) · 03 gain heatmaps · 04 non-normality · 05 gain budget · 06 entropy · 07 toy mechanism · 08 cross-position context.

---

## One-paragraph summary for the main agent

The "first layer expands, then constant" result is an artifact of measuring absolute norms in a residual stream whose scale grows ×30 in block 0 and ×1.05–1.18/layer thereafter (×441–591 total): normalized by the local state scale, **every block contracts (×0.48–0.72 at block 0, ×0.89–0.95 after), every top-8 depth exponent at every depth becomes negative (max −0.034 nats/layer), the expansion entropy h_depth becomes −0.4…−1.2 nats/layer relative, and the depth-FSLE's ×206 absolute growth becomes ×0.46 relative** — the "expansion" is a radial drift that the final RMSNorm projects out before the logits, i.e. a gauge direction, not sensitive dependence. What survives normalization is a small, genuinely sustained aligned drift of +0.02 nats/layer (signed, token-dependent, with a systematic late-depth rise of component gains — MLP σ_max hits 58–93 at block 27 — that the accelerating scale growth exactly absorbs), and the composition sits ~13 orders below the naive σ_max-chain and ~5 orders below the ρ-chain because consecutive amplification directions rotate ~84° (cos²≈0.012): the measured angles and σ_max *predict* the measured chained gains within a factor ≲2 via (σ−1)cos²θ. The one token-layer outlier (t=143, L14: σ 9.2 with normal ρ, κ=5.9, at a 0.020 near-tie margin) is a clean non-normal transient-growth event that does not propagate — a flagged forking-token anecdote. Suggested paper moves: report relative (post-RMSNorm) exponents next to absolute ones, present `fig02`/`fig05`/`fig07` as the quantitative answer to "components amplify, composition may not", and keep all one-layer-transient quantities (FSLE τ=1, h(0)≈22 nats) explicitly labeled as scale-matching, not chaos.
