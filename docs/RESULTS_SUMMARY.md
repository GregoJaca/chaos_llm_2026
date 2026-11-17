# Results summary (first complete pass)

DeepSeek-1.5B, FP32, window 100, prompt `interstellar_travel`, 300 generated tokens.
All A1 runs use the same 20 simplex directions; `r=0` is a row in the same batch.

## 1. Baseline and A1

- Baseline is reproducible; duplicate `r=0` rows are bit-identical.
- Response `||δh_t||` at the last layer grows for ~60–100 generated tokens, then decays.
  - `r=0.3`: peak ≈1, decay to ≈0.1 at t=300.
  - `r=1e-3`: ≈1e-3 throughout.
- There is a numerical floor below `r≈1e-4` (FP32/RMSNorm ill-conditioning): response no longer scales with `r`. The usable linear range is roughly `r=1e-3…1e-1`.
- No sustained exponential growth on the token axis.

## 1b. Current-state perturbation and R(t,s)

- Input-embedding perturbation at generated step `s=16,50,100`, `r=1e-3`, 8 directions.
- The response `||δh_t||` starts at ~0.1 at `t=s` and decays; `R(t,s)` (median) starts near 1 and decays to ~0.1–0.3. There is a bump near t≈130 for `s=50` (a near-tie), not sustained growth.
- This complements the prompt perturbation; the current state is also not exponentially unstable.

## 2. Margins and first-flip threshold

- Window 100: smallest margins t=123 (0.0202), t=143 (0.0204), t=116 (0.069), t=2 (0.0696), t=16 (0.120).
- Window 200: smallest margins t=44 (0.0031), t=183 (0.0047), t=216 (0.0055), t=55 (0.0089).
- Window 100: **no flips at r=0.03 or 0.1** for 20 directions; at r=0.3, 1/20 flips at t=16. Prediction: 1 at 16, 1 at 123.
- Window 200: at r=0.1, 3/20 flip at 44/55; at r=0.3, ~12/20 flip at 44. Prediction at r=0.3: 13/20 (11 at 44, 1 at 55, 1 at 16). Very close.
- This is a real attention-window effect on the **margin spectrum**: window 200 has a much smaller margin at t=44, so the same r crosses it. It is not a numerical artifact.
- The first flip is at the near-tie with the smallest ratio `m_t/|δm_t|`, not necessarily the smallest margin. This is direct evidence for decision-boundary crossing, not smooth exponential instability.

## 3. FSLE

- Time axis: `λ≈0.043` for `ρ=2` over the initial transient; `ρ=4` is mostly censored. It is a transient growth rate.
- Depth axis (existing single-token data): `d(layer)/d(0)` jumps by ~200–300 at layer 1, then is flat/decaying. FSLE is trivially `lnρ/1` and flat in `r`. The depth “expansion” is a one-layer input transient.

## 4. Depth spectrum and expansion entropy

- Top-8 depth exponents are all positive, final `λ_i≈0.08–0.18`; `h_depth≈0.6–1.4` (lower bound).
- Local layer gain: layer 0 ≈10–20; layers 1–27 ≈1.1.
- The cumulative exponent starts at ~3 and decays to ~0.1. So the depth map is **transiently expanding at the input, then marginally expanding**.
- This is a concrete, theory-relevant object: vertical expansion without sustained large λ.

## 5. Non-normality

- Full block, layer 0: `σ_max≈95–194`, `ρ≈11–39`, `κ≈5–10`.
- Full block, layer 14: `σ_max≈3.1–3.3`, `ρ≈1.34–1.62`, `κ≈1.9–2.5`.
- Full block, layer 27: `σ_max≈6.7–9.1`, `ρ≈1.70–1.85`, `κ≈3.6–5.3`.
- MLP alone is much more amplifying: layer 27 `σ_max≈59–95`, `ρ≈17–24`, `κ≈3.3–3.9`; layer 14 `σ_max≈12`, `ρ≈4.5`; layer 0 `σ_max≈8–14`, `ρ≈2.5–5`.
- **Principal angles between consecutive top singular subspaces: median ≈79–84°.** The expanding direction is almost completely rotated at each layer.
- Interpretation: the MLP amplifies strongly, but the full residual block is much tamer (`σ_max` 6–9 vs 60–95), and the product can contract because the expanding subspace rotates. This is the quantitative answer to “components amplify, composition may not.”

## 6. D2 / RQA (negative)

- No stable local-slope plateau. First-layer D2 ≈13–27, far above the Eckmann–Ruelle bound 6.6 at N=2000; last-layer D2 ≈3–10; vocabulary null ≈5.35.
- Theiler 0/10/50 does not help.
- RQA DET ≈0.37–0.95; Theiler 20 barely changes it. This is temporal correlation, not evidence of deterministic chaos.
- **Conclusion:** D2/RQA are not reliable evidence for low-dimensional chaos at the accessible N. Report as negative.

## 7. Interpretation

The data support a **decision-boundary / stable-chaos** picture:
- The smooth response is transient and then decays; prompt perturbations are weak.
- Token flips occur only when a near-tie is crossed; the threshold is large (r≈0.1–0.3 for this prompt) and direction-dependent.
- The depth map has a one-layer input expansion followed by marginal expansion; it is not sustained chaos.
- D2/RQA do not provide independent evidence of low-dimensional chaos.

The strongest new result is the **quantitative prediction of the first-flip threshold** from teacher-forced margins and responses, validated by free-run generation.
