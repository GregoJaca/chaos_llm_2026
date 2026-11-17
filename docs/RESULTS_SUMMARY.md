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

## 2. Margins and first-flip threshold

- Smallest margins: t=123 (0.0202), t=143 (0.0204), t=116 (0.069), t=2 (0.0696), t=16 (0.120).
- `r_c(t,u)=m_t/|δm_t(u)/r|`; `r_*(u)=min_t r_c`.
- **No token flips at r=0.03 or r=0.1** for any of 20 directions, in 300 tokens.
- At `r=0.3`, 1/20 directions flips at t=16. Teacher forcing predicts a flip at t=16 (and an additional predicted flip at t=123 that does not occur in the free run).
- This is direct evidence for decision-boundary crossing, not smooth exponential instability.

## 3. FSLE

- Time axis: `λ≈0.043` for `ρ=2` over the initial transient; `ρ=4` is mostly censored. It is a transient growth rate.
- Depth axis (existing single-token data): `d(layer)/d(0)` jumps by ~200–300 at layer 1, then is flat/decaying. FSLE is trivially `lnρ/1` and flat in `r`. The depth “expansion” is a one-layer input transient.

## 4. Depth spectrum and expansion entropy

- Top-8 depth exponents are all positive, final `λ_i≈0.08–0.18`; `h_depth≈0.6–1.4` (lower bound).
- Local layer gain: layer 0 ≈10–20; layers 1–27 ≈1.1.
- The cumulative exponent starts at ~3 and decays to ~0.1. So the depth map is **transiently expanding at the input, then marginally expanding**.
- This is a concrete, theory-relevant object: vertical expansion without sustained large λ.

## 5. Non-normality

- Example: layer 14, token 50: `σ_max≈3.24`, `ρ≈1.36`, `κ≈2.4`.
- Full layer-wise/non-normality plots are being generated. The point is: per-component `σ_max>1` can coexist with a contracting product; `κ` and the product singular values quantify it.

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
