# Results summary (DeepSeek-1.5B, FP32)

Window 100 unless stated. Prompt `interstellar_travel`, 300 generated tokens. All A1 runs use in-batch `r=0`; duplicate `r=0` rows are bit-identical.

## 1. A1 response: two regimes, linear in r

- Response at the last layer rises for ~80 generated tokens, then decays.
- At `r=0.1`, layers 14/21/27 peak at `t≈84–85`:
  - rise log-slope `+0.0055 … +0.0126` per token;
  - decay log-slope `−0.0099 … −0.0120` per token.
- Layers 1/2/3/7 decay from the start (peak at t=2).
- The rise/decay is a transient, not sustained exponential growth.

**Scaling collapse.** For `r∈[1e-3,0.1]`, fit `δh_t ∝ r^{α(t,ℓ)}`:
- layers 7/14/21/27: `α≈0.95–0.98`;
- dividing by `r` collapses the curves; `√r` does not;
- fitted `r^α` is only marginally better than `r`.
- Early layers (1–3) become sublinear (`α<0.8`) late; deep layers stay near 1.
- Numerical floor: below `r≈1e-4`, response is dominated by FP32/RMSNorm ill-conditioning and stops scaling. Usable range is `1e-3…0.1` (large `r` tested to 2).

**Layer dependence.** At fixed `r`, L27 has the largest response; L21/L14 next; L1–L3 decay to the numerical floor. The depth growth is concentrated in the first layer.

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
- `m_t = z_t^(1)-z_t^(2)`, `δm_t(u)`, `r_c(t,u)=m_t/|δm_t(u)/r|`, `T(r,u)=min{t:r≥r_c}`.
- Window 100: no flips at `r=0.1`; at `r=0.3`, 1/20 flips at t=16. Prediction: 1 at 16 (plus 1 at 123).
- Window 200: at `r=0.1`, 3/20 flip at 44/55; at `r=0.3`, 11–12/20 flip at 44. Prediction: 13/20.
- Window 500: same controlling near-tie at 44; same behavior as window 200.
- Large `r`: 3/20 (r=0.5), 8/20 (r=1), 16/20 (r=2) flip. Text remains coherent/non-degenerate at 300 tokens (distinct-token ratio ~0.26–0.30); no obvious garbage at this horizon.

This is direct evidence for decision-boundary crossing.

## 3. Current-state perturbation and R(t,s)

- Input-embedding perturbation at `s=16/50/100/150`, `r=1e-3`, 32 directions.
- Pairwise correlation of log response curves across directions: median **0.94–0.98**; 88–100% above 0.9. Directions are mostly alike; no strong groups.
- `R(t,s)` starts near 1 and decays; bumps occur at near-ties, at different lags for different `s`. Lag-shifted curves do not fully collapse.
- Middle-layer (layer 14) residual injection produces a response ~10× smaller than input-embedding injection and decays faster.
- Both current-state injections are transient.

## 4. Depth map / FSLE

- Depth separation `d(layer)/d(0)` jumps by ~200–300 at layer 1, then is flat/decaying. Local gain after layer 2 is ≈1.
- Depth FSLE is `lnρ/1` and constant in `r`: the growth is one layer, not sustained.
- Depth spectrum (tangent product, top 8): `λ_i≈0.08–0.18`; `h_depth≈0.6–1.4`.
- Local layer gain: layer 0 ≈10–20; layers 1–27 ≈1.1.
- The cumulative exponent starts at ~3 and decays to ~0.1: input-layer transient, then marginal expansion.

## 5. Non-normality: composition answer

Full block, 6 tokens × 5 layers:
- Layer 0: `σ_max≈95–195`, `ρ≈11–39`, `κ≈5–8`.
- Layers 7–21: `σ_max≈2.7–3.6`, `ρ≈1.3–2.1`, `κ≈1.7–2.5`.
- Layer 27: `σ_max≈6.0–9.1`, `ρ≈1.7–2.0`, `κ≈3.0–5.3`.
- Token 143, layer 14 is an outlier: `κ≈5.9` (token-specific non-normality).
- MLP alone (layer 27): `σ_max≈59–95`, `ρ≈17–24`, `κ≈3.3–3.9`.
- Principal angles between consecutive top singular subspaces: median **82–85°**.

Interpretation: the MLP amplifies strongly one step, but the full residual block is much tamer, and the expanding direction rotates almost completely each layer. The product contracts despite `σ_max>1`. This is the quantitative answer to “components amplify, composition may not.”

## 6. D2 / RQA: conclusive negative

- Last-layer D2 varies 3–11 across 12 prompts; vocabulary null ≈5.0.
- **Controls (one long trajectory):** raw D2 5.75, **time-shuffled 5.69**, phase-randomized surrogate 21.7, vocabulary 5.04.
- **Across all 12 prompts:** raw D2 ≈ time-shuffled D2 (diagonal scatter).
- TwoNN: last-layer ≈5–7; vocabulary ≈18.6.
- Delay embedding D2 increases with embedding dimension (5→17); no low-dimensional delay attractor.
- RQA DET 0.37–0.95, Theiler 20 barely changes it.

Conclusion: the low D2 is a **static geometric property** of the hidden-state cloud, not temporal dynamics. D2/RQA do not provide evidence of low-dimensional chaos at this N. The negative result is robust because of the shuffled control.

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

- One model, one prompt, windows 50–500, 300 tokens (margins 600).
- Depth spectrum top-8 only; `h_depth` lower bound.
- Full-block non-normality at 5 layers × 6 tokens; MLP same.
- State injection at input embedding and one middle layer, not all layers.
- D2 N=2–3k; the negative is about this range.
- 600-token A1 hit the full-vocab logits memory ceiling; not pursued.

## 9. Files

- Figures: `results/figures/INDEX.md`
- Measurements/formulas: `docs/MEASUREMENTS.md`
- D2/RQA method: `docs/D2_RQA_RESULTS.md`
- Decisions: `docs/DECISIONS.md`
- Data/code map: `docs/DATA_AND_CODE_MAP.md`
