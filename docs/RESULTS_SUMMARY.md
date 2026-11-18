# Results summary (DeepSeek-1.5B, FP32)

Window 100 unless stated. Prompt `interstellar_travel`, 300 generated tokens. All A1 runs use in-batch `r=0`; duplicate `r=0` rows are bit-identical.

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
  The symmetric `|δm|` version is the adversarial rule and over-predicts flips; it was replaced.
- Window 200/500: the shared t=44 near-tie (m=0.0031) controls 13–14/20 directions; 18/20 directions are margin-destroying there. Window 100: t=123 (m=0.020) controls 12/20; only 7/20 are margin-destroying.
- Prediction scorecard (signed): exact flip-step agreement 20/20 and 19/20 for `r≤0.1`; 17–18/20 at `r=0.3`; 15/20 at `r=0.5`; 9–12/20 for `r=1–2`.
- All `r=0.3` misses lie in a soft band `r_*⁻∈[0.26,0.40]`.
- Small-`r` thresholds do **not** extrapolate to large `r`: median gain grows ~1.5× by `r=2`; the linear rule over-predicts (15 predicted vs 8 observed at `r=1`).
- Window 50: min margin 0.167; no flips at `r=0.1/0.3`; only 1/20 flips at `r=0.5`.
- Large `r`: 3/20 (r=0.5), 8/20 (r=1), 16/20 (r=2) flip; text remains coherent/non-degenerate at 300 tokens (distinct-token ratio ~0.26–0.30).

This is direct evidence for decision-boundary crossing.

## 3. Current-state perturbation and R(t,s)

- Input-embedding injection at `s=16/50/100/150`, 32 directions, `r=1e-3`; state r-scan at 3e-3/1e-2/3e-2.
- **Directions are mostly equivalent:** PC1 explains 94–97% of log-curve variance; median pairwise r=0.94–0.98; residual direction effects are only ×1.15–1.23; no clusters, no special directions, and no direction keeps its deviation across different `s`.
- **`R(t,s)` is not a function of lag:** normalization by `dh(s)`, early peak, or `dh(s+5)` leaves ×1.56–1.63 RMS spread (max ×2.5). Lag/s rescaling and first-bump alignment fail. The field is only approximately separable (`log dh≈a(s)+b(t)`, ×1.17 late).
- Bumps are **not** pinned to margin near-ties (enrichment ≈1; lagged correlation with −log margin only 0.07–0.44, s-dependent phase).
- Gain at injection varies ×13 across s (10–124); decay is two-timescale (half-life 2–8 tokens, then e-folding 30–125). `s=16` is special because it is a near-tie injection, not a regime change.
- **Response is linear in r** over 1e-3…3e-2 at injection (slope ≈1); s=16 gain ~10× larger than s=50/100.
- Layer-14 residual injection is **9–36× weaker at injection**; it does not decay faster after ~5 tokens — its tail is flatter and plateaus at a persistent ≈2.9e-4 footprint (real, not FP32 noise).
## 4. Depth map / FSLE: the “expansion” is scale drift

- Absolute depth separation `d(layer)/d(0)` jumps ~200–300 at layer 1, then flat; FSLE `lnρ/1`, r-independent.
- The residual-stream norm grows ×30 in block 0 and ×1.05–1.18/layer thereafter (×441–591 total).
- **Normalized by the local state scale, every block contracts:** block 0 ×0.48–0.72; later ×0.89–0.95. The maximum scale-normalized exponent over 8 tokens × 28 depths × 8 directions is **−0.034 nats/layer**.
- Depth FSLE: ×206 absolute growth becomes **×0.46 relative**.
- The top-8 absolute exponents (λ≈0.08–0.18, `h_depth≈0.6–1.4`) are the entropy of radial drift; the final RMSNorm projects it out before logits. Report both absolute and relative.
- What survives: a small sustained aligned drift **+0.02 nats/layer**, with a late-depth component rise (MLP σ_max 18→71 at block 27) absorbed by faster scale growth.

## 5. Non-normality: composition answer

Full block, 6 tokens × 5 layers:
- Layer 0: `σ_max≈95–195`, `ρ≈11–39`, `κ≈5–8`.
- Layers 7–21: `σ_max≈2.7–3.6`, `ρ≈1.3–2.1`, `κ≈1.7–2.5`.
- Layer 27: `σ_max≈6.0–9.1`, `ρ≈1.7–2.0`, `κ≈3.0–5.3`.
- Token 143, layer 14: `σ_max=9.2`, `ρ=1.57`, `κ=5.9` — a genuine non-normal transient spike at a 0.020 near-tie; does not propagate (h_depth mid-pack). Flagged anecdote.
- MLP alone (layer 27): `σ_max≈58–93`, `ρ≈17–24`, `κ≈1.8–3.9`.
- Principal angles: median **83.6°**, IQR 78.7–87.0°, no layer trend; `|cos|` 5× random but `cos²≈0.012`.
- Misalignment budget `(σ−1)cos²θ` with measured σ and angles predicts measured chained gains within ≲2×. The measured product is ~13 orders below the naive σ_max-chain and ~5 below the ρ-chain.
- Interpretation: components amplify one step, but the full block tames them and the amplification direction rotates; the composition contracts. This is the quantitative answer to “components amplify, composition may not.”

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
