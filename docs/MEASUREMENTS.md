# Measurements and results (theory-facing)

Status: 2026-11-17. DeepSeek-1.5B, FP32, window 100, prompt `interstellar_travel`, 300 generated tokens.
This file defines each measured quantity, where it is implemented, and what the first results show.
Use it to decide what to expand in the theory section.

## 1. A1: teacher-forced response

- **Definition:** `δh_t^(ℓ) = h_t^(ℓ)(E+Δ) - h_t^(ℓ)(E)`, with the baseline token sequence fixed; `Δ` is a prompt perturbation of norm `r` and direction `u`.
- **Implementation:** `src/chaos_llm/run_a1.py` (`main`); data in `results/a1_ds_w100/`.
- **Result:** the response grows for ~60–100 generated tokens, then decays. At `r=0.3`, the last-layer median response peaks near 1 and decays to ~0.1 at t=300. At `r=1e-3`, it stays ~1e-3. No sustained exponential growth on the token axis.
- **Numerical floor:** below `r≈1e-4`, the response is dominated by FP32 forward error / RMSNorm ill-conditioning (a floor of ~1e-5 at layer 1, ~1e-3 at layer 27). The linear regime is roughly `r∈[1e-3, 10^-1]`.

## 1b. Two-time response and current-state perturbation

- **Definition:** `R(t,s)=||∂h_t/∂h_s||₂`; the largest change at step `t` caused by a unit change at step `s`. `λ(t,s)=log R(t,s)/(t-s)`.
- **Implementation:** `src/chaos_llm/run_state.py`; injects a random unit vector at the input embedding of generated token `s` (not the residual stream), then measures `δh_t` for `t≥s`.
- **Interpretation:** this is the complementary experiment to the prompt perturbation. It asks whether the *current* state is sensitive, without the persistent prompt perturbation in the KV cache. It is not a replacement for the prompt experiment; it is a different initial condition.

## 2. Margins, thresholds, predicted first flip

- **Margin:** `m_t = z_t^(1) - z_t^(2)`.
- **Margin response:** `δm_t(u) = δz_t^(1) - δz_t^(2)`, using baseline top-1/top-2 indices.
- **Signed local flip threshold:** `r_c⁻(t,u) = m_t·r / max(−δm_t(u),0)`. A flip requires `δm_t ≤ −m_t`; the symmetric `|δm|` version is adversarial and over-predicts.
- **Predicted first flip:** `T(r,u) = min{t: r ≥ r_c⁻(t,u)}`.
- **Flux-tube radius:** `r_*(u) = min_t r_c⁻(t,u)`.
- **Implementation:** `run_a1.py` stores `margins`, `dm`, `r_c`, `pred_T` (symmetric); the signed analysis is in `results/subagent_threshold/` (`s1_signed.py`, figures).
- **Result:** margins have near-ties at t=123 (0.0202) and t=143 (0.0204) for window 100, and t=44 (0.0031)/t=55 (0.0089) for window 200. Window 200: 13–14/20 directions controlled by t=44; 18/20 directions are margin-destroying there. Window 100: t=123 controls 12/20; only 7/20 are margin-destroying. Signed prediction matches free-run 20/20 and 19/20 at `r≤0.1`, 17–18/20 at `r=0.3`, 15/20 at `r=0.5`, 9–12/20 at `r=1–2`. Small-`r` thresholds do not extrapolate: gain grows ~1.5× by `r=2`.

## 3. Time-axis FSLE

- **Definition:** `τ_ρ(δ)=inf{t:d(t)≥ρδ}`, `λ(δ)=lnρ/⟨τ_ρ(δ)⟩`.
- **Implementation:** `scripts/fsle_a1.py`; data in `results/fsle_time.json`.
- **Result:** for `ρ=2`, median `λ≈0.043` over the initial transient; for `ρ=4`, many runs are censored. This is a transient growth rate, not sustained exponential growth. The response decays after ~100 tokens.

**Depth-axis FSLE (existing single-token data).** `d(layer)/d(0)` jumps by ~200–300 at layer 1, then is flat/decaying. The FSLE is `lnρ/1` and constant in `r`, because the entire growth is one layer. It is not evidence of sustained depth chaos. Implementation: `scripts/fsle_depth.py`; figure `fsle_depth.png`.

## 4. Depth map and depth spectrum

- **Depth map:** for a fixed token, `h_{ℓ+1}=F_ℓ(h_ℓ)`, `J_ℓ=∂F_ℓ/∂h`.
- **Product and spectrum:** `M=J_{L-1}…J_0`, `λ_i=(1/L)log σ_i(M)`; top-`k` via QR/Benettin.
- **Expansion entropy:** `h_depth=Σ_{λ_i>0} λ_i`.
- **Implementation:** `src/chaos_llm/run_depth.py` (`main`); data in `results/depth_w100/`.
- **Result:** all top-8 exponents are positive; final `λ_i≈0.08–0.18`, `h_depth≈0.6–1.4`. Layer 0 gain is 10–20; subsequent layers ≈1.1. The cumulative exponent starts at ~3 and decays to ~0.1: **the depth expansion is a transient input-layer effect followed by marginal expansion**, not a sustained large positive exponent.
- **Caveat:** only the top 8 of 1536 directions are sampled; `h_depth` is a lower bound.

## 5. Non-normality

- **Definitions:** `σ_max=||J||₂` (largest singular value; worst one-step amplification), `ρ(J)=max_i|λ_i|` (spectral radius; asymptotic growth for repeated application of the same `J`), `κ=σ_max/ρ`.
- **Implementation:** `src/chaos_llm/run_nonnorm.py` (`main`); data in `results/nonnorm_w100/`.
- **Result:** full block, layer 0: `σ_max≈95–194`, `ρ≈11–39`, `κ≈5–10`; layer 14: `σ_max≈3.1–3.3`, `ρ≈1.34–1.62`, `κ≈1.9–2.5`; layer 27: `σ_max≈6.7–9.1`, `ρ≈1.70–1.85`, `κ≈3.6–5.3`. MLP alone: layer 27 `σ_max≈59–95`, `ρ≈17–24`, `κ≈3.3–3.9`.
- **Principal angles:** median ≈79–84° between consecutive top singular subspaces. The expanding direction rotates almost completely each layer.
- **Interpretation:** a component can amplify one step (`σ_max>1`) while the product contracts. The MLP amplifies much more than the full residual block; the rotation is the mechanism. The product singular values and principal angles, not per-component norms, answer the reviewer objection.

## 6. Correlation dimension and RQA

- **Definitions:** `C(ε)=2/(N(N-1))Σ_{i<j}Θ(ε-||x_i-x_j||)`, local slope `d log C/d log ε`; RQA on the thresholded recurrence matrix.
- **Implementation:** `src/chaos_llm/analysis/dimension.py`; runner `scripts/run_d2.py`; data in `results/d2/`.
- **Result (negative):**
  - No stable local-slope plateau for first/last layer trajectories at N=2000–3000.
  - First-layer D2 ≈13–27, far above the Eckmann–Ruelle bound `2 log10 N≈6.6`; not reliable.
  - Last-layer D2 ≈3–10 and decreases with N for some prompts, but not consistently; it is not clearly below the vocabulary null (≈5.35).
  - Theiler window (0/10/50) does not produce a plateau.
  - RQA DET ≈0.37–0.95, Lmax up to ~1775; Theiler 20 barely changes it. This reflects strong temporal correlation, not necessarily deterministic chaos.
- **Conclusion:** D2/RQA are **not** reliable evidence for low-dimensional chaos in this data. The honest claim is “no robust low-dimensional scaling at the accessible N.”

## 7. Extra quantities that are nearly free

- **Hidden-state norm by layer:** from the same A1 forward (`norm_h0`); shows radial scaling.
- **Top1/top2 token identities at near-ties:** from `run_a1.py`; useful for “forking token” examples.
- **Cosine response:** `cos` array in `a1.npz`; separates angular from radial response.
- **Attention entropy:** not computed yet; cheap if we add hooks.
- **Per-layer response gain:** `depth_local_gain`; already available.

## 8. Promising directions for theory (only if supported by more data)

- **Vertical expansion vs horizontal contraction:** depth map expands (λ≈0.1) while the token-axis response decays. A “stretch in depth, fold in time” picture is plausible.
- **Transient non-normal amplification:** layer 0 gain 10–20, then marginal; `κ>1`. This is a concrete mechanism for “components amplify, composition may not.”
- **Decision-boundary geometry:** `r_c(t,u)` distribution and `r_*`; the first flip is a near-tie crossing, not a smooth instability.
- **Numerical floor as a physical scale:** below `r≈1e-4`, FP32 noise dominates; this is a practical reproducibility horizon.

## 9. Second-pass results

- **A1 universal curve:** for `r≤0.3`, `δh=r·G(t,ℓ)` exactly; collapse CV 2.0–3.3%; fitted `α=1.00±0.02`, no time/layer dependence. Shape rise → pre-flush surge (peak t≈17 then ×1.9–2.5 at t≈70–90) → tail; shallow layers go bitwise zero at t*=92/191/290 (window−1). w100 jumps ×1.73 at t=35, w200 ×1.51 at t=135, with wider windows as controls. Depth gain `A(ℓ)~ℓ^0.64`. Directions share shape and differ by scalar gain ×1.13–18.97.
- **Margins vs window:** min margin 0.167 (w50), 0.0201 (w100), 0.0031 (w200/500); count(m<0.05) 0,2,8,11,12. Window changes the margin spectrum.
- **Signed threshold:** `r_c⁻=m·r/max(−δm,0)`; 19–20/20 exact at r≤0.1, 17–18/20 at 0.3, 15/20 at 0.5, 9–12/20 at r=1–2. Gain grows 1.5× by r=2; no extrapolation.
- **State directions:** PC1 explains 94–97% of log-curve variance; median correlation 0.94–0.98; no clusters/special directions. `R(t,s)` is not a function of lag (collapse fails ×1.6). Middle-layer injection 9–36× weaker and plateaus at ≈2.9e-4. State response linear in r over 1e-3…3e-2.
- **Depth is scale drift:** residual norm ×30 in block 0, ×1.05–1.18/layer. Normalized, every block contracts (max relative exponent −0.034); FSLE ×206 absolute → ×0.46 relative; `h_depth` is gauge drift. A small +0.02 nats/layer aligned drift remains; late MLP σ rise (18→71) is absorbed by scale growth.
- **Non-normality:** full block layer 0 `σ_max≈95–195`, `ρ≈11–39`, `κ≈5–8`; layers 7–21 `σ≈3`; layer 27 `σ≈6–9`, `ρ≈1.7–2.0`. Median principal angle 83.6°; misalignment budget predicts chained gains within ≲2×. Token 143/L14 is a non-propagating κ=5.9 outlier.
- **D2 controls:** raw D2 ≈ time-shuffled D2 across all 12 prompts (5.75 vs 5.69); phase surrogate 21.7; vocab 5.04. Low D2 is static geometry, not temporal dynamics. Delay D2 rises with embedding dimension.
- **Large r:** flip fraction 3/20 (r=0.5), 8/20 (r=1), 16/20 (r=2); text remains coherent at 300 tokens.
- **Second prompt (childhood):** no flips at r=0.01; 5/20 at r=0.1; 13/20 at r=0.3, at near-ties 125/173. Generalizes.
