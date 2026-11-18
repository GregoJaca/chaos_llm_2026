# Reviewer objections → new evidence

This maps the two reviews to the new measurements. It is not the final response letter.

## R1.1 / R2.2 — component Jacobians ≠ full evolution

**New evidence.**
- Full-block and MLP `σ_max`, `ρ`, `κ` (5 layers × 6 tokens): components amplify one step (`σ_max` up to 195, `κ` 2–8), but the full block damps and rotates.
- Principal angles between consecutive top amplification directions: median **83.6°** (IQR 78.7–87.0°), no layer trend.
- Misalignment budget `(σ−1)cos²θ` with measured σ/angles predicts the measured chained gains within ≲2×.
- Scale-normalized depth spectrum: every top-8 relative exponent is **negative** (max −0.034); the absolute depth FSLE ×206 becomes ×0.46 relative. The absolute growth is residual-stream scale drift, projected out by the final RMSNorm.
- Chained bounds: naive σ_max-chain ~2.5×10¹⁵, ρ-chain ~5.5×10⁶, measured aligned product ~27.6.

**Claim supported:** “Individual components can amplify, but the composition is contracting on the output-relevant sphere; the apparent depth expansion is radial drift.”

## R1.2 / R2.1 — finite-time growth / token threshold vs chaos

**New evidence.**
- Teacher-forced response is exactly linear in `r` for `r≤0.3`: `δh=r·G(t,ℓ)`, collapse CV 2–3%, α=1.00±0.02.
- Response is transient: shallow layers decay; deep layers rise/surge then decay. The regimes are **window-driven**: w100 jumps ×1.73 at t=35 and w200 ×1.51 at t=135, with wider windows as byte-identical controls. Free-run token streams diverge only after clipping.
- Signed threshold `r_c⁻=m·r/max(−δm,0)` predicts free-run flips 20/20 and 19/20 at r≤0.1, 17–18/20 at r=0.3, 15/20 at r=0.5; breakdown at r≥1.
- No sustained exponential growth on the token axis; no power-law/exponential distinction is decisive.

**Claim supported:** “The unpredictability is a decision-boundary crossing (near-tie + finite perturbation), not sustained exponential amplification.”

## R2.3 / R1.3 / R1.4 — diagnostics not specific; RR/D₂ robustness

**New evidence.**
- No D2 plateau at any N; local slopes decline monotonically; small-ε slopes exceed the Eckmann–Ruelle ceiling; Theiler 0→1000 changes nothing.
- Three of five long trajectories are near-exact repetition loops (P=53/74/69); their low D2 is repeated-text degeneration, reproduced by a trivial cycle+drift+jitter synthetic.
- Raw vs time-shuffled D2 is a tautology (shuffling preserves the point cloud); it cannot test dynamics.
- Non-loop `interstellar` is a clean negative; `launch` has a suggestive kNN predictability gap (+0.15 over its linear ceiling) but no continuation sharpness and one realization.
- TwoNN/MLE: locally thin (3–7) but globally spread (90% variance needs 161/223 PCs); finite-N bias.

**Claim supported:** “D2/RQA are not evidence of low-dimensional chaos here; the low-dimensional-looking values are static geometry/repetition.”

## R1.5 — scope

**New evidence.**
- Two prompts added beyond interstellar: childhood, cybersecurity. The threshold picture generalizes (no flips at r=0.01–0.1; 3–12/20 flips at r=0.3; near-tie locations differ).
- One model only. Scope claims must stay restricted.

## R1.6 / R2.4 — what is new / predictive value

**New contributions.**
1. **Signed threshold prediction:** `r_c⁻` predicts the first flip from one baseline forward + teacher-forced responses, without generation; 19–20/20 at r≤0.1.
2. **Window–margin spectrum:** attention window collapses the margin floor ~50× (0.167→0.0031) before trajectories diverge; the response has causal clip jumps at t=35/135.
3. **Scale-normalized depth contraction:** resolves “components amplify, composition may not” quantitatively; absolute depth growth is radial drift.
4. **Direction gain:** directions share one temporal envelope and differ by a scalar gain up to ×16; not rank-1.
5. **Practical reproducibility horizon:** `r=0` must be in-batch; FP32 floor below ~1e-4; bf16 batch effects invalidate Phi runs.

## R2.5 — chain-of-thought

**Not yet addressed.** We have not shown that any of this is specific to reasoning/CoT. The current data support ordinary autoregressive dynamics. Either add a reasoning-vs-non-reasoning comparison or drop CoT from the title/claims.

## What remains unsupported / do not claim

- Positive Lyapunov exponent of the full greedy map.
- Low-dimensional chaotic attractor from D2/RQA.
- Sustained exponential depth expansion.
- A universal `⟨T⟩(r)` scaling law.
- That large prompt perturbations produce garbage (they stay coherent at r=2, 300 tokens).
- Any Phi/bf16 result from the old data.
- That the teacher-forced response is a clean hard-windowed system (Qwen2 windows only last layers; long-lag tail persists).
