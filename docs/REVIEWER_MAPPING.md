# Reviewer objections to evidence (current)

This maps the reviews to the measurements we now have, with the corrections from the later rounds. It is not the response letter.

## R1.1 / R2.2 — component Jacobians are not the full evolution

**What we can claim now.**
- Within one block, a component can amplify: `σ_max` up to 195 at layer 0, 2.7–3.6 mid, 6–9 at layer 27; `κ=σ/ρ` 2–8; MLP alone 58–93 at layer 27.
- The block composition is strongly non-normal: the top amplification direction rotates by a median of **83.6°** between consecutive layers; the misalignment budget `(σ−1)cos²θ` predicts the measured chained gains within ≲2×; the measured product is ~13 orders below the naive `σ_max` chain and ~5 below the `ρ` chain.
- **A fixed token position's** depth product contracts relative to the state scale (max normalised exponent −0.034; FSLE ×206 absolute → ×0.46 relative).
- **But the full response does not contract:** the perturbation is nearly perpendicular to the state (86–87°), and its relative size grows ×10 through the depth (shrinking only in the final band, ×0.84). The state's length grows ×452; the output renormalises length away, so the logits see only the perpendicular part, and the measured margin change tracks it linearly (`|δm| ≈ 4 × perpendicular/|h|`).

**Claim supported.** "Components amplify one step, but the composition rotates and the same-position product contracts; the observable effect is a small, nearly perpendicular direction change whose size grows across depth. Do not confuse the state's length growth with amplification."

**Do not claim:** "the depth expansion is only radial scale drift" (wrong), or "the full map contracts" (wrong).

## R1.2 / R2.1 — finite-time growth vs chaos; token threshold

**What we can claim now.**
- Response is **exactly linear in the perturbation size** for `r ≤ 0.3` (α = 1.00 ± 0.02, collapse CV 2–3%), for prompt perturbations and for single-token kicks (slope 0.99–1.00).
- The rise/decay shape is **caused by the sliding window**, proven with identical inputs and five windows: each curve is bit-identical to the no-clipping control until `t = window − prompt_len`, then departs (35/85/135/235 for a 65-token prompt; 38/138 for a 62-token prompt). Generated text splits 4–6 tokens later.
- A flip is a **near-tie crossing**. The signed local threshold (measured at the radius being predicted) predicts 318/320 direction-runs (99.4%): 57/59 exact words, 0 false alarms, 0 misses, across 3 prompts and 19 experiments.
- **One teacher-forced pass is an exact simulator up to the first flip**: 29/29 flips exact (step and token), 0 false positives in 111 no-flip cases.
- Two flip kinds: 17/27 immediate at the response peak; 10/27 delayed at a later close call, after the response has faded below 30% of its peak. A single-token kick at s=16 flips the word at t=123 — **107 tokens later** — confirmed autoregressively (4/4, same directions and step).
- Extrapolation of a small-radius threshold is valid only up to `r ≈ 0.3`; above that the gain is superlinear (×1.55 by r=2) and tie-dependent.

**Claim supported.** "Unpredictability is a decision-boundary crossing (near-tie plus finite perturbation) with a measurable latency, not sustained exponential amplification."

## R2.3 / R1.3 / R1.4 — diagnostics are not specific

**What we can claim now.**
- No D2 scaling region exists on non-loop trajectories: the whole rise of `C(ε)` spans 0.16–0.19 decades; the interval below the finite-N ceiling and above the depletion floor is 0.00–0.04 decades wide. Centering, N, ε normalisation, z-scoring and cosine do not change this.
- The estimator is sound: at the same N it recovers Hénon (D2 = 1.157, flat over 2.57 decades) and Lorenz (1.984, r² = 1.000).
- The "good-looking" shapes seen earlier are reproduced by five artifacts: repetition loops (D2 = 1.11, r² = 0.999), first-layer exact duplicates, m=2 delay embedding (fill dimension; white noise gives the same), first-PC saturation (IAAFT nulls match), and `cdist` self-pairs / linear ε grids.
- Raw vs time-shuffled D2 is a tautology (point-cloud statistic).
- Theiler windows do nothing except remove loop-copy pairs.
- Non-loop predictability: one old trajectory showed a suggestive gap over IAAFT; it is being tested on 16 fresh 1000-token non-loop trajectories.

**Claim supported.** "D2/RQA are descriptive at best; on this data they are artifacts."

## R1.5 — scope

- DeepSeek-R1-Distill-Qwen-1.5B only; three prompts used for threshold work (propulsion, childhood, cybersecurity); windows 50–500; 300 generated tokens (450–500 in one run); 16 fresh 1000-token trajectories for dynamics.
- Claim scope must stay at "this model, this prompt set, this horizon".

## R1.6 / R2.4 — what is new / predictive value

1. **Signed threshold + measurement at the target radius**: 99.4% accurate, no false alarms; the sign-blind variant is 89.1% with 21 false alarms.
2. **The prediction is a survival forecast**: 0/75 flips re-synchronise, mean agreement 2.7% afterwards, surviving fraction = first-flip step / 300; no-flip runs are bit-identical.
3. **Window–margin law**: departure step = window − prompt length, verified on two prompts with identical inputs.
4. **Teacher forcing is exact up to the first flip** (29/29), so the response work needs no generation.
5. **Latency**: perturbations can stay latent for ~100 tokens and act at a later close call (validated autoregressively).
6. **Directions are not equivalent**: gain spread ×7–12 (stable across continuations), shape differences (split-half r=0.999), and opposite signs at the same close call; flips are stratified by direction.
7. **Practical reproducibility horizon**: in-batch `r=0`, FP32 floor at `r≈3e-4`, linear window `3e-3…0.3`, superlinear above 0.5.

## R2.5 — chain-of-thought

Not addressed; out of scope per the latest instruction. Do not claim anything about reasoning-specific dynamics.

## What remains unsupported / do not claim

- A positive Lyapunov exponent of the full greedy map.
- A low-dimensional chaotic attractor (D2/RQA).
- Sustained exponential depth expansion.
- A universal `⟨T⟩(r)` scaling law.
- That large perturbations produce garbage (they stay coherent at r=2, 300 tokens).
- That "all blocks contract" describes the response.
- Any Phi/bf16 result from the old data.
- That the teacher-forced response follows a hard window at all layers (Qwen2 windows only the last layers; free runs are the ground truth).
- CoT-specific dynamics.
