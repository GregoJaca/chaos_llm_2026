# Work that needs more than the current 24 h of GPU

The current batch should cover the core claims. If more GPU time is available (the user offered
up to ~8 more days of a T4), these are the highest-value additions, in order.

## 1. Stable chaos: statistical power

- 12–20 prompts × 20 directions × 12 radii × 300 tokens of free runs (currently 4–6 prompts).
  Cost: ~40 min/prompt at 12 radii and 20 dirs with batch 6. Purpose: turn the flip-time
  discreteness and the survival-curve match into proper population statistics (per-prompt
  p-values, mixed-effects model of `T` on `r`, prompt and direction).
- A second model (Qwen2.5-1.5B or Phi-3.5-mini in FP32/bf16-consistent) with the same protocol.
  This is the single strongest generality claim; the current paper covers one model.

## 2. Window flush: dose–response

- Prompt-length scan: P ≈ 20, 40, 65, 100, 150, 250 tokens × W ∈ {50…600}. The law predicts
  `t*_1 = p_last + (W−P)`; varying P tests the `−P` term directly.
- Layer scan to ℓ = 12 with 1000-token generations, to see whether the cascade continues or stops
  when the response is carried by generated positions.
- Perturbation-support scan: choose directions confined to early / middle / late prompt positions
  and verify `t*_1` tracks `p_last` exactly. This turns the law into a controlled experiment.

## 3. Window regimes: the real question

- A 2D grid: W ∈ {50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 500, 1000, 4096} × P ∈ {20, 40,
  65, 100, 150} × 3 radii × 20 directions, 500 tokens. Purpose: map the (W, P) plane, with W<P,
  W≈P and W>P regions, and measure λ in the pre-clip window only.
- State perturbations on the same grid (positions 8–200), 24 directions.
- Repeat with sampling (temperature 0.7, fixed seed) to see whether the phenomenon survives
  stochastic decoding; the current work is greedy only.

## 4. Lyapunov-like analyses that need long horizons

- 2000–4000-token generations with fixed prompts, measuring λ(t) to t = 4000. The current horizon
  (300–500) cannot distinguish a slowly decaying rate from zero.
- Benettin-style renormalisation is not meaningful in the linear regime (rescaling the perturbation
  is equivalent to one run), but a *long-horizon* finite-time exponent can still be estimated:
  λ(T) = (1/T) log(a(T)/a(1)) for T up to 4000, with block bootstrap.
- Compare with matched synthetic systems (stable-chaos map, Lorenz, Hénon) at the same N to show
  what a real positive rate looks like in these plots.

## 5. Depth and architecture ablations

- Freeze or zero individual components (attention output, MLP output, a single head) and repeat the
  A1 and layer-probe measurements. The user's core interest is "how each component affects the
  dynamics"; the current data only has full blocks.
- Per-head attention knockout at the layers where the flush cascade happens.

## 6. Data hygiene

- Regenerate the 8 quasi-loop trajectories with anti-repetition *evaluation* only (do not change
  the main protocol): the loops are a property of greedy decoding, so record them as such.
- Save the token sequences and all hidden states for every perturbation run (currently only
  summaries), so future analyses do not need the GPU.
