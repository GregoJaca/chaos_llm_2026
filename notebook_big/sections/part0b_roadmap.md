# Part 0b. What is running, and what will appear here next

*(written at the start of the overnight campaign, 2026-11-20 ~23:30 UTC)*

The GB10 is scheduled solid for the next ~30 hours through five phases; each is resume-safe and skips completed steps. As each finishes, its section appears above with the full data.

| phase | content | expected |
|---|---|---|
| 1 (done) | DS-14B calibration + 1000-direction A1 (full attention — see correction) | Parts 1–3 above |
| 1-free (running) | free-run validation of all 1000 predictions at r=0.05/0.1/0.3 | Part 4 |
| 1b | window-actuation test, true-window-100 core (calibration, A1, 2-radius validation) | Part 5, and margin/threshold numbers to compare against Parts 1–3 |
| 2 | the window program: W-sweep on fixed tokens, perturbation-support scan, prompt-length scan, 1500-token cascade; state kicks at 10 positions; depth tangent spectrum (8 tokens × 48 blocks); MLP + full-block Jacobians ($\sigma_{\max}$, $\rho$, $\kappa$, principal angles); W=500 regime; 2 more prompts; 12 × 1000-token trajectories | Parts 6–9 |
| 3 | Qwen3.5-9B: the hybrid model (24 linear-attention Gated-DeltaNet blocks + 8 full-attention blocks, no sliding window). Core A1 saves all 32 block profiles, so the response can be compared *between the two block types* — a genuinely new measurement, not a replication | Part 10 |
| 4 | phi-4 (Phi3 family, 40 blocks, no window): core program | Part 11 |
| 5 | Qwen3-14B (same size class as the DS-14B, different training): core program | Part 12 |

**What will be checked first when the window test lands.** The single most consequential question of the night: with the window genuinely active, does the 14B reproduce the exact death-time law $t^* = p_{\rm last} + (W-P)$ (1.5B: exact to the token, 5 windows) and the per-block cascade $t^*_\ell = t^*_1 + (\ell-1)(W-1)$? Our version is stronger than the 1.5B test: each of the 1000 directions has its *own* last perturbed position $p_{\rm last}(u)$, so the law is tested as a per-direction prediction (1000 points per window), not as one number per window.

**On the two models without windows.** phi-4 and Qwen3-14B have no sliding-window mechanism at all; for them the window experiments are undefined and we document the null rather than pretend. Qwen3.5's "window" question is different and more interesting: its linear-attention blocks have a *fixed-size recurrent state* instead of a key-value cache, so information decays structurally rather than being masked — the response profile through depth will show whether perturbations survive differently through Gated-DeltaNet blocks than through full-attention blocks.
