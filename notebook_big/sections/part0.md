# Large models on the DGX Spark: definitions, data, and every figure explained

Companion of `notebook/notebook.md` (the DeepSeek-R1-Distill-Qwen-1.5B reference document, written on the Tesla T4 work). This one covers the large models on the new 128 GB machine: DeepSeek-R1-Distill-Qwen-14B first, then Qwen3.5-9B, phi-4, and Qwen3-14B. Symbols and procedures are defined there and re-defined here only when they differ.

---

# Part 0. What changed when the machine changed, and what it lets us do

## 0.1 One paragraph of motivation

The 1.5B findings — flips happen at near-ties of the top-1/top-2 logit gap, the window sets exact response edges, no growing instability — were established with 20 directions and 12 radii on one small model. The open questions are all about generality: does the same picture hold at 10× the parameters, where the logit gaps, the residual-stream scales, and the numerical floor all sit in different places? The new machine's single pool of 128 GB memory makes FP32 possible for 14B models (impossible on the 15 GB T4), and its weak-but-wide compute rewards batching, which makes 1000-direction ensembles cheap. So the plan is: same protocol, same prompt, many more directions, bigger models, and explicit measurement of everything that has to be re-calibrated (the usable radius window, the noise floor, the margin spectrum).

## 0.2 Hardware facts that shaped the method (measured, not asserted)

- **Unified memory, 128 GB.** GPU and CPU share one physical pool (~273 GB/s). A 14B model in FP32 is 59 GB of weights; colleagues' servers hold ~49 GB; ~10 GB remains for everything else. Consequence: the loader streams shards straight onto the GPU (`device_map`); a CPU-side copy followed by `.cuda()` would transiently need 118 GB and kill the machine.
- **FP32 compute is ~4.4 TFLOPS** (measured on a 4096³ matmul). Consequence: batch everything. One forward pass sweeps the weights once, so 16 perturbed rows cost about the same per-row as 1 row.
- **The fp32 fallback attention path materialises the full score matrix** $[B,H,L,L]$. At $L=3065$, batch 8, that alone is 12 GB. Consequence: long teacher-forced horizons are capped (~1500 tokens at small batch), and long trajectories are generated with the KV cache instead.

## 0.3 The numerical floor moved up 30×: where $r$ is meaningful

**Data.** Calibration run on the 14B: prompt `interstellar_travel` ($P=65$ tokens), 400 greedy tokens, window setting 100 (inactive — see the correction note; the numbers below are unaffected by this because they only depend on forward-pass rounding), 8 perturbation directions at 9 radii, FP32, teacher-forced on the model's own tokens. File: `results_big/ds14/calib/calib.npz`.

The quantity is $\mathrm{median}_{u,t\in[100,400)}\lVert\delta h_t^{\mathrm{final}}(u)\rVert / r$: the last-layer response per unit radius. The division by $r$ is the point: in a perfectly linear regime this ratio is a constant, the *gain*; where it inflates, the response is pinned at the forward-pass rounding floor and says nothing about the perturbation.

| $r$ | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.1 | 0.3 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| $\lVert\delta h\rVert$ | 2.73e-3 | 2.75e-3 | 2.82e-3 | 3.26e-3 | 5.6e-3 | 1.46e-2 | 4.77e-2 | 1.43e-1 | 5.64e-1 |
| $\lVert\delta h\rVert/r$ | 27.3 | 9.16 | 2.82 | 1.09 | 0.563 | **0.487** | **0.477** | **0.478** | 0.564 |

Reading it: below $r\approx0.01$ the response stops shrinking with $r$ — it is stuck at $\approx2.7\times10^{-3}$, which is the accumulated FP32 rounding of a 48-layer network evaluated twice (perturbed minus unperturbed). On the 1.5B (28 layers, on the T4) the same floor sat near $10^{-3}$ with a clean window of $r\in[10^{-3},10^{-1}]$; on the 14B the clean window is $r\in[0.03,0.3]$, shifted up by roughly 30×. Above $r\approx0.3$ the response begins to grow superlinearly (0.564 at $r=1$), the same saturation seen on the small model. Relevance: the usable perturbation range is a property of numerical precision and depth, not of the model's own scales; every big-model radius is chosen inside this window, and the campaign radius is **$r=0.1$**.

**An opinion, since these notes are supposed to argue:** the fp32 floor is not just an inconvenience — it is a *physical scale* of the system "network + arithmetic", and it grows with depth. That means "perturbation response" experiments on even deeper models in FP32 will eventually be floor-dominated unless radii grow too. At some depth, FP32 stops being a precision that can resolve small perturbations at all. We have not hit that wall at 48 layers; the 14B floor is 0.6% of the $r=0.1$ response, comfortably below signal.

## 0.4 The margin spectrum of the 14B

Same calibration run. $m_t=\ell_t^{(1)}-\ell_t^{(2)}$, the top-1 minus top-2 logit gap of the unperturbed greedy continuation — the only scalar that says how close a decision was.

Over the first 300 steps (the horizon used by the main ensembles): median $m=3.096$ nats, mean $4.996$, p10 $0.435$; 8/300 steps below 0.1 nats; 3 below 0.01. The five smallest:

| $t$ | 21 | 19 | 189 | 273 | 255 |
|---|---:|---|---|---|---|
| $m_t$ | **9.73e-5** | 1.14e-3 | 2.22e-3 | 1.92e-2 | 5.15e-2 |

(Earlier drafts of this Part quoted "median 1.68" — that number came from the smoke test, which used a *different prompt*; the correct value for this prompt is 3.096. The minimum and the smallest-five table reproduce exactly in the main A1 run.)

Relevance: razor-thin decisions — the substrate of the whole flip story — exist at 10× model size, are rarer than in the 1.5B per token (8 vs 6 steps below 0.1 nats out of 300, but the 1.5B's thinnest was 2.0e-2, so the 14B's thinnest is 207× thinner), and the first one sits very early ($t=21$).

## 0.5 Protocol details identical to the 1.5B, and what is new

Identical: FP32, one precision per model, TF32 off (it truncates mantissas and would fake margins); in-batch duplicate $r=0$ rows verified bit-identical; fixed-length greedy decoding; teacher forcing as the workhorse (perturbed passes run on the *baseline* token sequence, isolating the continuous response); free runs to validate flips; the signed threshold rule (below).

New: directions are built as **independent simplex blocks** — 50 blocks of 20 directions, each block on its own random set of 20 coordinates of the $65\times5120$ prompt embedding, blocks disjoint, so cross-block directions are exactly orthogonal and within-block pairs have inner product $-1/19$. This is the 1.5B's 20-direction geometry, replicated 50 times with independent subspaces. One radius per direction suffices because the response is linear in $r$ throughout the clean window — the 1.5B work established this and the 14B calibration confirms it (the table above), so the threshold ladder $r_c^-(t,u)=m_t r/\max(-\delta m_t(u),0)$ is computed from $\delta m/r$ and is radius-independent.
