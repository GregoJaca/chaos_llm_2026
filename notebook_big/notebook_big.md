# Large models on the DGX Spark: definitions, data, and every figure explained

This document covers the experiments on the new 128 GB GB10 machine (models: DeepSeek-R1-Distill-Qwen-14B, Qwen3.5-9B, phi-4, Qwen3-14B). It is the companion of `notebook/notebook.md` (the 1.5B reference document); symbols and procedures are defined there and re-defined here only when they differ.

---

# Part 0. What is different on the big machine

## 0.1 Hardware and why it changes the method

The GB10 in the Spark has 128 GB of **unified memory** (GPU and CPU share one physical pool, ~273 GB/s) and weak fp32 compute (~4.4 TFLOPS measured on a 4096³ matmul). Two consequences drive the whole design:

- **Weights fit in FP32.** A 14B model needs 59 GB — impossible on the T4, comfortable here (colleagues' servers hold ~49 GB; ~10 GB is left for activations). All science runs are FP32, one precision per model, TF32 disabled (it truncates the mantissa and would fake margins).
- **Batches are nearly free.** Every forward pass sweeps the weights once, so 16 perturbed rows cost barely more than 1. The engine therefore (a) batches teacher-forced forwards (one pass over prompt+continuation = one weight sweep), and (b) never materialises the full logits: the margin response uses
  $$\delta m_t = \delta h_t^{\text{final}}\cdot\big(W_{U}[\text{top1}_t]-W_{U}[\text{top2}_t]\big),$$
  i.e. one dot product per position instead of a $[B,T,V]$ tensor.

## 0.2 Protocol (unchanged in substance)

Every rule from the 1.5B protocol carries over: in-batch $r=0$ with bit-identical duplicates, fixed-length greedy decoding, teacher forcing as the workhorse, free runs for flip validation, one precision per model. Directions are built as independent 20-direction simplex blocks on disjoint random coordinates of the prompt embedding (the 1.5B used one such block of 20; here 50 blocks give 1000 directions with the same per-block geometry).

## 0.3 The numerical floor moved up

**Figure 0.1 — fp32 response floor, DS-R1-14B, window 100, 400 tokens.**
Data: calibration run `results_big/ds14/calib` (FP32, greedy, prompt `interstellar_travel`, P=65). Each row of one direction at radius $r$; the quantity shown is the median over 8 directions and over $t\in[100,400)$ of the last-layer response $\|\delta h_t\|$ (norm of the hidden-state difference at the final norm), divided by $r$.

| $r$ | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.1 | 0.3 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| $\|\delta h\|/r$ | 27.3 | 9.2 | 2.8 | 1.09 | 0.563 | **0.487** | **0.477** | **0.478** | 0.564 |

Reading: below $r\approx 0.01$ the response stops shrinking with $r$ — it is pinned at $\|\delta h\|\approx 2.7\times10^{-3}$, the accumulated FP32 forward error of a 48-layer network in fp32. On the 1.5B (28 layers) the same floor sat at $\sim10^{-4}$–$10^{-3}$ and the clean window was $r\in[10^{-3},0.1]$; on the 14B the clean window is **$r\in[0.03,0.3]$** (response strictly proportional to $r$, spread 2%). Relevance: the usable perturbation range is set by numerical precision, not by the model's own scales, and it moves with depth. All big-model ensembles therefore run at $r=0.1$.

## 0.4 Margins on the 14B

**Data: same calibration run.** The margin $m_t$ is the top1−top2 logit gap of the unperturbed greedy continuation. Median $m=1.68$ (the 14B is much more confident than the 1.5B, whose margins were $\sim$0.1–0.5), but near-ties still exist:

| step $t$ | 21 | 19 | 189 | 273 | 339 |
|---|---|---|---|---|---|
| $m_t$ | **0.0001** | 0.0011 | 0.0022 | 0.0191 | 0.0347 |

Only 6/400 steps have $m<0.05$ (1.5%: comparable to the 1.5B). Relevance: razor-thin decisions — the substrate of the whole flip/threshold story — survive at 10× model size, but they are rarer, and the first one sits very early ($t=21$).

## 0.5 First signed thresholds

From the same 8 calibration directions, the signed flux-tube radius $r_*(u)=\min_t r_c^-(t,u)$ (the smallest radius at which direction $u$ would flip some token): median $0.14$, min $0.012$ (controlled by the $t=21$ tie), max $0.69$. At the campaign radius $r=0.1$ a substantial minority of directions sit below their threshold — the interesting regime where free runs decide which predictions come true.

*(figures and further parts are appended as data arrives)*
