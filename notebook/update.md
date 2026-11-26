# Update 2 — all old figures, rebuilt from raw data

**Date:** 26 November 2026  
**What this is:** every figure you said you liked, every figure you asked to improve, and the new analyses, all recomputed from the raw `.npz` files. The old `unified.md` figures are also replaced in place (with a warning banner) so `unified.pdf` now shows measured plots.

**How to rebuild everything:** `python analysis_update/run_all.py`. It regenerates every figure in `update_figures/` and writes the number files in `analysis_update/`. There is no hard-coded numerical data in the plotting scripts; every array is read from `results/` or `results_big_small/`.

**Status of the long campaign (2026-11-26 23:20 CET, still running).** Two 36–48 h queues are generating the long-horizon looping data that this version only partially has (the closed-vs-replay figures currently use 600-step orbits except DeepSeek-1.5B, which already has 3000 steps):

- **x11**: 8 small models, 3000 steps, `T=1` loop+replay, `T=2`, CRN/independent sampling, 1000-token teacher-forced, `T=0.5`.
- **Spark**: Qwen3-14B, DeepSeek-14B, Qwen3.5-9B, Phi-4 at fp16, 1600 steps, same rounds with 800-token teacher-forced.
- After those finish, a second waiter queue runs **32-direction** loops (percentile bands) and **multi-prompt** replication (the single most important missing measurement — everything here is one prompt).

`analysis_update/sync_from_cluster.ps1` pulls the new data and `pack_loop_loglog_models.py` automatically switches to the longest orbit available, so the figures in §F sharpen on every re-run without editing code. The exact commands and current state are in `docs/HANDOFF.md`.

**The precision tiers (read this first).** The old small-model A1 files are not all the same precision, and the difference is large:

| tier | models | what is valid |
|---|---|---|
| **fp32** | DeepSeek-1.5B, Qwen2.5-3B (new run), Qwen3.5-2B (new run), DeepSeek-14B, Qwen3-14B, Phi-4 | absolute response, linearity, $r_*$, angles |
| **fp16/bf16** | Qwen3-4B, Phi-4-mini, Qwen3.5-9B | discrete flips and margins only; absolute $\|\delta h\|$ is floor-dominated |

The proof is Fig. G1: for the *same* model (Qwen2.5-3B), the fp16 run has a flat response of 0.65 at every $r$, while the fp32 run scales from $7\times10^{-4}$ to 0.23. The old `meta.json` says `float32` for the fp16 runs; it is wrong. This single fact invalidates the old cross-model absolute comparisons, and it is the reason many old figures looked “too perfect”.

**What “synthetic” meant.** The experiments were real. The *arrays passed to the plotting functions* in `scripts/build_unified_figures.py` were generated in code: `np.random.lognormal(...)` for the ECDFs, `np.linspace(...)` for the curves, `exp(-layers/6)` for the depth spectra, and fixed tables for the flip fractions. That is not sampling noise; it is data invented at plot time. `scripts/softloop_figs*.py` (the `new.md` figures) do **not** do this: they load the `.npz` files. The audit is in §1.

---

## 1. Provenance audit

### 1.1 `unified.md` (the figures were fabricated at plot time)

| old figure | what the script did | status |
|---|---|---|
| Fig. 1 flux-tube ECDF | `np.random.lognormal` per model | synthetic |
| Fig. 1b $r_*$ decomposition | hard-coded tables | synthetic |
| Fig. 2 $r_*$ vs gain | `np.random.normal` | synthetic |
| Fig. 3 time dynamics | `np.linspace`, fixed temperature grid, analytic decay | partly synthetic |
| Fig. 4 window kinematics | synthetic $\tau$, synthetic attention curve, fixed survival table | synthetic |
| Fig. 5 depth fits | analytic `0.02+1.4 exp(...)` | synthetic |
| Fig. 6 Qwen/Phi depth | `exp(-layers/6)(1+0.15 cos)` | synthetic |
| Fig. 7a perpendicular angles | hard-coded medians/std | synthetic |
| Fig. 7b rotation | `np.random.normal(83.6, 5.5)` | synthetic |
| Fig. 7c radial expansion | `np.cumprod(1.10)` | synthetic |
| Fig. 8 architecture profiles | block types by index | synthetic |
| Fig. 9 linearity | `np.random.normal`, fixed superposition table | partly synthetic |
| Fig. 10 flip fractions | fixed table | numbers real, plot hard-coded |
| Fig. 11 transport | `np.linspace`, analytic heatmap | synthetic |

I replaced the old image files with measured versions under the same names (backup in `unified_figures/_synthetic_backup/`). `unified.md` now begins with a banner saying its text is the old draft and its figures are the new measured ones.

### 1.2 `new.md` (the figures are real)

`softloop_figs.py`, `softloop_figs2.py`, `softloop_master.py`, `make_sampling_and_response_figs.py` and `make_entropy_fig.py` load the `.npz` files (`np.load`); they contain no `np.random`. Some panels overlay analytic fit curves (e.g. the gate identity), but those are fits, not invented data. The one substantive correction is that `ent_ref` is computed at $T=1$ (`run_softloop.py:245`), so “entropy vs $T$” is not a temperature effect (§F6).

### 1.3 Old figures replaced in this document

| old figure | new figure(s) here |
|---|---|
| fig01 flux tube | Fig. A1, A2, A3, A4 |
| fig01b decomposition | Fig. A2 |
| fig01c fragility | Fig. A1 |
| fig02 r* vs gain | Fig. A3 |
| fig03/3b/3c/3d time | Fig. D1, D2, D3 |
| fig04/4b window | Fig. E1, E2, E3 |
| fig05/5b/6 depth | Fig. B1–B5 |
| fig07a/b/c/e angles | Fig. C1–C5 |
| fig08 architecture | Fig. B1, B2 |
| fig09 shape/linearity | Fig. G1–G4 |
| fig10 scorecard | Fig. H1 |
| fig11 transport/superposition | Fig. I1, I2 |
| `new.md` panels | Fig. F1–F8 |

---

## 2. Definitions, including the readout cross-covariance

| symbol | definition | what it answers |
|---|---|---|
| $h_\ell(t)\in\mathbb{R}^d$ | residual-stream state at layer $\ell$, token $t$ | the state |
| $r$ | Euclidean norm of the perturbation added to the input embeddings (whole $P\times d$ tensor, or one $d$-vector) | perturbation size |
| $\delta h_\ell(t)$ | $h'_\ell(t)-h_\ell(t)$ | response |
| teacher-forced | the perturbed run is fed the *unperturbed* tokens | isolates the network |
| free run | greedy autoregressive generation | what the user sees |
| replay | teacher-forced along the recorded soft-token orbit | network inside the loop |
| $m_t$ | $z_t^{(1)}-z_t^{(2)}$ | top-two logit gap |
| $\delta m_t$ | change of $m_t$ under perturbation | push on the decision |
| $r_*(u)$ | $\min_t m_t\,r/\max(-\delta m_t(u),0)$ | smallest radius that flips a token (linear estimate) |
| $G(u)$ | $\|\delta h_L(T)\|/r$ | bulk gain |
| gate | a step with small $m_t/T$ | where the readout acts |
| $\mathrm{cum}_\ell$ | $\log\sigma_1(J_\ell\cdots J_1)$ on the tracked subspace | log product gain |
| $g_\ell$ | mean $\|J_\ell v_i\|$ on the tracked basis | per-block gain |
| $\lambda_L$ | $\mathrm{cum}_L/L$ | mean per-block exponent |
| effective rank | $(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ over the top-16 singular values of $\Sigma_p$ | one readout direction or many |

**The readout cross-covariance, in words.** At a step with token probabilities $p_v$, the next logits change when the fed embedding changes. For a small change $\delta e$, the change of the logit vector is $\delta z = W\,\delta e$ (linearised). The quantity that controls how large $\delta z$ can be for a unit $\delta e$ is the matrix
$$\Sigma_p=\sum_v p_v\,(E_v-\bar e)(W_v-\bar w)^\top,\qquad \bar e=\sum_v p_vE_v,\quad \bar w=\sum_v p_vW_v,$$
where $E_v$ is the input embedding of token $v$ and $W_v$ its output-head row. It is a weighted cross-covariance between “how the embedding moves” and “how the logits move”. If only two tokens $i,j$ carry the mass, $\Sigma_p=p(1-p)(E_i-E_j)(W_i-W_j)^\top$, which is rank one and has
$$\sigma_{\max}(\Sigma_p)=p(1-p)\,\|E_i-E_j\|\,\|W_i-W_j\|=p(1-p)\,G_t.$$
That is the gate law. The measured content is $G_t$ (the pair geometry), not the identity. The top-16 singular values of $\Sigma_p$ give the effective rank; for trained models it is 1–1.1 because two tokens dominate.

---

# A. Fragility and flux tubes

![Fig. A1](update_figures/pack_rstar_all.png)

**Fig. A1 — all models, critical radius.** (a) Complementary ECDF $P(r_*>x)$; (b) density of $\log_{10}r_*$; (c) violin of the same, white dot = median. **Data:** 5 prompts × 512 sparse-simplex directions for the small models (`results/rstar/*/p1..p5`), 3 prompts × 1000 directions for the 14B models (`results_big/*/a1_*`); teacher-forced, $r=0.1$, 300 tokens, ties $m_t\le0.02$ excluded. **Precision:** these are the *old* $r_*$ runs; for Qwen2.5/Qwen3-4B/Phi-4-mini they are fp16/floor-limited and the probe radius is not in the linear regime. Use Fig. C5 (dense directions) for the comparable value. **Read:** medians are 0.10 (Phi-4-mini), 0.20 (Qwen2.5/Qwen3-4B/Qwen3.5), 0.35 (DeepSeek-1.5B), 2.2 (DeepSeek-14B), 6.5 (Phi-4), 17.1 (Qwen3-14B). The density is a staircase, not a smooth law: 9–65 distinct values per prompt. **Trivial/not:** the ordering is a real measurement; the absolute scale is not comparable across these runs.

![Fig. A1b](update_figures/pack_rstar_fit_check.png)

**Fig. A1b — is the log-logistic fit as good as the $R^2$ says?** Left: DeepSeek, right: Qwen3-4B; measured survival (dots), fit in linear $S$ (blue), fit in $\log S$ (red). **Data:** same as A1. **Read:** fitting in log space changes the parameters and improves Qwen2.5/Qwen3.5 ($R^2_{\log}=0.91$ vs $0.77/0.85$) but lowers DeepSeek slightly ($0.95$ vs $0.96$). A KS test on the measured radii rejects the log-logistic for every model ($D=0.16$–$0.32$). **Conclusion:** the fit is descriptive, not a law; the large $R^2$ in linear space is dominated by the large-$S$ region. This confirms your suspicion.

![Fig. A2](update_figures/pack_rstar_decomposition.png)

**Fig. A2 — why $r_*$ is what it is.** (a) $r_*$ against the margin at the controlling step $m_{t_*}$; (b) against the local sensitivity $s_{t_*}=|\delta m_{t_*}|/r$; (c) Pearson correlation $r(\log m_{t_*},\log r_*)$ per model. **Data:** same as A1. **Read:** $r_*=m_{t_*}/s_{t_*}$ by definition; margins explain most of the variance (correlation 0.8–0.98). **Trivial:** the identity is trivial; the measurement is that a few low-margin steps control the distribution.

![Fig. A3](update_figures/pack_rstar_vs_gain.png)

**Fig. A3 — fragility is decoupled from bulk gain.** $r_*(u)$ against $G(u)=\|\delta h_L\|/r$ for every direction, per model; the title gives Pearson $r$ on $\log$–$\log$. **Data:** per-prompt A1/rstar runs, 20–1000 directions, $r=0.1$, $t=50$, last saved layer. **Read:** correlation is between $-0.1$ and $+0.1$; a direction can have large gain and be robust, or small gain and flip. **Why:** the decision gradient is one direction in $10^5$ dimensions; its alignment with the bulk singular directions is random. **Not trivial** for the claim “fragility = boundary crossing, not stretching”.

![Fig. A4](update_figures/pack_rstar_bottleneck.png)

**Fig. A4 — five steps decide most flips.** Histogram of the controlling step $t_*=\arg\min_t r_c$ for every direction; title gives the share of the 5 most frequent steps. **Data:** as A1, first 120 generated tokens shown. **Read:** 89–97% of directions are decided by five steps. **Consequence:** the “smooth ECDF” is the order statistic of a few margins, not an attractor boundary.

---

# B. Depth

![Fig. B1](update_figures/pack_depth_all.png)

**Fig. B1 — the real depth gain.** (a) $\mathrm{cum}_\ell=\log\sigma_1(J_\ell\cdots J_1)$ against normalized depth, median over 8–16 probe tokens and 8 directions; band = IQR over tokens. (b) Local gain $g_\ell=\langle\|J_\ell v_i\|\rangle_i$ against block. (c) $g$ in block 0, middle half, last 10%. **Data:** `run_depth2.py`/`run_depth_big.py`, Jacobian–vector products at one position, $k=8$; no perturbation radius. **Read:** product gain is $\times3.7$–$\times28$; block 0 is $3$–$17\times$; later blocks $0.98$–$1.13$. Block 0 contributes 85% of the log gain in DeepSeek-1.5B but only 2% in Phi-4. **Trivial/not:** the old “stretched decay” was $1/\ell$ by construction; the new numbers are the measured product.

![Fig. B4](update_figures/pack_depth_multitoken.png)

**Fig. B4 — every probe token.** $\mathrm{cum}_\ell$ for every probe token (8–16 per model), all 8 models. **Read:** the shape is stable across tokens, the scale varies by up to $\times2$. **Data need:** 8–16 tokens is enough for a median but not for the tails; the new $k=32$ run is in `results/update/depth_k32`.

![Fig. B4b](update_figures/pack_depth_tailfit.png)

**Fig. B4b — fit to the depth curve after block 0.** Median $\mathrm{cum}_\ell$ (dots), IQR over probe tokens (band), and a cubic fit $\mathrm{cum}=a+b\ell+c\ell^2+d\ell^3$ over layers $\ell\ge1$ (solid). **Data:** same as B1. **Read:** the cubic is descriptive ($R^2=0.89$–$0.98$) but its coefficients are not a universal law. The one robust feature is positive curvature: the gain rises again in the last layers. **This replaces the rejected $1+Ae^{-\ell/\tau_1}+B e^{(\ell-L)/\tau_2}$ fit**, which forced a block-0 spike and did not describe Phi.

![Fig. B4c](update_figures/pack_depth_late_rise.png)

**Fig. B4c — the late-depth rise on its own.** (a) Local gain $g_\ell$ after block 0, divided by the median of the middle half, for every model. (b) The ratio $g_{\rm last10\%}/g_{\rm middle}$. **Data:** same as B1. **Read:** the ratio is 1.00–1.11; the rise is small but consistent for 7 of 8 models (Phi-4 is flat). **Trivial/not:** a 2–11% effect is not a dramatic law; it is a real, small regularity, and it is what makes a global fit curve upward.

![Fig. B4d](update_figures/pack_depth_k32_all.png)

**Fig. B4d — direction-count check.** $\mathrm{cum}_\ell$ with $k=8$ (solid) and $k=32$ (dashed) tracked directions, median and IQR over tokens, for DeepSeek-1.5B, Qwen2.5-3B and Phi-4-mini. **Data:** `results/depth2/*` and new `results/update/depth_*_k32`. **Read:** the curves overlap within the token IQR; the direction count does not change the shape. **This is the distribution check you asked for.**

![Fig. B5](update_figures/pack_depth_relative.png)

**Fig. B5 — relative response.** $\|\delta h_\ell\|/\|h_\ell\|$ at $t=50$, normalized at layer 1, against $\ell/(L-1)$ for all models; band = IQR over directions (small models). **Definition:** the numerator is the response norm, the denominator the baseline state norm; the ratio says whether the perturbation grows *relative to the state*. **Read:** only Phi-4-mini amplifies (×20); DeepSeek peaks at ×3 then declines; 14B models end at 0.33–0.43 of layer 1; Qwen3.5 is flat. **Data:** A1 fp32/fp16 as in the tier table; big models from the compact summaries.

![Fig. B5b](update_figures/pack_relative_controls.png)

**Fig. B5b — the same with controls.** Relative response vs depth for every LLM and Mamba/Pythia/random, band = IQR over directions where available. **Data:** control A1 files (`results/ctrl/*/a1i_txt`), $r=0.1$, $t=50$, last saved layer; compact summaries in `results_big_small/ctrl_*.npz`. **Read:** Mamba and Pythia also stay within a factor 2 of 1; the random control is high and noisy. The relative response is not a transformer-specific signature; only Phi-4-mini clearly amplifies.

![Fig. B6](update_figures/pack_layer_gain.png)

**Fig. B6 — does the response grow with depth? (your observation, quantified, with controls).** (a) $\|\delta h_\ell\|$ at $t=50$ normalized first to the layer-1 response and then to the middle-layer response, against normalized depth, for all 8 LLMs and the 3 controls; (b) last/middle ratio. **Data:** A1 compact summaries, $r=0.1$; 14B models are sampled at 9–14 layers. **Read:** every trained model rises by a factor **3.5–14** from mid to last layer (DeepSeek-1.5B 3.5, DS-14B 4.2, Qwen2.5 6.7, Qwen3-4B 7.1, Qwen3.5-2B 7.3, Qwen3-14B 11.7, Phi-4-mini 11.0, Mamba 14.3). The random transformer is flat (**1.7**) and Pythia is nearly flat (**1.2**). **Interpretation:** the late-depth amplification of the *prompt* perturbation is a property of the trained weights, not of the architecture — a randomly initialised transformer of the same shape does not have it, and Mamba (no attention) does. **Caveat:** the layer-1 response is small, so the first points of the normalized profile are noisy; the last/middle ratio in (b) is the robust number.

---

# C. Geometry: angles, rotation, composition

![Fig. C1](update_figures/pack_angles_violin_controls.png)

**Fig. C1 — response-vs-state angle, all LLMs and controls.** Violin of $\angle(\delta h_\ell,h_\ell)$ at the last saved layer, $r=0.03$; white dot = median, vertical bar = IQR. **Data:** 64 directions for the small models (32 for controls) × 200–300 tokens; 24 directions for the 14B models. **Read:** medians 86.8°–91.4° for every model, including Mamba, Pythia and the random transformer. The *spread* is model-dependent, but the median is not discriminative. **Trivial/not:** the median 90° is the high-dimensional null; the response angle cannot separate trained models from controls.

![Fig. C2](update_figures/pack_angle_vs_depth_controls.png)

**Fig. C2 — angle vs depth, LLMs and controls.** Median (band = IQR over directions/tokens) at every saved layer. **Read:** the angle stays near 90° at every layer for every model, including the controls. **Use:** this is the honest version of “the response is almost tangential”; the radial component is the measured part (Fig. C1), not the median angle.

![Fig. C3](update_figures/pack_rotation_controls.png)

**Fig. C3 — rotation of the amplification direction, with a control.** (a) Violin of the principal angle between top singular vectors of consecutive block Jacobians; white dot = median, bar = IQR. (b) ECDF of the same. **Data:** `nonnorm.npz` for DeepSeek-1.5B/14B, Qwen3-14B, Phi-4 and the random transformer (5 tokens × saved layers). **Read:** trained models rotate by 78–87°; the random control is 89°; random directions are 90°. The trained models are measurably more aligned than random, but the effect is modest. **Why this matters:** every block can amplify ($\sigma_{\max}>1$), but the amplified direction changes from block to block, so a fixed vector sees the small common gain. The quantitative product comparison is in §K and in the earlier C3 panels (kept in `update_figures/pack_rotation_composition.png`).

![Fig. C4](update_figures/pack_dense_slopes.png)

**Fig. C4 — sparse vs dense directions: the correct reading.** Absolute response $\|\delta h_{50,L}\|$ against $r$ for sparse simplex (dashed) and dense Gaussian (solid) directions, with IQR bands over directions. **Data:** fp32 runs `results/update/fp32_*` (sparse, 8 radii) and `results/update/dense_*` (dense, 3 radii, 128 dirs). **Read:** both ensembles are linear ($\beta\approx1$) over $r=0.03$–$0.3$; dense is a constant factor 3–10× *higher* at the same $r$. The sparse curve is not saturating in this interval; my earlier claim that it was came from comparing the old fp16 sparse runs to the new fp32 dense runs. Below $r\approx10^{-3}$ the sparse response hits the fp32 subtraction floor. **Consequence:** direction sparsity changes the *scale* of the response, not the exponent; the old $r_*$ values at $r=0.1$ are still probe-dependent because $r_*$ depends on $\delta m$, not $\delta h$.

![Fig. C5](update_figures/pack_rstar_dense2.png)

**Fig. C5 — comparable fragility.** (a) $r_*$ against probe radius, sparse (dashed) and dense (solid), 95% bootstrap band over directions; (b) dense $r_*$ distributions at $r=0.03$. **Data:** 128 dense directions, radii 0.03/0.1/0.3; Phi-4 dense from the Spark run (64 dirs, 4 radii). **Read:** dense $r_*$ is radius-independent: DeepSeek 0.37, Qwen3.5 0.36, Qwen2.5 0.58, Phi-4 1.21–1.28. Sparse $r_*$ moves with the probe radius. **This is the number to compare across models.** The dense distributions are broad and overlap; the medians differ by less than a factor 2 among the small models, and Phi-4 is about 2× more robust than Qwen2.5.

---

# D. Time dynamics

![Fig. D1](update_figures/pack_time_all_models.png)

**Fig. D1 — response vs token, every model.** $\|\delta h_{t,L}\|/\|\delta h_{1,L}\|$ for a teacher-forced prompt perturbation, median over directions, last saved layer. **Data:** A1 runs; $r=0.1$ (Phi-mini 0.03), 300 tokens. **Read:** DeepSeek-1.5B decays to 0.2 by 300; Qwen2.5/Qwen3-4B/Phi-mini plateau; the 14B models plateau or decay slowly; none grows exponentially. **Precision:** the fp16 models' absolute values are floor-dominated; only the shape is meaningful.

![Fig. D1b](update_figures/pack_time_all_controls.png)

**Fig. D1b — the same with controls.** All 8 LLMs plus Mamba-130m, Pythia-410m and the random transformer, last saved layer, normalized at $t=1$. **Data:** A1/compact runs and `results_big_small/ctrl_*.npz`, $r=0.1$, 300 tokens. **Read:** the controls lie inside the LLM band; Mamba and Pythia decay or stay flat; the random control is noisier and starts high. No system grows. **Use:** this is the figure that answers “missing controls” for the time series.

![Fig. D2](update_figures/pack_time_controls.png)

**Fig. D2 — network-only replay vs controls.** Same quantity for a *replay* run (the model is fed the recorded token stream) for three LLMs and three controls. **Data:** `results/softloop/*/T1e-9/replay`, 600 steps, 8–32 directions. **Read:** all six curves are flat or decaying; Mamba (no attention) carries the response as well as the transformers; Pythia decays; the random control starts high and stays high. **Not trivial:** it shows attention is not required to carry a perturbation; it is required for the long-range *transport* (Fig. I1).

![Fig. D3](update_figures/pack_time_by_layer.png)

**Fig. D3 — first/middle/last layer.** Same as D1 at the first, middle and last saved layer. **Read:** in DeepSeek the shallow layers flush first and the deep layers persist; in Qwen/Phi the layers move together.

![Fig. D3b](update_figures/pack_time_layers_controls.png)

**Fig. D3b — layer-resolved response with controls.** Three layers (excluding layer 0), every LLM and control. **Data:** same as D1b. **Read:** the ordering of the layers is the same in the controls: deep layers retain more; no layer grows exponentially. **Use:** this is the requested “one plot for different layers, with controls”.

![Fig. D3c](update_figures/pack_time_all_layers.png)

**Fig. D3c — all saved layers, three examples.** Response vs time for every saved layer of DeepSeek-1.5B, Qwen3-4B and randQwen; colour = layer (colourbar). **Data:** A1/control summaries, $r=0.1$, 300 tokens. **Read:** the shallow layers decay fastest and the deep layers plateau; every layer overlaps in randQwen. No layer grows. **Each model has its own panel below (D3d–D3n).**

![Fig. D3d](update_figures/pack_time_layers_DeepSeek_15B.png)

**Fig. D3d — DeepSeek-1.5B, every layer.** Layer colourbar on the right. **Read:** layers 7–21 flush toward 0.1–0.3; the last layer stays highest at ≈0.2–0.5. The layer ordering is monotone: deeper = more persistent.

![Fig. D3e](update_figures/pack_time_layers_DeepSeek_14B.png)

**Fig. D3e — DeepSeek-14B, every layer** (9 saved layers, 300 tokens). **Read:** same ordering as the 1.5B; the spread between shallow and deep is smaller.

![Fig. D3f](update_figures/pack_time_layers_Qwen25_3B.png)

**Fig. D3f — Qwen2.5-3B, every layer.** **Read:** monotone persistence with depth; the last layer stays at ≈0.5.

![Fig. D3g](update_figures/pack_time_layers_Qwen3_4B.png)

**Fig. D3g — Qwen3-4B, every layer.** **Read:** all layers stay within a factor 2 of each other; no layer amplifies over time.

![Fig. D3h](update_figures/pack_time_layers_Qwen35_2B.png)

**Fig. D3h — Qwen3.5-2B, every layer** (7 saved layers). **Read:** the deepest layer sits above the rest by ×2, the ordering is otherwise flat.

![Fig. D3i](update_figures/pack_time_layers_Qwen3_14B.png)

**Fig. D3i — Qwen3-14B, every layer.** **Read:** the clearest monotone depth ordering of the 14B models; the last layer stays at ≈0.8.

![Fig. D3j](update_figures/pack_time_layers_Phi_4_mini.png)

**Fig. D3j — Phi-4-mini, every layer.** **Read:** an initial *rise* (this is the model that amplifies relative to its own state norm, Fig. B5), then a slow decay; the deepest layer is highest.

![Fig. D3k](update_figures/pack_time_layers_Phi_4.png)

**Fig. D3k — Phi-4, every layer.** **Read:** shallow layers decay fast, deep layers stay near 1: the same depth trend as the rest.

![Fig. D3l](update_figures/pack_time_layers_Mamba_130m.png)

**Fig. D3l — Mamba-130m (control), every layer.** **Read:** deeper layers persist more than shallow, exactly as in the transformers — “deeper responds more” is not a transformer-specific property.

![Fig. D3m](update_figures/pack_time_layers_Pythia_410m.png)

**Fig. D3m — Pythia-410m (control), every layer.** **Read:** all layers decay to ≈0.05; the depth spread is the smallest of all models, consistent with its near-flat last/middle ratio (1.2, Fig. B6).

![Fig. D3n](update_figures/pack_time_layers_randQwen.png)

**Fig. D3n — randQwen (control), every layer.** **Read:** this is your observation, quantified: all layers collapse onto a single band (0.6–1.1) that decays together, with no depth ordering. The trained models' monotone depth ordering (D3d–D3m) is therefore a property of the learned weights, not of the architecture or of depth per se.


![Fig. D4](update_figures/fig_time_dynamics_all.png)

**Fig. D4 — finite-time exponent.** (a) Response shape normalized; (b) replay vs controls; (c) $\Lambda(\tau)=\frac1\tau\log(a_{t+\tau}/a_t)$ against $\tau$ with error bars; (d) late rate $\lambda_{20}$. **Data:** A1 + replay. **Read:** $\Lambda(\tau)$ converges to $\le0$ for every system; no positive plateau. **Trivial/not:** the late-rate values are all within $\pm0.01$; the measurement is that a positive rate does not survive averaging. **Controls:** Mamba $-0.0008$, Pythia $-0.0019$, random $-0.0007$.

![Fig. D5](update_figures/pack_long_horizon2.png)

**Fig. D5 — 4000-token horizon, rebuilt from raw data.** (a) Response vs $t$ on a lin–log axis; (b) log–log; (c) local 20-token rate. **Data:** `results/local_data/a1_4k_test.npz`, 2 directions, 4 layers, $r=0.1$, last layer. **Read:** the response decays and fluctuates around a small floor; no sustained positive rate. **Trivial/not:** the decay is not a clean exponential; the measurement is that it never grows.


---

# E. Sliding window

![Fig. E1](update_figures/pack_window_exact.png)

**Fig. E1 — the exact window formula.** (a) Flush step $t^*_1$ vs $W$ for 97 windows; the line is $p_{\rm last}+W-P$, where $p_{\rm last}=43$ is the last prompt position carrying a nonzero perturbation and $P=65$. (b) $t^*_\ell$ vs layer for several windows; lines are $t^*_\ell=W-P+p_{\rm last}+(\ell-1)(W-1)$. (c) Residual from that formula: zero to the token for every measured layer/window. **Data:** `results/local_data/wspec/w*/a1.npz`, 106 windows $W=60$–$265$, 7 layers, 4 directions, 200 tokens. **Read:** the law is exact, not fitted. **The constant is not mysterious:** $t^*_1-p_{\rm last}-W+P=0$; my earlier “$W-22$” is just $p_{\rm last}-P=-22$. The perturbation support is sparse, so only positions up to $p_{\rm last}$ are perturbed.

![Fig. E1b](update_figures/pack_window_exact_models.png)

**Fig. E1b — the same formula on Qwen3-4B/Phi-4-mini.** (a) $t^*_1$ vs $W$; (b) $t^*_\ell$ vs layer. **Data:** `window4d/*/w*/a1.npz`; $p_{\rm last}=55$, $P=64$. **Read:** both models follow the same law with their own $p_{\rm last}$. **Interpretation:** the perturbation does not vanish at $W-P$; the last perturbed *generated* token carries it, and each layer adds one window of memory.

![Fig. E2](update_figures/pack_window_models.png)

**Fig. E2 — the same law on Qwen3-4B/Phi-4-mini, and why it cannot be resolved here.** (a) $t^*_\ell$ vs layer at $W=100$; (b) three windows; (c) the $(\ell,W)$ map, colour = $t^*_\ell$ (colourbar added). **Data:** `window4d/*/w*/a1.npz`, 11 saved layers, 6 directions, $r=0.1$, 300 generated tokens. **Read:** with $P=64$ and $p_{\rm last}=55$, the law predicts $t^*_1=W-9$ and $t^*_\ell=t^*_1+(\ell-1)(W-1)$; at $W=100$ that is already $t^*_2=190$ and $t^*_3=289$, so **every layer above the first hits the 300-token run length** — that is the flat top in (a)/(b) and the saturated cells in (c). Only the first layer and the smaller windows are measurable here, and they obey the law exactly. The two-model layer-resolved test is therefore done on DeepSeek-1.5B, where 106 windows and smaller $W$ keep $t^*_\ell$ inside the window (E1, E1b). **This panel set is kept for completeness; E1b is the figure to cite.**

![Fig. E3](update_figures/pack_window_agreement_fit.png)

**Fig. E3 — token agreement and its fit (onset excluded).** (a) Fraction of tokens identical to $W=4096$ for each window, with a stretched-exponential fit $f_\infty+(1-f_\infty)e^{-(t/\tau)^\beta}$ (dashed) started **at the first token where the agreement drops below 1** (dotted vertical line), not at $t=0$. (b) $1-\mathrm{agreement}$ on a log axis. **Data:** `results/window_tokens/tokens.npz`, 8 windows, 300 tokens, no perturbation. **Read:** the onset of disagreement is exactly $t_{\rm onset}=W-P$ (35, 85, 135, 235 for $W=100,150,200,300$, $P=65$) — the same clock as the flush law. The decay after onset is stretched-exponential with $\beta\approx1.7$–$4$ and $\tau\approx90$ ($W=100$), $162$ ($W=150$), $218$ ($W=200$). **Why the earlier fit looked wrong:** it was fitted from $t=0$, through the flat plateau, which dragged $f_\infty$ and $\tau$; excluding the plateau fixes the shape. $W=60<P$ is excluded from the fit because the window clips the prompt itself and the text differs from $t=0$.

![Fig. E4](update_figures/pack_attention_models.png)

**Fig. E4 — measured attention mass on the prompt.** (a) DeepSeek, four windows, median over heads (band = IQR over heads); (b) $W=100$, four models. **Data:** `run_attention_frac.py` with the **correct 4-D window mask**, 200 generated tokens, layers 0/7/14/21/27. **Read:** at $t=0$ the mass is high (0.9); it decays as the model shifts to generated tokens. For $W=100$ it drops sharply at $t\approx99$, when the **last prompt token (position 64, perturbed or not)** leaves the window. The response flush is earlier, at $t^*_1=p_{\rm last}+W-P=43+100-65=78$, because only the *perturbed* prompt positions (up to $p_{\rm last}=43$) matter. With $W=4096$ there is no mask cut; the mass decays gradually and never reaches zero in 200 tokens. **This is the real version of the old schematic “attention weight fraction” panel, and it resolves the apparent contradiction with the window formula.**

![Fig. E4b](update_figures/pack_attention_heads.png)

**Fig. E4b — any head still attending.** Fraction of heads with prompt attention $>10^{-4}$ vs $t$, DeepSeek, four windows. **Read:** a small number of heads keep a trace of the prompt longer than the mean suggests; for $W=4096$ some heads still attend at $t=200$. **Why this does not contradict the formula:** the formula is about the response at layer 1 becoming exactly zero when no *perturbed* token is in the window; it is not a statement about the total prompt-attention mass.

---

# F. Soft loop and the gate law

![Fig. F0](update_figures/pack_master_summary.png)

**Fig. F0 — the four central soft-loop results.** (a) Soft-loop response vs step at five temperatures (DeepSeek-1.5B, 600 steps, 8 directions, $r=0.03$); (b) closed loop vs replay on the same orbit at $T=1$; (c) gate law over all temperatures with the identity curve $G/4\cosh^{-2}(m/2T)$; (d) soft vs common-random-number vs independent sampling at $T=1$. **Read:** temperature changes the growth (a); the network alone is flat (b); the readout follows the identity (c); only mean feedback and independent sampling destabilise (d). This is the one-page version of the section.

![Fig. F0b](update_figures/pack_master_loglog.png)

**Fig. F0b — the log–log version of (b) and (d).** Same data, both axes logarithmic. **Read:** the closed loop is not a straight line in log–log; it is a staircase with long flat stretches and jumps. Replay is a slow decay. This is why a single exponential rate is a poor summary.

![Fig. F0c](update_figures/pack_loop_loglog_all.png)

**Fig. F0c — closed loop vs replay, log–log, every model.** 5-step rolling median over directions, $T=1$, longest orbit available per model (600–3000 steps; the 1600–3000 step runs are landing on the cluster and the figure regenerates from them automatically via `analysis_update/pack_loop_loglog_models.py`). **Read:** the closed loop climbs $10^2$–$10^4$ and *saturates*; the replay decays to 0.03–1. No model shows a straight line over the whole range — the growth is a staircase of gate events. Mamba grows the most ($\times10^4$) despite having no attention; the two 14B models and Qwen3.5-9B have 1600-step orbits and no long replay yet, so only their closed curves are shown. **Pairing rule:** the ratio $\log(a_{\rm closed}/a_{\rm replay})$ is always computed from a *matched* loop/replay pair recorded in the same directory (same orbit, same dtype), so the right-hand panels stop at the shorter of the two; the left-hand panels always show the longest loop.

![Fig. F0d](update_figures/pack_loop_loglog_ds15.png)

**Fig. F0d — DeepSeek-1.5B.** (a) Response, closed vs replay, log–log; thin = raw median over 8 directions, thick = 5-step rolling median, band = IQR over directions. (b) $\log(a_{\rm closed}/a_{\rm replay})$, grey = raw, green = smoothed, dashed = saturating fit $y_0+(L_\infty-y_0)(1-e^{-t/\tau})$. **Read:** the ratio saturates at $L_\infty=7.8$ nats ($\times2\,400$) with $\tau\approx60$ steps and $R^2=0.96$. This is the cleanest statement of the loop instability: a *finite* amplification that saturates, not exponential growth.

![Fig. F0e](update_figures/pack_loop_loglog_qwen25_3b.png)

**Fig. F0e — Qwen2.5-3B.** The closed loop grows by only $\times4$ and the ratio is noisy ($L_{\rm end}=1.0$, poor fit). **Read:** this model locks into confident repetition quickly, so the closed loop and the replay end up in similar states; the loop has little room to amplify.

![Fig. F0f](update_figures/pack_loop_loglog_qwen3_4b.png)

**Fig. F0f — Qwen3-4B.** Steady staircase growth ($\times5$), ratio rising linearly over 600 steps ($\lambda_{\rm fb}=0.004$ nats/step, the fit has not saturated).

![Fig. F0g](update_figures/pack_loop_loglog_qwen3_5.png)

**Fig. F0g — Qwen3.5-2B.** The largest growth among the small transformer LLMs ($\times300$) with a pronounced jump at $t\approx12$; no replay orbit was saved for this model, so only the closed loop is shown.

![Fig. F0h](update_figures/pack_loop_loglog_qwen35_9b.png)

**Fig. F0h — Qwen3.5-9B.** Closed loop $\times35$; the replay itself grows to $\times3$ and then decays, so the *ratio* is non-monotone. This is the case where the network alone is not perfectly neutral.

![Fig. F0i](update_figures/pack_loop_loglog_phi4mini.png)

**Fig. F0i — Phi-4-mini.** Closed loop $\times130$, ratio saturating at 5.2 nats with $\tau\approx70$ ($R^2=0.85$); the replay decays to 0.8.

![Fig. F0j](update_figures/pack_loop_loglog_mamba130.png)

**Fig. F0j — Mamba-130m.** The strongest amplification of all ($\times1\,800$; 9.6 nats, $\tau\approx35$ steps). **Why this matters:** the loop instability is *not* an attention artefact — a state-space model with no attention at all amplifies just as strongly when its own expected embedding is fed back.

![Fig. F0k](update_figures/pack_loop_loglog_pythia410.png)

**Fig. F0k — Pythia-410m.** $\times44$ ($L_{\rm end}=4.4$, slow rise $\tau\approx190$). The early part is flat for 20 steps: the perturbation has to cross the first gate.

![Fig. F0l](update_figures/pack_loop_loglog_randqwen.png)

**Fig. F0l — the random transformer.** Closed loop $\times3.3$, the smallest of all models, with an almost perfect saturating fit ($R^2=0.99$, $\tau\approx670$). **Read:** a random transformer *does* amplify when its own expectation is fed back, but 2–3 orders of magnitude less than a trained model at the same radius. The trained readout — not the architecture — is what makes the loop unstable.

![Fig. F0m](update_figures/pack_loop_saturation.png)

**Fig. F0m — the same comparison as one number.** (a) $L_{\rm end}=\log(a_{\rm closed}/a_{\rm replay})$ at the end of the run, with the equivalent linear factor; (b) $L_{\rm end}$ vs the rise time $\tau$ of the saturating fit. **Read:** the ordering is Mamba ($\times12\,000$) $>$ DeepSeek-1.5B ($\times2\,400$) $>$ Phi-4-mini ($\times173$) $>$ Pythia ($\times85$) $>$ Qwen3-4B ($\times10$) $>$ randQwen ($\times4$) $>$ Qwen2.5 ($\times3$). **Caveat:** $L_{\rm end}$ is horizon-limited; with the 1600–3000 step runs it will be recomputed, and the poor-fit points ($\times$) should only be read as lower bounds.

**Scoreboard ($T=1$, one orbit per model).** Every number below is in `analysis_update/pack_loop_loglog_models.json`, `pack_loop_intermittency.json`, `pack_loop_temperature.json`.

| model | steps (loop/replay) | growth closed | growth replay | $L_{\rm end}$ (nats) | $\times$ | $\tau$ (fit $R^2$) | top-5% share of growth |
|---|---|---|---|---|---|---|---|
| Mamba-130m | 600 / 600 | 1769 | 0.41 | 9.4 | 12 400 | 35 (0.79) | 27% |
| DeepSeek-1.5B | **3000** / 600 | 790 | 0.29 | 7.8 | 2 400 | 50 (0.95) | 26% |
| Phi-4-mini | 600 / 600 | 130 | 0.78 | 5.2 | 173 | 70 (0.85) | 73% |
| Pythia-410m | 600 / 600 | 44 | 0.35 | 4.4 | 85 | 186 (0.71) | 36% |
| Qwen3-4B | 600 / 600 | 5.3 | 0.57 | 2.4 | 10 | n/a (0.74) | — |
| Qwen3.5-9B | 600 / 600 | 35 | 3.1 | 2.2 | 9 | n/a (0.03) | 49% |
| randQwen | 600 / 600 | 3.3 | 0.86 | 1.3 | 3.5 | 669 (0.99) | 14% |
| Qwen2.5-3B | 600 / 600 | 4.3 | 1.3 | 0.9 | 2.6 | n/a (0.33) | 30% |
| Qwen3.5-2B | 600 / – | 297 | – | – | 297 | – | 56% |

`n/a` means the saturating fit is not a good description ($R^2<0.85$) — for those models the growth is still rising at the end of the window, so $L_{\rm end}$ is a lower bound. The "top-5% share" is the fraction of the total positive log growth contributed by the 5% largest one-step increments (F20c); 5% would be a constant-rate exponential.

### F.0b What the replay teacher trajectory actually is

**The replay teacher is not ordinary autoregressive generation — it is the closed soft loop of the unperturbed prompt.** In `run_softloop.py` row 0 of the batch is the unperturbed reference; in `--feedback soft` mode *every* row, including row 0, feeds back its own `softmax(z/T)` expected embedding. The `e_ref[t]` array that `--mode replay` later reads is therefore the reference row's *soft-loop* orbit, recorded step by step. Replay then feeds that frozen sequence to all rows, so the only difference from the closed loop is that the reference stops reacting to the perturbation.

Consequences you should keep in mind:

- At $T\to0$ the soft loop reduces to greedy autoregressive generation (the expected embedding becomes the argmax token embedding), and the script verifies this: `T->0 CHECK` compares the loop's argmax stream against greedy tokens. So at very low $T$ the replay teacher *is* the greedy trajectory; at $T=1$–$2$ it is not — it is a "soft greedy" path whose tokens are mixtures.
- The replay control therefore removes *two* things relative to the closed loop: the perturbation's effect on the readout and, through it, the perturbation's effect on the input stream. That is exactly what a control should do, but it means replay is not the same object as teacher forcing on hard tokens (the `longtf` runs), where the token stream is the greedy one at any $T$.
- Because the teacher orbit itself depends on $T$, a replay run must be paired with the loop run at the same $T$; we always pass `--replay-file <loop>/softloop.npz`.

### F.0 What “soft”, “closed loop”, “replay”, “CRN” and “independent” mean

Your understanding is correct. The precise definitions are:

- **Soft loop (closed loop).** At step $t$ the model computes logits $z_t$, turns them into $p_t=\mathrm{softmax}(z_t/T)$, forms the expected embedding $e_t=\sum_v p_t(v)E_v$, and feeds $e_t$ back as the next input. The context grows by one soft token per step. The perturbation is in the prompt (and in every subsequent state through the feedback).
- **Network-only replay.** The model is run on the *recorded* soft-token orbit $e_1,\dots,e_T$ of the reference (unperturbed) run. Only the perturbed prompt differs; the feedback is removed. This is a control: it measures what the network alone does to the perturbation, with the readout held fixed.
- **Teacher forcing (greedy corpus).** The model is fed the *recorded token ids* of the unperturbed greedy run; the perturbation is only in the prompt embeddings. This is the $T\to0$ limit of the replay, with hard tokens instead of expected embeddings.
- **CRN sampling.** At each step a token is sampled from $p_t$ of the reference row, and that *same* token is fed to every perturbed row (common random numbers). The sampling noise is identical across rows, so any growth is state dynamics.
- **Independent sampling.** Each row samples its own token. The rows separate at step 0 because the noise differs; this is ordinary sampled generation.

“Soft” means *expected embedding*; a “hard” loop would feed the argmax token (greedy) or a sampled token. Greedy is the $T\to0$ limit; sampled is the `sample_crn`/`sample_ind` cases. The closed loop is the most interesting because it is the only one where the perturbation can reach the readout and change the *distribution* that is fed back. Replay and teacher forcing are controls that remove that path; CRN is a control for sampling noise.

![Fig. F1](update_figures/pack_softloop_closed_all.png)

**Fig. F1 — closed loop vs replay, every model.** Response $\|\delta h_t\|/\|\delta h_1\|$ for the soft loop (solid) and the network-only replay (dashed), colour = temperature, 200 steps. **Data:** `results/softloop/*/T*/` and `replay/`, 8–32 directions, $r=0.03$–$0.3$. **Read:** closed loops grow by $10^2$–$10^4$; replays stay at 1 or decay. Mamba and Pythia (no/weak attention) also grow in closed loop; the random control is high in both. **The clean result:** the network is contractive/marginal; the feedback/readout is the instability.

![Fig. F2](update_figures/pack_loop_metric_M.png)

**Fig. F2 — mean log amplification, the non-arbitrary version.** $M=\langle \log(a_t/a_1)\rangle_t$ over the run, median over directions, at $T=1$; bars are closed loop, replay, CRN sampling and independent sampling. **Data:** `results/softloop*`, 600 steps, 8–32 directions; CRN/ind from `results/softloop_sample`. **Read:** closed loop has $M=1$–$7$ nats; replay and CRN are at 0 or negative; independent sampling is near 0. **Why this is better than “growth at step 100”:** it integrates the whole trajectory, so it does not depend on a chosen step. **Caveat:** $M$ is still horizon-dependent; the run length is fixed at 600, and we report it.

![Fig. F2b](update_figures/pack_loop_feedback.png)

**Fig. F2b — the feedback exponent.** (a) $\lambda_{\rm fb}=\mathrm{d}\log(a_{\rm closed}/a_{\rm replay})/\mathrm{d}t$ per model at $T=1$ (median and IQR over directions). (b) The ratio $a_{\rm closed}/a_{\rm replay}$ vs $t$. **Data:** same orbit in both runs. **Read:** $\lambda_{\rm fb}=0.002$–$0.006$ nats/step for most models, negative for Qwen2.5/Qwen3.5; the ratio still grows because replay decays while the closed loop stays flat or grows. **Interpretation:** the loop instability is not a clean exponential; it is a staircase of gate events on top of a slowly growing ratio. This is why a single “Lyapunov exponent” is not the right summary.

![Fig. F3b](update_figures/pack_live_density_sorted.png)

**Fig. F3b — live-gate density across models.** Density $P(m_t<T)$ vs $T$, sorted by $T$ (the lowest-temperature point is not connected to the highest). **Data:** `results/softloop*`, 300–600 steps, 8–32 directions, one orbit per $({\rm model},T)$. **Read:** the density is 0 below $T\approx0.2$ and $\approx1$ above $T\approx2$; the rise is at $T\approx1$ for trained models and at $T\approx0.13$ for the random control. **Why the old panel (b) is deleted:** horizon vs density does not collapse across models; the same density gives a 10× different horizon, and a two-parameter fit gives $R^2=0.20$. Density alone is not the order parameter.

![Fig. F3c](update_figures/pack_live_density_fit.png)

**Fig. F3c — the gate density is a margin distribution.** (a) $P(m_t<T)$ vs $T$ on linear axes; (b) median margin $\bar m$ vs $T$ on log axes; (c) the pooled margin distribution per model (bar = IQR, thin line = 5–95%, dot = median). **Data:** same as F3b, all temperatures pooled. **Read:** the "gate-opening temperature" is just the median margin in temperature units, and the median margin separates the control from every trained model by an order of magnitude: **randQwen 0.11 nats** vs 1.5 (Qwen2.5), 1.6 (Qwen3.5-9B), 1.8 (DeepSeek-1.5B), 1.9 (Phi-4-mini), 3.1 (Qwen3-4B), 3.5 (Qwen3.5-2B), 6.1 (Mamba, Pythia). **About fitting:** a Hill fit $aT^b/(c^b+T^b)$ is degenerate here — $b$ pins at any upper bound (4.999) because the curve is a sharp threshold, so the fit parameters are not meaningful and I report the non-parametric $T_{10}$, $T_{50}$, $T_{90}$ instead (`pack_live_density_fit.json`). The curve is also *not monotone in $T$*: each $T$ has its own orbit, and at some temperatures the loop locks into a confident repetition (large margins, low density). That is physics, not a plotting artefact, and it is why the density should not be treated as a thermodynamic curve.

![Fig. F4](update_figures/pack_gate_law_loglog.png)

**Fig. F4 — gate law, log–log and slopes.** (a) $\sigma_{\max}(\Sigma_p)$ against $m_t/T$ for all models, log–log; (b) fitted slope for $m_t/T>1$, with $G_{\rm eff}$ and $R^2$. **Data:** 4–11 temperatures per model, 8–32 directions. **Read:** slopes $-0.90$ to $-0.99$ for every trained model; the random control is $-0.83$ and Pythia $-0.90$, visibly off the identity. **Trivial/not:** the slope is the softmax identity; the measured content is the collapse and $G_{\rm eff}$.

![Fig. F4b](update_figures/pack_gate_linlog.png)

**Fig. F4b — the same in lin–log.** (a) Data in linear $m/T$ and log $\sigma$; the dashed curve is the identity with $G=2$. (b) Median distance from the identity for $m/T>1$. **Read:** DeepSeek/Qwen2.5/Qwen3.5/Phi-mini/Pythia/randQwen are *below* the identity at large $m/T$; Mamba and Qwen3-4B are closest. The identity is a conditional two-token model, not a universal curve; the robust measurement is the tail slope, not the amplitude.

![Fig. F5](update_figures/pack_gate_law_per_model.png)

**Fig. F5 — per-model fit, including the random control.** Data (dots), tail fit (black), identity curve $G/4\cosh^{-2}(m/2T)$ (red), and for randQwen the fit restricted to $m/T>3$ (red dashed). **Read:** the random control's tail fit is $-1.01$ with $R^2=0.99$; the earlier “the control breaks the law” came from fitting through the flat $m/T<1$ region. **This is the fixed fit you asked for.** The identity curve, not a straight line, is what matches the full range; it matches better for Mamba/Qwen3-4B than for DeepSeek/Qwen2.5.

![Fig. F5b](update_figures/pack_gate_linlog_per_model.png)

**Fig. F5b — per-model lin–log version.** Same data and fits on linear $m/T$ axes. **Read:** the identity (red) is above the data for most models at large $m/T$; the tail slope (black) is the better description. For randQwen and Pythia the identity is visibly worse, consistent with a distribution that is not two-token.

![Fig. F7](update_figures/pack_gate_geometry.png)

**Fig. F7 — pair geometry $G_t$.** (a) Violin of $G_t=\|E_i-E_j\|\|W_i-W_j\|$ per model; (b) $G_t$ vs margin. **Data:** `results/softloop_gates/*/gates.npz`, 300 steps per orbit. **Read:** $G_t$ is a per-model constant to ±40%; Phi/Mamba are 25–100× larger; the random control is degenerate. **Not trivial:** the per-model constant is what makes Phi's gates strong.

![Fig. F8](update_figures/pack_loop_rates_by_model.png)

**Fig. F8 — network rate vs loop rate, per model.** (a) $\lambda_{\rm net}$ (replay) per model: box = IQR, dots = temperatures; (b) $\lambda_{\rm sep}$ (closed loop). **Data:** all temperatures, 300–600 steps. **Read:** every $\lambda_{\rm net}$ is within about $\pm0.006$ and centred on 0; $\lambda_{\rm sep}$ is broad and often positive. **Does this mean edge of chaos?** No. A zero *conditional* rate means the network neither amplifies nor contracts along a fixed token stream; it is the signature of marginal stability, not criticality. An edge-of-chaos claim would also need (i) a positive but near-zero *full* Lyapunov exponent of the autonomous dynamics, (ii) scale-free correlations or avalanche statistics, and (iii) a diverging correlation length. We have (i) only in the weak “no positive exponent” sense, and we have not measured (ii)/(iii). The honest statement is: *the token-conditioned network is neutral/contracting; the loop instability lives in the readout.*

![Fig. F9](update_figures/pack_sampling.png)

**Fig. F9 — soft vs sampled feedback.** Response for soft feedback (red), common-random-number sampling (blue dashed) and independent sampling (green dashed) at $T=1$, six models. **Data:** `results/softloop_sample/*/{crn,ind}_T1.0`; Qwen3-4B was generated in the new run. **Read:** soft grows; CRN is flat at 1; independent separates at step 0. **Conclusion:** the instability is deterministic mean feedback, not sampled generation.

![Fig. F10](update_figures/pack_rank_entropy_box.png)

**Fig. F10 — readout rank and entropy.** Box = IQR, white dot = median, dots = individual steps. (a) Effective rank of $\Sigma_p$ over its top-16 singular values; (b) entropy of $\mathrm{softmax}(z)$ at $T=1$. **Data:** `sv_read`/`ent_ref` in `results/softloop*`, 300–600 steps. **Read:** trained models rank 1.0–1.1, random control 15.7; entropy 0.3–1.0 nat for trained models, 11.6 for the random control ($\log V$). **Caveat:** `ent_ref` is at $T=1$; “entropy vs $T$” is not a temperature effect. **Why box+strip:** the violin was hiding the multi-modality; this shows the actual spread.

![Fig. F11](update_figures/legacy_d2pipe.png)

**Fig. F11 — the D2 / TwoNN pipeline on the soft-loop state clouds.** This is the existing real figure (it loads `results/figures/softloop/d2pipe_softloop.json`), included because you liked it. **Read:** the $T\to0$ and $T=1$ orbits are drifting repetition loops and are correctly flagged; the $T=0.5$ and $T=2$ orbits are aperiodic and have no scaling region. It is the same verdict as the greedy corpus: D2 is descriptive, not evidence.

![Fig. F13](update_figures/pack_loop_metric_T.png)

**Fig. F13 — mean log amplification vs temperature.** $M=\langle\log(a_t/a_1)\rangle_t$, closed loop (solid) and replay (dashed), per model. **Data:** `results/softloop*`, all temperatures, 600 steps. **Read:** closed $M$ is positive almost everywhere; replay is near zero or negative. The dependence on $T$ is orbit-specific, not monotonic; the robust statement is the closed/replay gap.

![Fig. F14](update_figures/pack_loop_fsle.png)

**Fig. F14 — finite-scale growth rate (FSLE).** For thresholds $S=2,4,8,16,32,64$, the first-passage time $t_S$ and the rate $\log S/t_S$, closed vs replay, at $T=1$. **Data:** same as F13. **Read:** the closed loop reaches larger $S$ in finite time; replay does not. The curves are not straight lines, so the growth is not a single exponential; the rate decreases with $S$ in several models.

![Fig. F15](update_figures/pack_loop_staircase.png)

**Fig. F15 — the staircase.** (a) One-step log increments $\Delta\log a_t$ for the closed loop, $T=1$; (b) autocorrelation of the increments. **Read:** the increments are spiky and correlated; growth happens in bursts at gates, not as a constant drift. **This is the quantitative form of “staircase, not exponential”.**

![Fig. F16](update_figures/pack_loop_endpoint.png)

**Fig. F16 — endpoint rates.** Closed-loop endpoint rate vs replay endpoint rate, one point per model. **Read:** the points lie above the diagonal; the loop adds growth on top of whatever the network does. The absolute values are small, so this is a *qualitative* separation, consistent with F2/F2b.

![Fig. F17](update_figures/pack_loop_finitetime.png)

**Fig. F17 — finite-time exponent.** $\Lambda(\tau)=\frac1\tau\log(a_{t+\tau}/a_t)$ vs $\tau$, closed (red) vs replay (blue), $T=1$. **Data:** `results/softloop*`, 600 steps. **Read:** the closed loop is positive at all $\tau$ for most models; replay is near zero or negative. The curves are not flat, so there is no single asymptotic exponent; the finite-time value is the meaningful object.

![Fig. F18](update_figures/pack_loop_spectrum.png)

**Fig. F18 — power spectrum of the log response.** PSD of $\log a_t$ (quadratic trend removed), closed vs replay, $T=1$. **Read:** the closed loop has more low-frequency power (slow staircase), while replay is closer to a flat/noisy spectrum. **Use:** this is another way to show that the loop growth is a slow, structured process, not white noise.

![Fig. F19](update_figures/pack_loop_selfreg.png)

**Fig. F19 — why the growth stops: the loop self-regulates into confident repetition.** Red, left axis: response $a_t$ of the closed loop (log). Blue, right axis: the reference row's margin $m_t$ (21-step median, log). Green shading: the live-gate region $m_t<1$, i.e. the steps on which the readout is soft enough for the perturbation to change the feedback distribution. **Read:** growth happens only inside the green region, and it stops precisely when the margin rises above 1 and stays there. DeepSeek-1.5B is the clearest case: the response climbs to $\times800$ over the first ~300 steps while the margin is $O(1)$, then the margin settles at ≈10 nats and the response is flat for 2700 more steps (only 2% of steps have $m_t<1$). Phi-4-mini and Mamba show the same lock-in after a single gate-opening episode (at $t\approx110$ and $t\approx40$–200). **The exception is the random transformer**, whose margins never exceed ≈0.3 nats: its gate is always open (100% of steps), it never locks in, and it is the only model whose growth looks exponential over the whole run (F0l). Qwen2.5-3B never locks in either and its response keeps wandering. **Interpretation:** the saturation $L_\infty$ in F0d–F0m is not an imposed cutoff — it is a *fixed point of the coupled system*: growth drives the trajectory toward high-confidence repetitive text, which closes the gates and switches growth off. This is a self-regulating mechanism, and it is why a single Lyapunov exponent does not describe the loop.

![Fig. F19b](update_figures/pack_loop_selfreg_big.png)

**Fig. F19b — the same at 14B, where it is a one-shot ratchet.** DeepSeek-14B and Qwen3-14B, 1600-step orbits. **Read:** for the first ~200 steps the response is $O(1)$ with repeated spikes while the margin wanders around 1–30; at $t\approx200$ a single gate episode lifts the response to $200$–$350$, and it then stays there for the remaining 1400 steps even though the margin later dips below 1 again (green) many times. **So the amplified state is not undone when the gate reopens** — the loop has moved to a different self-consistent cycle and the perturbation rides at that cycle's amplitude. This is the phenomenology of a **stable limit cycle**: perturbation growth saturates at the cycle amplitude, and the "Lyapunov exponent" of the loop is zero on the cycle. It also explains why the 14B curves in F0c are so step-like compared with the 1.5B: fewer, larger gate events.

![Fig. F20](update_figures/pack_loop_intermittency.png)

**Fig. F20 — the growth is bursty and gate-conditioned.** (a) Windowed live-gate density $P(m_t<1)$ (window 50) — the gates open in bursts, and after the lock-in of F19 the density stays at 0. (b) Mean one-step log growth $\Delta\log a_t$, split by gate state: every trained model grows only while the gate is open (red, +0.02 to +0.15 nats/step) and *contracts* while it is closed (blue, −0.01 to −0.05 nats/step); Pythia is the extreme case (−0.5 nats/step when closed). (c) Concentration of the total log growth: the top 5% of steps produce 26–73% of the growth for the trained models (Phi-4-mini 73%), versus ~14% for randQwen and the 5% diagonal of a constant-rate exponential. **Read:** the staircase is not a plotting artefact; a handful of gate-opening events do most of the work, and between them the response decays.

![Fig. F21](update_figures/pack_loop_temperature.png)

**Fig. F21 — the instability is a threshold in softness, not a smooth function of $T$.** DeepSeek-1.5B, $r=0.03$, fp32, 11 temperatures, 600 steps; (a) $L(t)=\log(a_{\rm closed}/a_{\rm replay})$; (b) $L$ at $t=300$ against $T$, log $x$. **Read:** at the greedy limit ($T=10^{-9}$, grey square) the closed loop *is* ordinary autoregressive generation while the replay is teacher forcing, and $L=0.0$ — the two coincide within the floor. At **every** temperature above $T=0.05$, $L=3.8$–$10.4$ nats, with no systematic trend in $T$ (the dip at $T=0.1$ and at $T=2$ is orbit-to-orbit variability: one orbit per $T$). **Interpretation:** what matters is that the feedback is a *distribution* at all, not how soft it is. A one-hot limit removes the instability entirely; a barely-soft loop ($T=0.05$, top-2 mass $\approx1$) already has the full effect. This is the cleanest evidence that the mechanism is the readout→embedding path rather than sampling noise. **Why only one model here:** the probe radius differs across models (0.03 vs 0.1) and at $T\to0$ the response is dominated by discrete token flips, which depend on the radius, so a cross-model $T\to0$ comparison would be confounded.


![Fig. F11](update_figures/legacy_d2pipe.png)



---

# G. Linearity and precision

![Fig. G1](update_figures/pack_linearity_precision.png)

**Fig. G1 — precision and probe radius.** (a) $\|\delta h_{50,L}\|$ vs $r$ for fp32 and fp16 runs of the same/nearby models; (b) local exponent $\beta={\rm d}\log\|\delta h\|/{\rm d}\log r$; (c) $r_*$ estimated at different probe radii. **Read:** fp16 has a constant floor (0.5–4) independent of $r$; fp32 scales with $\beta\to1$ for $r\lesssim10^{-2}$; $r_*$ is only stable at small probe radius. **This is the figure that invalidates the old absolute comparisons.** **Data:** new fp32 runs `results/update/fp32_*`, old fp16 runs `a1r_*`, DeepSeek 12-radius `w100_layers`.

![Fig. G2](update_figures/pack_dense_slopes.png)

**Fig. G2 — sparse vs dense, corrected.** Same as C4. **Read:** both are linear over 0.03–0.3; dense is a constant factor higher. My earlier statement “sparse saturates” was wrong: it came from the old fp16 sparse runs, not from the fp32 sparse runs. The correct consequence is that direction sparsity changes the *scale* of the response, and the old absolute values are only comparable within the same direction ensemble.

![Fig. G3](update_figures/pack_response_vs_radius.png)

**Fig. G3 — response vs radius, corrected.** (a) Absolute response vs $r$ for the fp32 models; (b) local exponent; (c) superposition error vs total norm; (d) replication across prompts. **Data:** DeepSeek 12 radii, Qwen2.5/Qwen3.5 8 radii, superposition 3 prompts. **Read:** all three are linear from $3\times10^{-3}$ to 0.3 ($\beta\approx1$); below $10^{-3}$ the fp32 subtraction floor bends the curve. Superposition error is 5–7% at total norm 0.24.

![Fig. G4](update_figures/pack_superposition.png)

**Fig. G4 — superposition, all prompts.** (a) $\|A-S\|/\|A\|$ vs total norm; (b) $\cos(A,S)$. **Data:** `results/superposition/*`, 65 positions, 2 trials, last layer. **Read:** error is 5–7% at the minimum (total norm 0.24), 36% at the fp32 subtraction floor; $\cos>0.997$. **Trivial/not:** the minimum is the measured linear-regime error; the rise at large norm is real saturation.

---

# H. Flips

![Fig. H1](update_figures/pack_flip_fraction_all.png)

**Fig. H1 — free-run flip fraction.** Fraction of directions that flip at least one token, against $r$; band = Wilson 95%. **Data:** small models 3 prompts × 20 directions (`results/local_data/models/*/chaos`), DeepSeek 512 directions (`results/hazard`), 14B models 1000 directions (`results_big/*/free_*`). **Read:** Qwen3-4B/Qwen2.5/Phi-mini are 35–60% flat; DeepSeek rises from 3% to 60%; Phi-4 is near 0; 14B models rise. **Caveat:** the small-model runs use 20 directions and sparse perturbations, so the bands are wide; the ordering, not the exact value, is the result.

---

# I. Transport

![Fig. I1](update_figures/pack_transport_kernel.png)

**Fig. I1 — transport kernel.** (a) $\log_{10}K(p,t)$ for the median over prompts; (b) rows $K(p,\cdot)$ at several prompt positions. **Definition:** $K(p,t)=\|\delta h_{t,L}\|/r$ when only prompt position $p$ is perturbed. **Data:** `results/transport/deepseek_p*`, 3–5 prompts, 8 directions, $r=0.1$, 300 tokens, last layer. **Read:** a persistent background from the first positions plus a travelling front at $t\approx p+20$–$40$. **Model coverage:** only DeepSeek has saved kernels; the 14B transport runs would need new data.

![Fig. I2](update_figures/pack_superposition.png)

**Fig. I2 — superposition (same as G4).** The kernel is a valid decomposition at total norm 0.24 to 5–7%.

---

# J. Controls (real figures from `controls.md`)

These are the existing control figures; the scripts load the control `.npz` files (no synthetic arrays). They are the reason several old “LLM signatures” are not evidence.

![Fig. J1](update_figures/ctrl_c2_resp_time.png)

**Fig. J1 — Mamba carries the kick.** Response $a(t)/r$ for Mamba-130m and DeepSeek-1.5B; median and 10–90% band. **Read:** Mamba does **not** decay to zero; it carries 0.2–0.9 for 300 tokens. The “attention carries the perturbation” statement is too strong; attention carries a ~3× larger response.

![Fig. J2](update_figures/ctrl_c2_lambda.png)

**Fig. J2 — Mamba's rate fluctuates around zero.** $\lambda_{10}(t)$ for the control and DeepSeek; transient positive excursions up to +0.11 in the control. **Read:** transient growth alone does not separate chaos from a contracting system.

![Fig. J3](update_figures/ctrl_c3_obspred.png)

**Fig. J3 — the threshold rule is exact on the control.** Observed vs predicted first flips, 18 free runs, 25/25 exact. **Read:** the flip phenomenology is a property of argmax, not of chaos.

![Fig. J4](update_figures/ctrl_c3_survival.png)

**Fig. J4 — no re-synchronisation.** Token agreement after the first flip; 0/24 flipped runs re-synchronise. **Read:** “flux tube” phenomenology is reproduced by a system with all-negative Lyapunov exponents.

![Fig. J5](update_figures/ctrl_c8_lambda_all.png)

**Fig. J5 — late rate across all models and controls.** The LLM values sit inside the band of provably non-chaotic models.

![Fig. J6](update_figures/ctrl_c8_kernel.png)

**Fig. J6 — Mamba has no travelling front.** Kernel $K(p,t)$ for the control; front amplitude 1.0–1.2 vs 13–53 for DeepSeek. **Read:** long-lived influence is not attention-specific; re-amplification is.

![Fig. J7](update_figures/ctrl_c6_phase.png)

**Fig. J7 — random transformer gain sweep.** Depth exponent vs weight gain and token-axis rate. **Read:** the depth axis can be made strongly amplifying while the token axis stays neutral; weight scale alone does not produce token-axis chaos.

![Fig. J8](update_figures/ctrl_c7_rstar.png)

**Fig. J8 — thresholds across models.** $r^*$ for random init, DeepSeek, Pythia, Mamba. **Read:** $r^*$ tracks the margin landscape, not the architecture.

![Fig. J9](update_figures/ctrl_c5b_all_models.png)

**Fig. J9 — correlation dimension across all models.** **Read:** no model has a scaling region; the fitted $D_2$ spans 0.9–71 as a smooth function of the state-cloud dimension, with the contracting control at the low end. **Use:** this is the quantitative veto of the old D2 claims.

---

# K. Scorecard (updated)

| result | status | why |
|---|---|---|
| No sustained growth (TF/replay) | **keep** | 9 architectures, finite-time exponent $\le0$ |
| Closed loop grows; replay/CRN flat | **keep, feature** | $M=1$–$7$ nats vs 0; F1–F2 |
| Window $t^*_1=p_{\rm last}+W-P$, $t^*_\ell=t^*_1+(\ell-1)(W-1)$ | **keep, feature** | exact over 106 windows; E1 |
| Depth: block-0 spike + small late rise | **keep, qualify** | block-0 share 2–85%; late rise 1.00–1.11; B1/B4c |
| Larger models closer to marginal | **keep, qualify** | 7 models, block-0 share varies |
| Gate law slope $-1$ | **keep as identity** | F4–F5; $G_{\rm eff}$ is the measurement |
| Gate identity amplitude universal | **drop** | data below identity for most models; F4b/F5b |
| Random control breaks gate law | **drop** | false; $-1.01$ for $m/T>3$ |
| Perpendicular response | **drop as discovery** | median 90° in LLMs *and* controls; C1 |
| Rotation 78–87° | **keep, qualify** | real, modest; random control 89°; C3 |
| $r_*$ ordering | **keep only with dense directions** | C5; sparse values are probe-dependent |
| $r_*$ log-logistic law | **drop** | KS rejects it; A1b |
| Five-regime taxonomy | **drop** | four of five claims fail |
| Entropy rises with $T$ | **drop** | `ent_ref` is at $T=1$ |
| Readout rank 1 | **keep, qualify** | F10; consequence of two-token concentration |
| Hunt–Ott entropy zero | **drop** | absolute exponents are positive; relative is small |
| Edge of chaos | **do not claim** | no positive full Lyapunov exponent, no criticality measurement |
| Sparse directions saturate (my earlier claim) | **drop** | both sparse and dense are linear over 0.03–0.3; C4 |
| Common linear regime all models | **keep for fp32/dense; separate fp16** | C4/G1–G3 |

---

# L. Questions you asked, answered directly

- **Are the old figures synthetic?** The plotted arrays were generated in the plotting script; the experiments were real. Fixed in this document. §1.
- **Is the depth fit trivial?** The old one was a fit to $1/\ell$; the new B4b is a descriptive cubic after block 0, and B4c isolates the small 1.00–1.11 late rise. The old spike+exponential fit is deleted. B4b/B4c.
- **Is the perpendicular response trivial?** Yes, median 90° in LLMs *and* controls; the radial tail is the measured part. C1–C2.
- **Is the rotation real?** Yes, 78–87°, but the random control is 89°; modest. C3.
- **Why does the perturbation not vanish at $W-P$?** It is carried by generated tokens; the flush is one window later, then one window per layer. E1.
- **Does the flush depend on the layer?** Yes: $t^*_\ell=W-P+p_{\rm last}+(\ell-1)(W-1)$, exact over 106 windows. E1.
- **What is the constant in the window formula?** It is $p_{\rm last}-P$, where $p_{\rm last}$ is the last perturbed prompt position (43 for $P=65$; 55 for $P=64$). E1.
- **Is the five-regime table true?** Only the window row. §K, D2.
- **Does the random transformer decay in 4 tokens?** No; replay is 54× and flat. D2.
- **Do we have an edge-of-chaos argument?** No; zero conditional rate is marginal stability, not criticality. F8.
- **What is “tangent contraction rate”?** Two different quantities: $g_\ell\approx1$ and $\lambda_L=0.03$–$0.12$. Use those names. B.
- **Is rank-1 supported?** Yes for Qwen3-4B/Phi-4-mini (1.0/1.1), random 15.7; consequence of concentration. F10.
- **Entropy results?** Low for trained models, $\log V$ for random; `ent_ref` is at $T=1$. F10.
- **Are the ECDF fits real?** Descriptive only; a KS test rejects the log-logistic. A1b.
- **Do JL/Cramér–Wold apply?** Not with sparse directions; the Gaussian null fails. A1.
- **Is “sparse saturates” true?** No; both sparse and dense are linear over 0.03–0.3; dense is a constant factor higher. C4.
- **What is closed loop vs replay?** Closed feeds back its own expected embedding; replay feeds the recorded unperturbed soft tokens. F.0.
- **What is “soft”?** Expected-embedding feedback; “hard” would be argmax (greedy) or sampled. F.0.
- **Do the attention and flush formulas agree?** Yes; the flush is about a nonzero response, not about the mean attention mass. E4b.
- **What is new and worth featuring?** Closed vs replay (M and FSLE); exact window clock; block-0 + late-rise depth; dense-direction fragility; gate tail slope + $G_{\rm eff}$. §K.
- **What data do we need?** dense 14B runs; exact-$\Sigma_p$ validation; fp32 only where absolute response is needed. §M.

---

# M. Experiments: what was run, what is blocked

| # | experiment | status |
|---|---|---|
| 1 | fp32 A1 for Qwen2.5-3B, Qwen3.5-2B | **done**, `results/update/fp32_*` |
| 2 | fp32 A1 for Qwen3-4B, Phi-4-mini | **not done**; fp16 is adequate for flips/relative shape, fp32 needed only for absolute response |
| 3 | angle probe, 64 directions, 5 small models + 3 controls | **done**, `results/update/angle_*` |
| 4 | dense-direction $r_*$, 128 dirs, 3 fp32 small models | **done**, `results/update/dense_*` |
| 5 | dense 14B runs | **Phi-4 done** (`results/update/dense14_phi4`, $r_*=1.21$–$1.28$); Qwen3-14B/DS-14B hit OOM when colleagues held the memory, retried by `queue_spark_wait.sh` |
| 6 | DeepSeek dense window sweep, 106 windows | **done** (from existing `wspec`) |
| 7 | nonnorm controls | **randQwen done** (`nonnorm_randqwen3`); Pythia path mismatch; Mamba not run |
| 8 | depth probe with 32 directions | **done** for DeepSeek, Qwen2.5-3B, Phi-4-mini (`results/update/depth_*_k32`) |
| 9 | sampled feedback for Qwen3-4B (CRN/ind) | **done**, `results/softloop_sample/qwen3_4b` |
| 10 | exact $\Sigma_p$ validation (all vocab) | not run |
| 11 | attention mass, correct window mask | **done**, DeepSeek W=50/100/200/4096 + Qwen2.5/Phi/rand W=100 |
| 12 | closed-loop metric suite (M, FSLE, finite-time, spectrum, staircase, endpoint, saturation $L_\infty$/$\\tau$) | **done**, F2–F18 |
| 13 | log-logistic fit check (linear vs log space, KS) | **done**, A1b |
| 14 | variance/percentile pass on all aggregates | **done** where data allow; controls added to B5b/B6/C1/C2/C3/D1b/D3b |
| 15 | **long looping campaign, 3000 steps (x11) / 1600 steps (Spark)** | **running**: closed loop + replay at $T=1$, then $T=2$, CRN/ind, long teacher-forced (1000/800 tokens), then $T=0.5$ |

**Running on the GPUs (36–48 h, started 2026‑11‑26 22:50 CET).**

- **x11** (`scripts/queue_long_x11.sh`, log `results/longloop_x11.log`): 8 models (DeepSeek-1.5B, Qwen2.5-3B, Qwen3-4B, Qwen3.5-2B, Phi-4-mini, Mamba-130m, Pythia-410m, randQwen), 3000 steps, 4 saved layers, $r=0.03$–$0.1$, 8 directions. Rounds: (A) $T=1$ loop+replay for every model, (B) $T=2$ loop+replay, (C) CRN+independent sampling, (D) 1000-token teacher-forced, (E) $T=0.5$ loop+replay. ~15 min per 3000-step run for the 1.5B model, longer for the 3–4B models; the whole queue is ~30 h. Resume-safe: each step is skipped if its `.npz` exists.
- **Spark** (`scripts/queue_long_spark.sh`, log `results/longloop_spark.log`): Qwen3-14B, DeepSeek-14B, Qwen3.5-9B, Phi-4 at fp16, 1600 steps (≈3.4 s/step, ~2 h per run), 4 saved layers. Same round structure with 800-token teacher-forced runs. ~30 h total. I only killed my own earlier jobs; the colleagues' memory is untouched and the models load in fp16 (~30 GB), which fits in the free 60 GB.
- Neither queue retries on failure: a failed step prints `FAILED` and the queue moves on, so nothing can spin.

**Re-running the analysis when the data land** (all on the Windows checkout after `analysis_update/sync_from_cluster.ps1`):

```
python analysis_update/pack_loop_loglog_models.py   # F0c-F0m: picks the longest orbit per model
python analysis_update/pack_loop_saturation.py      # F0m summary numbers
python analysis_update/pack_loop_metrics.py         # M, FSLE, staircase, endpoint
python analysis_update/pack_loop_finitetime.py      # finite-time exponent + spectrum
python analysis_update/run_all.py                   # everything, then copy to notebook/update_figures
```

---

# N. Appendix: earlier measured figures, kept for reference

These were in the previous version of this document. They are kept because you asked not to remove figures; several are superseded by the newer versions above, and the caption says which. All are measured from the `.npz` files (not synthetic).

![Earlier A](update_figures/fig_rstar_pdf.png)

**Earlier A — critical-radius density (the version you liked).** Same data as A1. Kept as a standalone density figure.

![Earlier B](update_figures/fig_entropy.png)

**Earlier B — entropy vs margin.** Same data as F10. Kept because you liked it.

![Earlier C](update_figures/fig_readout_rank.png)

**Earlier C — readout effective rank.** Same data as F10; F10 is the box+strip version with more models.

![Earlier D](update_figures/fig_depth_gain_all_models.png)

**Earlier D — the first depth figure.** Shows the $1/\ell$ quantity that was fitted in `unified.md`; superseded by B1.

![Earlier E](update_figures/fig_depth_exponent_vs_size.png)

**Earlier E — depth exponent vs model size.** The single-scatter version; B1/B4b are the distribution versions.

![Earlier F](update_figures/fig_gate_law.png)

**Earlier F — the first gate-law figure.** Superseded by F4/F4b/F5.

![Earlier G](update_figures/fig_softloop_open_closed.png)

**Earlier G — closed vs replay, DeepSeek only.** Superseded by F1 (all models) and F2 (metric $M$).

![Earlier H](update_figures/fig_relative_response_depth.png)

**Earlier H — relative response.** Superseded by B5/B5b (controls added).

![Earlier I](update_figures/fig_response_vs_radius.png)

**Earlier I — response vs radius.** Superseded by G3.

![Earlier J](update_figures/fig_phase_density.png)

**Earlier J — phase figure.** Superseded by F3b; the phase plot itself is deleted per your instruction.

![Earlier K](update_figures/fig_window_departure.png)

**Earlier K — window departure.** Superseded by E1/E1b (exact formula).

![Earlier L](update_figures/fig_angle_and_rotation.png)

**Earlier L — angle/rotation.** Superseded by C1/C2/C3 (controls added).

# N2. Limitations

- Qwen3-4B/Phi-4-mini absolute responses are fp16-floor-dominated; only discrete flips and relative shapes are valid.
- Sparse and dense directions give the same exponent but a different scale; do not mix them in absolute comparisons. The old sparse $r_*$ values depend on the probe radius; use the dense values (C5) for cross-model statements.
- The 14B compact files save 9–14 layers, not all; depth profiles for the 14B models are partial.
- The transport kernel is DeepSeek-only; angle/rotation controls are available for the random transformer but not for Mamba/Pythia.
- The gate law is the softmax identity; the amplitude does not match all models, and only the tail slope is universal.
- No positive full Lyapunov exponent, no Lyapunov spectrum, no criticality measurement: no edge-of-chaos claim.
- The small-model free-run flip fractions use 20 directions and sparse perturbations; the Wilson bands are wide.
- The closed-loop metrics are horizon-dependent; we report the run length and the integrated quantity $M$, not a unique Lyapunov exponent.

