# Update 2 — all old figures, rebuilt from raw data

**Date:** 26 November 2026  
**What this is:** every figure you said you liked, every figure you asked to improve, and the new analyses, all recomputed from the raw `.npz` files. The old `unified.md` figures are also replaced in place (with a warning banner) so `unified.pdf` now shows measured plots.

**How to rebuild everything:** `python analysis_update/run_all.py`. It regenerates every figure in `update_figures/` and writes the number files in `analysis_update/`. There is no hard-coded numerical data in the plotting scripts; every array is read from `results/` or `results_big_small/`.

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

![Fig. B5](update_figures/pack_depth_relative.png)

**Fig. B5 — relative response.** $\|\delta h_\ell\|/\|h_\ell\|$ at $t=50$, normalized at layer 1, against $\ell/(L-1)$ for all models; band = IQR over directions (small models). **Definition:** the numerator is the response norm, the denominator the baseline state norm; the ratio says whether the perturbation grows *relative to the state*. **Read:** only Phi-4-mini amplifies (×20); DeepSeek peaks at ×3 then declines; 14B models end at 0.33–0.43 of layer 1; Qwen3.5 is flat. **Data:** A1 fp32/fp16 as in the tier table; big models from the compact summaries.

![Fig. B5b](update_figures/pack_relative_controls.png)

**Fig. B5b — the same with controls.** Relative response vs depth for every LLM and Mamba/Pythia/random, band = IQR over directions where available. **Data:** control A1 files (`results/ctrl/*/a1i_txt`), $r=0.1$, $t=50$, last saved layer; compact summaries in `results_big_small/ctrl_*.npz`. **Read:** Mamba and Pythia also stay within a factor 2 of 1; the random control is high and noisy. The relative response is not a transformer-specific signature; only Phi-4-mini clearly amplifies.

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

**Fig. C5 — comparable fragility.** (a) $r_*$ against probe radius, sparse (dashed) and dense (solid), 95% bootstrap band over directions; (b) dense $r_*$ distributions at $r=0.03$. **Data:** 128 dense directions, radii 0.03/0.1/0.3. **Read:** dense $r_*$ is radius-independent (DeepSeek 0.37, Qwen3.5 0.36, Qwen2.5 0.58); sparse $r_*$ moves with the probe radius. **This is the number to compare across models.** The dense distributions are broad and overlap; the medians differ by less than a factor 2.

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

**Fig. D3c — all saved layers.** Response vs time for every saved layer of DeepSeek-1.5B, Qwen3-4B and Mamba-130m. **Data:** A1/control summaries, $r=0.1$, 300 tokens. **Read:** the shallow layers decay fastest; the deep layers plateau. No layer grows. **Use:** a single view of the full layer dependence; layer 0 is included but is the input embedding.

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

**Fig. E2 — flush vs layer, Qwen3-4B/Phi-4-mini.** (a) $t^*_\ell$ vs layer at $W=100$; (b) three windows; (c) $(\ell,W)$ heatmap. **Data:** `window4d/*/w*/a1.npz`, 11 layers, 6 directions, $r=0.1$, 300 tokens. **Read:** both models give the same arithmetic progression; the two models are identical to the token.

![Fig. E3](update_figures/pack_window_agreement_fit.png)

**Fig. E3 — token agreement and its fit.** (a) Fraction of tokens identical to $W=4096$ for each window, with a stretched-exponential fit $f_\infty+(1-f_\infty)e^{-(t/\tau)^\beta}$ (dashed). (b) $1-\mathrm{agreement}$ on a log axis. **Data:** `results/window_tokens/tokens.npz`, 8 windows, 300 tokens, no perturbation. **Read:** the decay is a text statistic, not a clean exponential; the fitted $\tau$ grows with $W$ (W=100: $\tau\approx90$, $\beta\approx2$). **Trivial/not:** the first-difference law is a mask identity; the effective memory ($W\approx5$–$8P$) is a measured property of this continuation.

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

**Fig. F3b — live-gate density across models.** Density $P(m_t<T)$ vs $T$, sorted by $T$ (the lowest-temperature point is no longer connected to the highest). **Data:** `results/softloop*`, 300–600 steps, 8–32 directions. **Read:** density rises with $T$ at different rates. **Why the old panel (b) is deleted:** horizon vs density does not collapse across models; the same density gives a 10× different horizon, and a two-parameter fit gives $R^2=0.20$. Density alone is not the order parameter.

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

**Fig. F9 — soft vs sampled feedback.** Response for soft feedback (red), common-random-number sampling (blue dashed) and independent sampling (green dashed) at $T=1$, five models. **Data:** `results/softloop_sample/*/{crn,ind}_T1.0`. **Read:** soft grows; CRN is flat at 1; independent separates at step 0. **Conclusion:** the instability is deterministic mean feedback, not sampled generation. **Qwen3-4B is absent** because no sampled-feedback run was saved for it; the empty panel is removed rather than shown blank.

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
| 5 | dense 14B runs | **queued** on Spark (`queue_spark_wait.sh`), waits for 66 GB free; Qwen3-14B first attempt hit OOM because colleagues took the memory |
| 6 | DeepSeek dense window sweep, 106 windows | **done** (from existing `wspec`) |
| 7 | nonnorm controls | **randQwen done** (`nonnorm_randqwen3`); Pythia path mismatch; Mamba not run |
| 8 | depth probe with 32 directions | **done** for DeepSeek; Qwen2.5/Phi-4-mini **running** (`queue_extra.sh`) |
| 9 | sampled feedback for Qwen3-4B (CRN/ind) | **running** (`queue_extra.sh`), fills the missing F9 panel |
| 10 | exact $\Sigma_p$ validation (all vocab) | not run |
| 11 | attention mass, correct window mask | **done**, DeepSeek W=50/100/200/4096 + Qwen2.5/Phi/rand W=100 |
| 12 | closed-loop metric suite (M, FSLE, finite-time, spectrum, staircase, endpoint) | **done**, F2–F18 |
| 13 | log-logistic fit check (linear vs log space, KS) | **done**, A1b |
| 14 | variance/percentile pass on all aggregates | **done** where data allow; controls added to B5b/C1/C2/C3/D1b/D3b |

**Running on the GPUs.** On x11: depth $k=32$ for Qwen2.5-3B and Phi-4-mini, then CRN/independent sampling for Qwen3-4B (`scripts/queue_extra.sh`). On Spark: a memory-gated queue for dense 14B $r_*$ (`scripts/queue_spark_wait.sh`); it waits for 66 GB free and runs Qwen3-14B, Phi-4, DeepSeek-14B one at a time. I did not kill the colleagues' jobs. The x11 jobs finish in ~20–30 min; the Spark one may wait for a long window. **Recommendation:** the highest-value remaining measurement is the **dense 14B $r_*$**; fp32 for Qwen3-4B/Phi-mini is optional because your own analyses use fp16 for flips and shapes.

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

