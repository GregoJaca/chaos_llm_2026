p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()

new = r"""

---

# Part 8. Geometry and dimensionality, all four models

**Data.** For each model: the input embedding matrix $E$ ($V\times D$), the unembedding $W_U$, and
the hidden states at every layer for 3 prompts × 400 generated tokens (1200 states), plus 128
positions of logits. All computed on CPU from Grams and state dumps produced on the GPU
(`results/dim/`, report and 103 figures in `results/subagent_dim/`).

## 8.1 Vocabulary geometry

![vocab spectra](figures/dim/fig_A1_spectra_grid.png)

**Singular spectra of $E$ (and $W_U$ when untied), log–log.** One dominant direction, then a smooth
high-dimensional bulk with **no sharp rank**.

| model | PR | stable rank | eff-rank 90% | row cosine | cone $\|\mu\|/\overline{\|e\|}$ |
|---|---|---|---|---|---|
| DeepSeek (E) | 13.3 | 4 | 1172 | +0.27 | 0.51 |
| DeepSeek (W_U) | 9.3 | 4 | 1140 | — | — |
| Qwen3-4B | 136.1 | 12 | 2083 | +0.08 | 0.28 |
| Phi-4-mini | 4.1 | 2 | 1021 | +0.17 | 0.54 |
| Qwen3.5-2B | 22.3 | 5 | 1622 | +0.19 | 0.44 |

**Reading.** All vocabularies are a cone: one shared direction (the mean, $s_1^2\approx V\|\mu\|^2$)
plus a genuinely high-dimensional bulk needing 1000–2000 dimensions for 90% of the variance. Qwen3-4B
is the near-isotropic outlier (row cosine 0.08); Phi-4-mini is the most anisotropic (half of the
variance in one direction). DeepSeek is untied and its two spectra differ slightly; the $E$–$W_U$
principal angles cannot be computed from separate Grams (we did not save the cross-Gram).

## 8.2 The state lives on a ~10-dimensional manifold inside 1536–3072 dimensions

![per prompt ID](figures/dim/fig_B6e_perprompt_id_all.png)

**Intrinsic dimension of a single generation** (400 tokens, three estimators, bands over 3 prompts).

**Reading.** A single trajectory has ID **≈3 at block 1, rising to 8–12** at the last block and
plateauing; the three estimators agree on the shape but not the level (factor 2–4, the honest
uncertainty). The states occupy a 37–49-dimensional ellipsoid (participation ratio) and a
196–395-dimensional affine hull (eff-rank 90%) inside $D=1536$–$3072$. Local (per-token) dimension is
**not constant**: its distribution is wide (median 7, IQR 4–12 for DeepSeek) and it moves along the
trajectory.

![measures all](figures/dim/fig_B6_measures_all.png)

**Depth profiles on common axes.** Anisotropy (mean pairwise cosine) and the cone ratio rise with
depth in DeepSeek and Qwen3-4B and fall in Phi and Qwen3.5. Qwen3-4B has a mid-stack collapse of the
centred participation ratio (1.4–2.5 at blocks 17–23): one giant direction dominates there.

## 8.3 The unembedding collapses the state by 2–45×; Phi's decisions are one-dimensional

![logit rank](figures/dim/fig_C1b_rank_bars.png)

**Effective rank (90%) of the 128×V logit matrix**, against the post-norm state at the same
positions.

| model | logit eff-rank (of 128) | state eff-rank | collapse | centred-logit rank 80% |
|---|---|---|---|---|
| DeepSeek | 24 | 41 | 0.59 | 33 |
| Qwen3-4B | 10 | 54 | 0.19 | 45 |
| Phi-4-mini | **1** | 61 | **0.02** | 3 |
| Qwen3.5-2B | 36 | 65 | 0.55 | 56 |

**Reading.** Over these positions Phi-4-mini's logits are **80% one direction**: the decision space
is effectively one-dimensional, which is exactly what a model with a ×23 kick amplification and
frequent flips should look like. The unembedding is a strong dimensionality bottleneck in every
model, but by a factor that varies 2–45×.

## 8.4 Does geometry order the dynamics? (n=4: an ordering, not evidence)

![spearman](figures/dim/fig_D1c_spearman_bars.png)

**Spearman correlation between each geometric measure and the single-kick gain, across the four
models.** With four points only $\rho=\pm1$ is significant, so this is a hypothesis, not a result.

- **Anisotropy orders the kick gain exactly**: late-layer mean cosine 0.24 (Qwen3.5) < 0.41
  (DeepSeek) < 0.48 (Qwen3-4B) < 0.53 (Phi) ↔ kick gain 0.47 < 0.66 < 5.9 < 23.0, $\rho=+1.00$.
- The spectrum slope anti-orders it ($\rho=-0.80$): flatter spectrum ↔ bigger amplification.
- **Intrinsic dimension does not** ($\rho=+0.40$): Phi has both the highest ID and the highest gain,
  but DeepSeek and Qwen3.5 swap.
- Nothing orders the late rate $\lambda$ or the direction spread.
- Within DeepSeek the same pairing holds layer by layer (cos vs gain Pearson +0.96, n=27), but the
  sign flips in Qwen3.5 (cos falls while gain rises) — the within-model correlations ride the shared
  depth trend and are not independent evidence.

**The falsifiable version:** add ≥6 models; if the anisotropy↔gain Spearman does not stay
$\ge+0.6$, the four-model match is a coincidence. That is cheap and is Proposal P2 in Part 7.

## 8.5 The response is 99.4% tangential and lives outside the state subspace

![radial](figures/dim/fig_D3_radial_c2_deepseek.png)

**Decoding the stored $\cos(h+\delta h,h)$ with $1-\cos\approx(1-c^2)\rho^2/2$**, where
$c=\cos(\delta h,h)$ and $\rho=\|\delta h\|/\|h\|$ (DeepSeek, the only model with enough precision).

**Reading.** $c^2=0.006$ median: **0.6% of the response energy is radial** (8% of the amplitude), at
every depth and at every radius from $10^{-3}$ to $2$. So $\|\delta h\|$ *is* essentially the
perpendicular component — the answer to "are we tracking the relevant part" is yes, but only because
the radial part is negligible, not because the norm isolates it. It also means the response
**rotates** the state rather than rescaling it, which is why the argmax survives small perturbations
and why flips require tangential motion across the boundary.

![overlap](figures/dim/fig_D3e_subspace_overlap.png)

**Principal-angle overlap between the response subspace and the state subspace** (top-32 PCs, mean
squared cosines).

**Reading.** $\delta h$ overlaps the state's PCs at 0.046–0.071, i.e. 3–7× chance (0.01–0.02), but
**≈2× below the prompt-vs-prompt control** (0.11–0.23). So the perturbation does not simply excite
the directions that ordinary generation occupies: it is a low-rank field (eff-rank 90% = 81–104 of
$D$, except Qwen3-4B at 629) that is partially, but not mostly, aligned with the state.

## 8.6 Negative results and traps found in this analysis

- **The per-layer write is not lower-rank than the state.** eff-rank(write)/eff-rank(state) =
  1.2–1.6 in all four models. There is no "low-rank update" story to be had from this data.
- **Pooling prompts breaks intrinsic-dimension estimators.** Pooled MLE gives ~2 for DeepSeek while
  each trajectory alone gives ~8: three thin clusters read as one low-dimensional blob. Use
  per-trajectory values.
- **Forgetting to remove the mean makes the states look rank-1–5** (participation ratio ~10× too
  small). Most "LLMs are 2-dimensional" claims are this artifact.
- **The top saved state is the final RMSNorm, not the last block.** The apparent norm collapse and
  the anti-aligned final "write" are normalization; any "final block" claim from this data is wrong.
- **The stored `cos` key is $\cos(h+\delta h,h)$, not $\cos(\delta h,h)$** — it is ≈1 everywhere and
  grid-saturated in the fp16 models. Section 8.5 decodes it; do not use it naively.
- Qwen3-4B's mid-stack PR dip may be a float16 quantization artifact; needs float32 states.
- The window-sweep gain comparison across $W$ is baseline-shifted (the margins themselves differ
  with the window), so only within-$W$ profiles are clean.

## 8.7 What this changes in the main story

1. Q5 is now answered with a number: the radial part is 0.6% of the energy, so the norm is a good
   proxy for the dynamically relevant component.
2. The model differences are not just in the dynamics: the decision space of Phi is 1-D, its
   vocabulary cone is the narrowest, and its amplification is the largest. Whether that is causation
   is the open question (P2).
3. The state is a thin manifold (ID ≈10) inside a large ambient space, and the response is an even
   thinner field that partly avoids that manifold. Any theory of "where the perturbation goes" has to
   explain both numbers.
"""

s = s.rstrip() + new
open(p, "w", encoding="utf-8").write(s)
print("appended", len(new))
