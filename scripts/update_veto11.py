p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\veto.md"
s = open(p, encoding="utf-8").read()
add = r"""

---

## 11. Update after the full grid (5 models × 8 prompts × 192 directions)

This section supersedes the earlier verdicts where the data changed.

### 11.1 Margin distribution across windows — fitted

![margin fits](figures/veto/margin_fits.png)

**What is plotted.** (a) The survival function of the top-two logit gap $P(m>x)$ along the **same**
token sequence for five windows ($W=100\ldots1000$), with a fit per window over $m\in[0.02,10]$.
(b) The fitted shape parameters.

**Fit.** Stretched exponential $P(m>x)=c+a\,e^{-(x/\tau)^\beta}$ in four of five windows
($R^2=0.997$–$0.999$), with

| $W$ | $\tau$ | $\beta$ | $R^2$ |
|---|---|---|---|
| 100 | 1.91 | 1.10 | 0.997 |
| 150 | — (exponential, $\beta=1$) | — | 0.999 |
| 200 | 2.10 | 0.96 | 0.997 |
| 300 | 2.32 | 0.78 | 0.999 |
| 1000 | 2.20 | 0.77 | 0.999 |

**Verdict: the fit is real and it confirms the veto.** The margin distribution is a stretched
exponential with scale ≈2 logits and exponent ≈0.8–1.1, and the parameters are the same for all
windows. So the differences in the original `stablechaos_windows` (a) came from the different texts,
not from the window. This is now a quantitative statement, not an assertion.

### 11.2 The post-clip $1/W$ law does **not** reproduce

![regime](figures/veto/regime_scaling_deepseek.png)

**What is plotted.** The contraction rate after the window first clips, against $W$, on the DeepSeek
window sweep (same prompt, $W=50,100,200$), and the product $(-\lambda_{\rm post})W$.

**Numbers.** $\lambda_{\rm post}$ = −0.0105 ($W=50$), **−0.00216 ($W=100$)**, −0.00258 ($W=200$).
A $1/W$ law fitted to the $W=50$ and $W=200$ points predicts 0.0053 at $W=100$; the measurement is
2.4× smaller.

**Verdict: veto.** The clean $A/W$ line in the earlier figure was obtained with a different definition of
the post-clip interval (and a bootstrapped median); with the consistent estimator used everywhere else it
does not hold. It is not a law I would put in the paper without re-deriving the analysis. The *qualitative*
statement — that the decay after the clip weakens as $W$ grows — is supported; the $1/W$ form is not.

### 11.3 Flip probability, five models (now in `master.md` §8)

![flip fraction models](figures/fits/flip_fraction_models.png)

8 prompts × 192 directions × 4 radii per model, tie-corrected, with 95% CIs. The corrected fractions at
$r=0.1$ are **0.2% (DeepSeek), 3.6% (Qwen3.5), 16.8% (Qwen3-4B), 30.4% (Qwen2.5), 56.0% (Phi)**. The
ordering equals the amplification ordering, and the shapes differ (DeepSeek rises; Phi saturates).

**Verdict: include.** This is the strongest quantitative model comparison we have, and it replaces the
single-prompt, tie-contaminated version.

### 11.4 Superposition, three panels (now in `master.md` §7.2)

![superposition](figures/transport/superposition_3.png)

**(a)** Predicted vs measured amplitude lie on the identity line; **(b)** the ratio is 1 with a ±5–10%
spread; **(c)** the relative error falls with amplitude and is large only at the float32 floor.

**Verdict: include.** It is the validation of the kernel decomposition, and with the right axes it is
unambiguous.

### 11.5 Window token agreement, four panels (now in `master.md` §9)

![window tokens](figures/veto/window_token_agreement_multi.png)

First difference at $W-P$ (+2–7 tokens), surviving fraction $1-\exp(-(W/237)^{2.0})$ ($R^2=0.999$), text
bit-identical for 300 tokens once $W\gtrsim500$.

**Verdict: include (upgraded).** The earlier single-curve figure was a one-line remark; with eight
windows and a fit it is a result about the effective memory of the continuation.

### 11.6 Layer probe, temporal view (now in `master.md` §8)

![layer temporal](figures/fits/layer_probe_temporal.png)

At the perturbed position all models are ×1–3; 20 tokens later DeepSeek is ×0.05–0.7 while Qwen3-4B and
Phi are ×1–9. **Verdict: include** — it is the panel that shows the model difference is in transport.

### 11.7 Bottom line (updated)

| item | verdict |
|---|---|
| margin distribution vs window | **veto the original panel**; the fitted version belongs in the methods |
| post-clip $1/W$ law | **veto** — does not reproduce with the consistent estimator |
| flip probability (5 models) | **include** (master §8) |
| superposition (3 panels) | **include** (master §7.2) |
| window token agreement (4 panels) | **include** (master §9) |
| layer probe temporal | **include** (master §8) |
| layer probe depth-independence | one sentence |
| $r_*$ | ordering only |
| latent latency | include; decay fit veto |
| depth 6 prompts | veto |
"""
s = s.rstrip() + add
open(p, "w", encoding="utf-8").write(s)
print("veto updated")
