p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()

new = r"""

---

# Part 9. Functional forms: what each curve actually is

## 9.1 The tool

`src/chaos_llm/curvefit.py`, driven by `scripts/fit_curves.py`. Given $(x,y)$ it fits 22 families
and ranks them by BIC:

| family | form |
|---|---|
| polynomials | $a+bx$, $a+bx+cx^2$, $a+bx+cx^2+dx^3$ |
| logarithmic | $a+b\ln x$, $a+b\ln x+cx$ |
| inverse / rational | $a+b/x$, $a/(1+bx)$ |
| power laws | $a x^b$, $c+a x^b$, $c+a(x-x_0)^b$ |
| exponentials | $a e^{bx}$, $c+a e^{bx}$, $c+a e^{b(x-x_0)}$ |
| saturating | $c+a(1-e^{-x/\tau})$, $c+a(1-e^{-(x/\tau)^\beta})$ |
| stretched decay | $c+a e^{-(x/\tau)^\beta}$ |
| gaussian | $c+a e^{-(x/\tau)^2}$ |
| Hill / log-logistic | $c+a x^b/(k^b+x^b)$, $c+a/(1+(x/k)^b)$ |
| two-phase | $c+a(1-e^{-x/t_1})e^{-x/t_2}$, $c+a_1e^{-x/t_1}+a_2e^{x/t_2}$ |

**Rules, and why.**

1. **Log-domain fitting for pure exponential and power laws.** If $y$ spans decades, fitting
   $y=a e^{bx}$ by least squares lets the largest values decide everything: the error is absolute,
   but what you care about is relative. The tool fits $\ln y$ against $x$ (or $\ln x$), which
   minimises squared *relative* error. Offset models ($c+a e^{bx}$) cannot be log-transformed, so
   they are fitted by nonlinear least squares in the native domain. Both are scored in the domain
   they were fitted in, so the comparison is fair.
2. **Shifts are explicit.** A form with $x_0$ is a separate model with one more parameter; it wins
   only if the BIC gain pays for it. This matters below: the depth law needs $x_0=-1$, and without
   it the fitted exponent changes from $-1.05$ to $-0.60$.
3. **BIC**, not $R^2$, decides. $R^2$ always increases with parameters; BIC does not. A model within
   $\Delta$BIC $<2$ of the best is a tie and is reported as such.
4. **Standard errors** come from the Gauss-Newton covariance; every parameter is quoted with one.
5. Sign-degenerate fits (a rise written as a decay with $a<0$) are removed by comparing predictions.

Per curve the driver writes one PNG (data, best fit, second best, residual-free) and one JSON, plus
`results/fits/SUMMARY.md` with all of them.

## 9.2 The depth exponent: not an exponential, a shifted $1/\ell$ law

![depth four axes](figures/fits/depth_law_4axes.png)

**Data.** `results/depth_w100/depth.npz`. For each of 8 generated tokens, the tangent is propagated
through all 28 blocks at that position; $\sigma_i(\ell)$ is the $i$-th singular value of the product
of the first $\ell+1$ block Jacobians, and the *finite-time depth exponent* is
$$\lambda_i(\ell)=\frac{\log\sigma_i(\ell)}{\ell+1}.$$
The plotted curve is $\lambda_1(\ell)$ median over the 8 tokens; the band is the IQR.

**What the four panels show.** The same data look exponential in (b) and (c) — a straight line is not
obtained in any of them, and the curvature changes sign between (b) and (c). Only (a) and (d) with
the fitted curve are informative.

**Fit (BIC winner).**
$$\lambda_1(\ell)=0.022+3.03\,(\ell+1.07)^{-1.05},\qquad R^2=0.99985.$$
The runner-up (Hill/log-logistic) is $\Delta$BIC $=2.0$, i.e. a tie; all exponential, stretched and
Gaussian forms are $>100$ BIC worse. **So "it looks like an exponential decay" is wrong**: the curve
is a power law with exponent $-1$ and a shift of one layer.

**Why $-1$, and what is actually decaying.** $\lambda_1$ is defined with a $1/(\ell+1)$
normalisation. If you undo it,
$$\log\sigma_1(\ell)=\lambda_1(\ell)\,(\ell+1)=3.00+0.019\,\ell,$$
i.e. the cumulative top singular value is *flat to within 2% per layer*: $\sigma_1\approx20$ at every
depth, growing by $1.9\%$ per block. The "decay with depth" is therefore **not** a decay of the
tangent gain — the gain per block is constant and slightly above 1 — it is the $1/(\ell+1)$ of the
definition. This is the cleanest example in the project of a curve whose shape is kinematics, not
dynamics.

![depth logsigma](figures/fits/depth_logsigma.png)

**Panel (a).** $\log\sigma_1(\ell)$ with a linear fit: slope $+0.0190\pm0.0026$ nats/layer.
**Panel (b).** The same slope fitted per token: it clusters at 0 (mean $+0.02$), so the near-constancy
is not an averaging artefact.

![depth exponent per token](figures/fits/depth_exponent_per_token.png)

**The shift diagnosis.** Fitting $c+a\ell^b$ *without* the $+1$ gives $b=-0.60$ for all eight tokens;
with the shift it is $b=-1.05$. Both fit well over a limited range, which is exactly why the tool
tests shifts explicitly. The physically meaningful exponent is $-1$.

## 9.3 The same treatment for the layer-resolved response

![depth gain fit](figures/fits/fit_depthgain_DeepSeek-1.5B_t50__DeepSeek-1.5B.png)

**Data.** $\|\delta h_{t,\ell}\|/r$ vs layer (prompt perturbation, $r=0.1$, $t=50$).
**Fit.** A saturating/stretched rise wins; exponential and pure power laws lose. The absolute
response is an **integral** of the per-layer gain, which is why it rises monotonically while the
per-layer gain is constant (Part 3b, Q6): the growth is the state norm, not extra amplification.

The state norm itself (below) is one of the few curves that really is exponential in depth:

![h profile fit](figures/fits/fit_hprofile_Qwen3-4B_t50__Qwen3-4B.png)

$\|h_{t,\ell}\|$ vs layer, Qwen3-4B, fitted $c+a e^{bx}$ (log domain): the residual stream grows
geometrically with depth, by a factor that is model-specific.

## 9.4 The full table

`results/fits/SUMMARY.md` contains every curve fitted (≈60). Selected rows:

| curve | model | best form | key parameter | $R^2$ |
|---|---|---|---|---|
| $\lambda_1(\ell)$ | DeepSeek | $c+a(\ell-x_0)^b$ | $b=-1.05$, $x_0=-1.07$ | 0.9998 |
| $\log\sigma_1(\ell)$ | DeepSeek | linear | $+0.019$/layer | 0.996 |
| $\|h_\ell\|$ at $t=50$ | Qwen3-4B | $a e^{b\ell}$ | $b=+0.10$/layer | 0.996 |
| $\|h_\ell\|$ at $t=50$ | Phi-4-mini | $a e^{b\ell}$ | $b=+0.05$/layer | 0.992 |
| $P(r_*>x)$ | DeepSeek | log-logistic | $k=0.93$, $b=1.67$ | 0.9998 |
| $\lambda_{10}(t)$ | DeepSeek | two-phase | $R^2$ only 0.40 | 0.40 |
| $\|\delta h_t\|$ full curve | DeepSeek | (piecewise only) | — | — |

Two rows are negative results worth stating: **no single smooth form describes $\lambda_{10}(t)$**
(the best is a two-phase rise/decay with $R^2=0.40$), and **no single smooth form describes the full
response $\|\delta h_t\|$** — the earlier dedicated analysis reached the same conclusion with
piecewise forms (504/700 curves needed a piecewise power in $\log t$).

---

# Part 10. The FSLE, explained

## 10.1 Definition and what the stored plot is

The finite-size Lyapunov exponent answers one question: *given a perturbation of size $\delta$, how
fast does it become $\varrho$ times larger?* Fix a growth factor $\varrho$ (1.5, 2, 4 in the figure),
measure the first time $\tau$ at which
$$\|\delta h_\tau\|\ge \varrho\,\|\delta h_0\|,$$
and define
$$\lambda_{\rm FSLE}(\varrho;\delta)=\frac{\log\varrho}{\tau(\varrho;\delta)}.$$
If the trajectory never reaches $\varrho\delta$ inside the horizon it is **censored** (it does not
grow). The stored `fsle_time.png` plots the median of $\lambda_{\rm FSLE}$ over directions against the
initial radius $r=\delta$, for three $\varrho$.

![fsle vs r](figures/fsle/fsle_vs_r_censoring.png)

## 10.2 Answering the reading of the plot

- **$\varrho=2$ is exactly flat, not "roughly constant"**: every point is $0.04332=\log 2/16$ for all
  17 radii, and the censoring is zero. The doubling time is 16 tokens at $r=10^{-6}$ and at
  $r=0.3$. This is a statement of **linearity**: if $\delta h_t=r\,g(t)$ then $\tau$ cannot depend on
  $r$.
- **$\varrho=4$ is flat only where it is defined.** Below $r\approx10^{-3}$ it is 100% censored (the
  response never quadruples), and the median over the survivors is biased. Where it is defined,
  $\tau=57$ and $\lambda=0.0243$ — the number you saw as "0.0025" is $0.0243$.
- **$\varrho=1.5$ varies with $r$ only at the two ends**: $0.027$ for $r\le2\times10^{-4}$, rising to
  $0.05$ for $r\ge10^{-3}$. Both ends are artefacts: at the small-$r$ end the $t=0$ value is at the
  float32 floor (next figure), at the large-$r$ end nonlinearity sets in. In the middle it is flat,
  as linearity requires.

![fsle floor](figures/fsle/fsle_norm_curves.png)

**The object underneath.** $\|\delta h_t\|/\|\delta h_0\|$ at the last layer, median over 20
directions, for four radii spanning 300×. The curves collapse (max relative deviation 8% between
$r=0.01$ and $0.1$; 35% for $r=10^{-3}$, which is the floor; 16% for $r=0.3$, the nonlinearity).

**And this is the important part:** the response is **not a growth law**. It rises to
$\times4.5$ at $t\approx75$, falls back through 1, and decays to $0.2$ at $t=300$. The FSLE is the
average of this *transient*, taken from $t=0$; it is not a Lyapunov exponent and it is not constant in
time.

![fsle tau lambda](figures/fsle/fsle_tau_and_lambda_vs_rho.png)

**$\tau(\varrho)$ and $\lambda_{\rm FSLE}(\varrho)$** at $r=0.1$: the doubling time is 16 tokens, the
quadrupling time 57, and the FSLE falls from 0.035 to 0.016 as $\varrho$ grows. There is no single
FSLE number; there is a function of the threshold.

![fsle reference](figures/fsle/fsle_reference_choice.png)

**The value depends on where you start the clock.** With the reference at $t_0=0$ the doubling time
is 16 tokens; starting at $t_0=20$ (after the first rise) it is much longer, and for
$r\gtrsim0.1$ the curve does not double again at all inside 300 tokens. Any quoted FSLE must state
$t_0$, $\varrho$ and the horizon. This is the same caveat as for $\lambda_k(t)$.

![fsle vs lambda10](figures/fsle/fsle_vs_lambda10.png)

**FSLE vs the finite-time rate.** The FSLE (blue, global average over the transient) sits at
$+0.043$; the local rate $\lambda_{10}(t)$ (red) is positive for $t\lesssim90$ and negative
afterwards, with mean $-0.0073$ over the horizon. The two numbers are not contradictory: they answer
different questions. The FSLE measures the first doubling of an initially tiny perturbation; the
local rate measures stretching at a given time. For "does the system amplify?" the local rate and the
flip statistics are the right objects; for "how long until this perturbation is twice as big?" the
FSLE is.

## 10.3 Verdict

The FSLE plot is **correct but not a discovery**: its flatness is the linear-response theorem, its
absolute value is the 16-token arrival transient of the last layer, and the $\varrho=4$ curve is a
censoring artefact below $r=10^{-3}$. It is a useful *consistency check* on linearity, and it is
reported as such. The growth/decay physics lives in $\lambda_{10}(t)$ and in the survival curves.

---

# Part 11. The flux-tube radius $r_*$

## 11.1 Definition

For a direction $u$ and a token $t$, the margin $m_t$ is the logit gap between the top two tokens. A
perturbation of radius $r$ changes it by $\delta m_t$, linearly in $r$. The radius at which that
token's decision would be decided the other way is
$$r_c(t,u)=\frac{m_t}{|\delta m_t(u)|/r},$$
and the **flux-tube radius** of the direction is the smallest such radius over the horizon,
$$r_*(u)=\min_{0\le t<300} r_c(t,u).$$
So $r_*$ is the smallest perturbation that changes *any* token of the continuation — one number per
direction, with units of the embedding space. It is the natural "distance to a decision" of the
model, and the distribution over directions is what `rstar_distribution.png` showed for 20
directions.

## 11.2 The distribution is a log-logistic law

![rstar 1536](figures/fsle/rstar_1536.png)

**Data.** The 1536-direction run (one prompt, $r=0.1$, 300 tokens). **Panel (a).** CCDF of $r_*$,
1536 directions. **Panel (b).** Histogram of $\log_{10}r_*$. **Panel (c).** $r_*$ against the margin
coordinate $|u|$ at the closest call: $r_*=1/|u|$ by construction, which is what the dashed line is.

**The fit.**
$$P(r_*>x)=\frac{1}{1+(x/0.93)^{1.67}},\qquad R^2=0.9998,$$
a **log-logistic** (Fisk) distribution: median $k=0.93$, tail exponent $b=1.67$. The two-parameter
form wins by $\Delta$BIC $=3648$ over the best alternative. Quantiles: 0.15 (5%), 0.47 (25%), 0.90
(median), 1.72 (75%), 4.40 (95%) — a factor 100 between the 5th and 95th percentile.

**What it means.** The flip radius is the inverse of the margin coordinate, so a heavy right tail in
$r_*$ is a heavy left tail in $|u|$: most directions are far from a boundary, but a small fraction is
very close. Concretely, on this prompt:
- 2.7% of directions flip for $r<0.1$,
- 13.7% for $r<0.3$ (the linear-regime limit),
- 54.6% for $r<1$.

**Caveat that must be stated.** $r_c$ extrapolates the *linear* margin response to any radius. For
$r_*>0.3$ that extrapolation leaves the measured linear regime, so the *tail of the distribution*
(and therefore the fitted exponent) is an extrapolation. What is measured is the region $r_*<0.3$,
where the local slope of the CCDF is already $\approx-1.7$; the log-logistic is the simplest form
that matches both the measured region and the censored tail.

## 11.3 Reading the original plot, and the 20-direction version

![rstar ccdf hist](figures/fsle/rstar_ccdf_hist.png)

The stored `rstar_distribution.png` is panels (a)+(b) of this figure with 20 directions: a CCDF and an
ECDF of $\log_{10}r_*$. With 20 points the CCDF is a staircase; the 1536-direction run (§11.2) is the
same object with the statistics needed to see the law.

![rstar per direction](figures/fsle/rstar_per_direction.png)

**Variation across models and directions.**

![rstar models](figures/fsle/rstar_models.png)

CCDFs for the four models (DeepSeek at $r=0.1$, Phi at $r=0.3$). The ordering is the same as the
single-kick amplification: Phi has the smallest radii (most fragile), DeepSeek the largest. Note the
caveat: the four curves are computed at different reference radii, which matters only through the
linear extrapolation.

![rstar local](figures/fsle/rstar_local_curves.png)

**Where the closest call happens.** $r_c(t,u)$ for the five most fragile directions as a function of
$t$: the minima are sharp and occur at a handful of tokens (the margin dips of Part 2.0), not spread
over the horizon. This is the flux-tube picture in one panel: a direction is safe except at a few
gates.

![rstar vs margin](figures/fsle/rstar_vs_margin.png)

**$r_*$ tracks the margin.** Correlation between $\log r_*$ and the log margin at the argmin:
$\rho=$CORR. Directions that flip easily are the ones whose closest call has a small margin — the
distribution of $r_*$ is, to a good approximation, the distribution of margins of the closest calls.


## 11.4 Survival of flips

![flip survival](figures/fsle/flip_survival.png)

Fraction of directions that have not flipped by token $t$, in the free runs at several radii
(step curves). The hazard is largest early and saturates: a direction either flips in the first
~100 tokens or not at all within the horizon. This is the empirical counterpart of $r_*$: the
directions with small $r_*$ are exactly the ones that leave the survival curve first.
"""

# fill the correlation number from the saved json
import json as _json
try:
    _n = _json.load(open(r"C:\Users\hjaca\research\chaos_llm_2026\results\fsle_rstar_numbers.json"))
    corr = _n.get("corr_rstar_margin", float("nan"))
    new = new.replace("CORR", f"{corr:.2f}")
except Exception:
    new = new.replace("CORR", "n/a")

s = s.rstrip() + new
open(p, "w", encoding="utf-8").write(s)
print("appended", len(new))
