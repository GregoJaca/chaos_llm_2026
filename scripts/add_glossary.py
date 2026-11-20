p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()

gloss = r"""
---

# Part 0. Glossary — every symbol and every statistic used below

Read this once; the rest of the document uses these words without re-defining them.

## 0.1 The model and the experiment

| term | meaning |
|---|---|
| **residual stream** | the vector $h_{t,\ell}\in\mathbb{R}^d$ carried from block to block at position $t$; $\ell$ indexes the block (layer), $\ell=0$ is the embedding output, $\ell=L$ is after the last block. |
| **block / layer** | one transformer block: attention, then an MLP, each added to the residual stream. "Block $\ell$" = the map from $h_{t,\ell}$ to $h_{t,\ell+1}$. |
| **position** | the index of a token in the sequence. Prompt positions are $0\dots P-1$; generated positions are $P\dots P+T-1$; a generated position is also written $t=0,1,\dots$ counted from the first generated token. |
| **teacher forcing** | run the model on a *fixed* token sequence (the baseline greedy output), so that a perturbation cannot change which token comes next. This isolates the continuous dynamics. |
| **free run** | generate tokens greedily; if the perturbation changes a token, the continuation diverges. This is the real generation. |
| **greedy decoding** | at each step take the token with the largest logit. No temperature, no sampling, no repetition penalty. |
| **logits** | the $V$-dimensional output before the softmax, $z=W_U h$; $V$ is the vocabulary size. |
| **margin** $m_t$ | the gap between the largest and second-largest logit at step $t$: $m_t=z_{(1)}-z_{(2)}$. Small margin = close call. |
| **top-1 / top-2** | the highest and second-highest scoring tokens at a step. |
| **flip** | a step where the perturbed run's greedy token differs from the baseline's. The first flip is the first such step. |
| **perturbation radius** $r$ | the Euclidean norm of the vector added to the embeddings. All perturbations are additive noise of norm $r$; "linear regime" means the response is proportional to $r$. |
| **direction** $u$ | a unit vector; the perturbation is $r\,u$. Different $u$ = different directions in the same $d$-dimensional space. |

## 0.2 The quantities we plot

| symbol | definition | what it is |
|---|---|---|
| $\delta h_{t,\ell}$ | $h'_{t,\ell}-h_{t,\ell}$ | the difference between perturbed and baseline hidden state |
| $\|\delta h_{t,\ell}\|$ | its Euclidean norm | absolute size of the perturbation |
| $\|\delta h_{t,\ell}\|/\|h_{t,\ell}\|$ | ratio to the state's own norm | **relative** size; cancels the residual stream's growth |
| $g_\ell$ | $\|\delta h_{t,\ell}\|/\|\delta h_{t,\ell-1}\|$ | layer-to-layer gain at a fixed $t$ |
| $\lambda_k(t)$ | $\frac{1}{2k}\log\frac{\|\delta h_{t+k}\|}{\|\delta h_{t-k}\|}$ | finite-time stretching rate over a lag $k$, in nats per token |
| $\Lambda(t)$ | $\frac1t\log\frac{\|\delta h_t\|}{\|\delta h_0\|}$ | cumulative average rate from the start |
| $u_t$ | $\delta m_t/m_t$ | the margin coordinate: how far the margin moved, in units of itself; a flip needs $u_t\le-1$ |
| $r_c(t,u)$ | $m_t\,\big/\,\big(|\delta m_t(u)|/r\big)$ | the radius at which the margin at step $t$ would reach zero for direction $u$ |
| $r_*(u)$ | $\min_t r_c(t,u)$ | the smallest radius that flips *any* token of the continuation for that direction |
| $\sigma_i(\ell)$ | $i$-th singular value of the block product $P_\ell$ | the factor by which the tangent map can stretch the $i$-th direction; $\sigma_1$ = largest |
| $\lambda_i(\ell)$ | $\log\sigma_i(\ell)/(\ell+1)$ | log stretch per block; the finite-time Lyapunov exponent of the $i$-th direction |
| $J_\ell$ | $\partial h_{\ell+1}/\partial h_\ell$ | the block Jacobian at a fixed position |
| $P_\ell$ | $J_\ell J_{\ell-1}\cdots J_0$ | the product of block Jacobians through $\ell+1$ blocks |

## 0.3 Statistics

| term | meaning |
|---|---|
| **median** | the 50th percentile: half the values are below. Robust to outliers; used for almost every curve. |
| **mean** | the arithmetic average. Sensitive to outliers; used only when stated. |
| **IQR** | interquartile range, 25th to 75th percentile: the band that contains the middle half of the values. Plotted as a shaded band. |
| **10–90% band** | the range from the 10th to the 90th percentile: contains 80% of the values. Used when the distribution is wide. |
| **quantile** | the value below which a given fraction of the data lies; "5% quantile" = 5% of values are smaller. |
| **ECDF** | empirical cumulative distribution function: for a value $x$, the fraction of observations $\le x$. It starts at 0 and ends at 1, and it is a staircase for finite data. |
| **CCDF** | complementary cumulative distribution function: $1-\text{ECDF}(x)$ = the fraction of observations $>x$. Plotted on log–log axes when the tail is of interest. |
| **histogram** | counts of observations in bins. |
| **bootstrap CI** | a confidence interval obtained by resampling the data with replacement many times (here 2000) and taking the 2.5–97.5% range of the statistic. A 95% CI that excludes zero is the usual significance criterion. |
| **standard error (SE)** | the uncertainty of a fitted parameter, from the fit's covariance matrix. Quoted as "value ± SE". |
| **$R^2$** | fraction of the variance of $y$ explained by the fit: 1 = perfect, 0 = no better than the mean. |
| **BIC** | Bayesian information criterion: $n\ln(\text{RSS}/n)+k\ln n$, where RSS is the residual sum of squares, $n$ the number of points, $k$ the number of parameters. **Lower is better**; it penalises extra parameters, so it can compare models with different numbers of parameters. A difference $\Delta$BIC $<2$ is not evidence for either model. |
| **censored** | a measurement cut off by the horizon: e.g. "the perturbation never doubled within 300 tokens". Censored points are excluded from a rate and counted separately. |
| **survival function** $S(t)$ | fraction of cases still "alive" (not yet flipped) at time $t$. Starts at 1 and decreases. |
| **hazard** | the per-step probability of flipping given that it has not flipped yet. |
| **log-logistic** | the distribution with CCDF $1/(1+(x/k)^b)$; $k$ is its median and $b$ the tail exponent. Used as a candidate description of $r_*$. |
| **power law** | $y=a x^b$; on log–log axes it is a straight line of slope $b$. |
| **exponential** | $y=a e^{bx}$; on semilog axes (log $y$, linear $x$) it is a straight line. |
| **stretched exponential** | $y=a e^{-(x/\tau)^\beta}$; $\beta<1$ decays slower than an exponential. |
| **nats** | the natural unit of a logarithm: a rate of 1 nat/token means the quantity is multiplied by $e\approx2.718$ per token. |

## 0.4 The window

| term | meaning |
|---|---|
| **sliding window** $W$ | each position may attend only to the previous $W$ positions. It is a property of the attention mask, not of the weights. |
| **flush** | the step at which a perturbation becomes *exactly* zero at a layer because the window has removed every perturbed position from its receptive field. |
| **first departure** | the first step at which a run with window $W$ differs from a run with no window: it happens at $t=W-P$, when the first prompt token leaves the window. |

"""

anchor = "---\n\n# Part 1. Definitions"
assert anchor in s
s = s.replace(anchor, gloss + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("glossary inserted")
