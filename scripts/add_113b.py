p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
anchor = "## 11.4 Survival of flips"
new = r"""## 11.3b Flip probability with 512 directions — and a warning about ties

![flip fraction](figures/fits/flip_fraction_512.png)

**Data.** DeepSeek, one prompt, **512 directions**, four radii, 300 tokens. Wilson 95% confidence
intervals (the error bars in panel a).

| radius $r$ | flips / 512 | fraction [95% CI] | flips after token 20 |
|---|---|---|---|
| 0.1 | 18 | 3.5% [2.2, 5.5] | **0.4%** |
| 0.3 | 127 | 24.8% [21.3, 28.7] | **3.5%** |
| 0.6 | 199 | 38.9% [34.7, 43.2] | **8.6%** |
| 1.0 | 311 | 60.7% [56.4, 64.9] | **9.0%** |

**Panel (a).** The flip fraction rises smoothly with $r$ and saturates near 60% at $r=1$.
**Panel (b).** Survival curves. **Panel (c).** Distribution of first-flip times.

**The warning.** 85–90% of all flips happen at **one step**, $t=2$, where this prompt has a margin of
exactly zero: the top-1 and top-2 logits are equal, so *any* perturbation flips the token. That is a
quantisation tie, not a sensitivity. Excluding it (last column) the flip fraction at $r=0.1$ is
**0.4%**, not 3.5%.

**Consequences, stated as rules for the rest of the document:**
1. Any flip statistic must exclude steps whose margin is below the resolvable gap (0.02 here; the
   fp16 models have a ~0.01 logit resolution).
2. The earlier "flip fraction 25% at $r=0.05$" for the fp16 models is inflated by the same effect and
   should not be compared with DeepSeek.
3. What remains is still a real sensitivity curve: outside the tie, the fraction that flips grows
   from 0.4% at $r=0.1$ to 9% at $r=1$ — small, smooth, and consistent with the stable-chaos picture
   (no threshold avalanche).

""" + anchor
assert anchor in s
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
