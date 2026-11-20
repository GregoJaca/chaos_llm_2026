p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()

old_start = "**What we should report instead.** The per-block singular values, which localise the gain:"
i0 = s.index(old_start)
i1 = s.index("## 9.3 The spectrum is flat:", i0)

new = r"""**What we should report instead.** The per-block singular value, measured in the **full** dimension
by power iteration (not in the tracked subspace):

![block sigma full](figures/fits/block_sigma_full.png)

**Panel (a).** $\sigma_{\max}$ of **one block's** Jacobian at each depth, full $d$ dimensions, median
over 4 tokens. **Panel (b).** $\sigma_1$ of the *product*, from the depth map. **Panel (c).** What the
product would be if each block's amplification composed: $\prod_\ell\sigma_{\max}(J_\ell)$.

**The numbers.**

| model | $\sigma_{\max}$ block 0 | middle blocks | last block | product $\sigma_1$ | naive product |
|---|---|---|---|---|---|
| DeepSeek-1.5B | 119 | 3.6 | 7.9 | ~20 | $10^{17.5}$ |
| Qwen3-4B | 59 | 4.1 | 15.4 | ~5 | $10^{24.2}$ |
| Phi-4-mini | 9.3 | 4.9 | 6.4 | ~9 | $10^{23.3}$ |

**The result, in one sentence.** Every block amplifies some direction by 3–6×, so if those directions
composed the perturbation would grow by 17–24 orders of magnitude; the measured product grows by
about one order. **The gains do not compose, because the direction of maximum amplification rotates
from block to block** — the same effect measured earlier with the five-layer non-normality budget
(83.6° average rotation between consecutive top subspaces, 13 orders of cancellation).

This is the answer to "why doesn't the network explode": each block has a strongly amplifying
direction, but it is a *different* direction at every block, so a generic perturbation only ever sees
the small common gain. It is also why the tracked-subspace spectrum looked flat: a random 8- or
64-dimensional plane almost never contains the one direction a block amplifies most.

**Correction to what this section said before.** An earlier version claimed "the first block amplifies
and every other block is within a few percent of the identity". That was an artefact of the 8-dim
subspace: within it the blocks *are* near-identity (Section 9.3), but the full Jacobian is not. The
correct statement is the one above.

"""

s = s[:i0] + new + s[i1:]
open(p, "w", encoding="utf-8").write(s)
print("replaced")
