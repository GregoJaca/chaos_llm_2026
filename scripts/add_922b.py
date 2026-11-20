p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
anchor = "## 9.3 The spectrum is flat: what \"isotropic\" means here"
new = r"""## 9.2b What the depth map does *not* measure (and the right $y$-axis)

The depth map perturbs the state **at one position** and follows that position **through the blocks**,
holding every other position fixed. Our actual experiment is different: the perturbation sits on the
*prompt* positions, and the response at a generated position is created by **attention pulling it in
from other positions**. Those are not the same object, and the numbers show the gap:

![sigma vs real](figures/fits/sigma_vs_real.png)

**Data.** Depth map: 128 tokens, 8 directions, DeepSeek. One-token kick: `probe2`, injection at block
0's output at position $s$, 6 directions. Real response: the prompt-perturbation run (`a1`), all 28
layers, at generated position $t=50$.

**Panel (a).** From layer 1 to layer 27,
- the same-position tangent chain grows by **×1.54** ($\sigma_1$ 17.3 → 26.7);
- the actual prompt response grows by **×53** (0.046 → 2.46 in units of $r$).

So the same-position chain accounts for $\log(1.54)/\log(53)\approx 11\%$ of the depth growth. The
other ~89% comes from cross-position transport, which the depth map by construction cannot see.

**Panel (b).** A kick at $s=16$ grows to ×3.1 at the **same** position (DeepSeek), ×2.8 (Phi), ×4.7
(Qwen3-4B) — all comparable. **Panel (c).** Twenty tokens later the models separate: the far-field
response decays to ×0.67 in DeepSeek, but **grows to ×9.2 in Phi and ×5.9 in Qwen3-4B**. The
model difference lives in the **far field**, not in the same-position chain.

**So, is $\sigma_1$ the right thing to look at?** No, for two independent reasons:

1. **$\sigma_1$ is a maximum over all directions, not the growth of our perturbation.** For a fixed
   initial vector $v_0$ the growth is $\|P v_0\|$, which is $\le\sigma_1$ and equal only if $v_0$ is
   the top right-singular direction. The measured same-position growth for a random direction is
   ×3.1 (panel b), against $\sigma_1=27$: a factor 9. Quoting $\sigma_1$ would overstate the real
   stretch by an order of magnitude.
2. **The map is restricted to one position.** Even a perfect $\sigma_1$ of the same-position product
   would describe a mechanism that carries ~11% of the real growth.

**And "if it is isotropic, composing is fine"?** It is not isotropic in the sense that would justify
composing. Two facts have to be held together:
- within the 8-dimensional tangent subspace we track, the blocks are close to the identity, so the
  top-8 singular values of the *product* come out nearly equal (Section 9.3);
- the **full** block Jacobian is strongly non-normal — the earlier measurement gave $\sigma_{\max}$
  of 95–195 at block 0, and the top singular direction **rotates by 83.6° on average between
  consecutive blocks**, so the product of the per-block $\sigma_{\max}$ overestimates the true chained
  growth by ~13 orders of magnitude (Section F, `nonnorm_gain_budget.png`).

Both are true because "near identity" applies to the tiny subspace we happen to track, while the
full map has one huge direction that changes from block to block. **The gains do not compose**, which
is exactly what you suspected.

**What to plot instead.** The measured response, in the normalisation that matches the question:

| question | quantity | where |
|---|---|---|
| how much does the perturbation grow? | $\|\delta h_{t,\ell}\|/r$ | a1, all layers |
| does it matter relative to the state? | $\|\delta h_{t,\ell}\|/\|h_{t,\ell}\|$ | a1 (`norm_h0`) |
| where along the depth does the *real* response grow? | $g_\ell=\|\delta h_{\ell}\|/\|\delta h_{\ell-1}\|$ | a1 |
| where can a *local* kick go? | same-position vs far-field response | probe |

The un-normalised $\|\delta h\|$ mixes the perturbation's growth with the state's own growth (×18–30);
the relative version cancels it. **For every claim about amplification, the relative response is the
right $y$-axis.** The depth map stays in the notebook as a diagnostic of the local tangent, not as a
measure of the amplification.

""" + anchor
assert anchor in s
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("inserted")
