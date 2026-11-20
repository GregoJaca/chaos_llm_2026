p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()

start = "## 9.2 The depth exponent: not an exponential, a shifted $1/\\ell$ law"
end = "## 9.4 The layer-resolved gain, every model"
i0 = s.index(start)
i1 = s.index(end)

new = r"""## 9.2 The depth map: what is actually measured, and why $\lambda_1(\ell)$ was a bad curve

**First, the definitions in plain words.**

- Take the hidden state at one generated position and add a small vector $v$ to it. Push the rest of
  that position through block $\ell$ with the rest of the sequence held fixed. The change comes out
  linearly transformed: $\delta h' = J_\ell v$. $J_\ell$ is that block's Jacobian at that position.
- The **singular values** $\sigma_1\ge\sigma_2\ge\dots$ of $J_\ell$ are the largest and smallest
  factors by which it can stretch *any* vector. $\sigma_1$ is the biggest possible stretch. The
  subscript is the rank of the singular value, **not** the layer, the token, or the time.
- Push through $\ell+1$ blocks in a row and you get the product
  $P_\ell=J_\ell J_{\ell-1}\cdots J_0$. Its singular values $\sigma_i(\ell)$ say how much a
  perturbation is stretched after $\ell$ blocks. $\sigma_1(\ell)$ is the best-case stretch.
- The **finite-time Lyapunov exponent** is defined as
  $$\lambda_i(\ell)=\frac{\log\sigma_i(\ell)}{\ell+1}.$$
  The division by $\ell+1$ is there because the standard Lyapunov exponent is "log stretch per step",
  and for a long chain that average converges to a number independent of the length. It is the
  natural normalisation when the chain is long.

**Why the curve we plotted looked like a decay.** Because of that division. Take the measured
$\sigma_1(\ell)$: it is $16$–$17$ for the first 18 blocks and then rises to $28$ in the last ten. Over
28 blocks $\sigma_1$ changes by a factor $1.75$, while the denominator $\ell+1$ changes by a factor
28. The resulting $\lambda_1(\ell)$ therefore falls like $1/\ell$ — **the fall is created by dividing
by the block count, not by anything the network does.**

That is what "the shape is kinematics, not dynamics" was trying to say, and it was a bad way to say
it. The honest version: *we plotted a normalised quantity whose denominator grows, so its decay is
not evidence about the network.* I should have plotted the numerator.

![sigma1 vs depth](figures/fits/sigma1_vs_depth.png)

**Panel (a).** The numerator: $\sigma_1(\ell)$, median over 16 tokens, band = 25–75%. It is a
**plateau followed by a late rise**, not a decay.
**Panel (b).** Normalised by its first value: a small bump at blocks 3–7 (×1.05–1.11), a shallow dip
to 0.98 around block 9, a flat middle, then a **×1.75 rise over the last ten blocks**.
**Panel (c).** Fitted with the tool: the best form is a Hill/sigmoid
($R^2=0.979$), the second a stretched rise ($\Delta$BIC $=4.7$); a pure exponential is 20+ BIC worse.

**What is real and what is not.**

| statement | status |
|---|---|
| $\lambda_1(\ell)$ decays like $1/\ell$ | **artefact of the normalisation** — do not report as a finding |
| $\sigma_1$ is flat for 18 blocks then rises ×1.75 | real, measured |
| one block (block 0) already stretches by ×16 | real, measured (this is the entry of the prompt perturbation) |
| the per-block gain in the middle is ≈1.00–1.02 | real, and expected for residual blocks |
| the shape of $\sigma_1(\ell)$ is the same at all 16 tokens | supported (the band is narrow except at the late rise) |

So there is *some* non-trivial content — **where** the gain happens — but the decaying curve we
showed first was the wrong object. The interesting plot is $\sigma_1(\ell)$, and the interesting
question is whether the "plateau + late rise" shape is the same in other models, which the cross-model
depth maps (running now) will answer.

**And why 16 tokens?** Because each token costs 28 blocks × 8 directions of Jacobian-vector products.
This is a statistics question, not a compute question: the shape is stable across tokens (the earlier
8-token and the current 16-token curves agree to a few percent), so the 16-token curve is adequate
*for the shape*. A run with 128 tokens is in the queue to make the late-rise band tight, not to change
the conclusion.

## 9.3 The spectrum is flat: what "isotropic" means here

![depth spectrum shape](figures/fits/depth_spectrum_shape.png)

**Panel (a).** $\sigma_1,\sigma_2,\dots,\sigma_8$ of the depth product at four depths, all on one
axis: they are within a few percent of each other.
**Panel (b).** The ratio $\sigma_i/\sigma_1$ at every depth: flat at 1 within ±3%.
**Panel (c).** The slope of $\lambda_i$ against $i$ is zero at every depth ($-0.01\pm0.005$).

**In words.** Pick a perturbation direction at random. After 20 blocks it is stretched by almost the
same factor whatever direction you picked: the map behaves like "multiply everything by 20", not like
"stretch one direction and squash the others". That is what *isotropic* means here.

**Consequence for $\lambda_1$.** In the usual chaos story $\lambda_1$ is special because it is the
growth rate of the *one* direction that dominates. Here all eight computed directions grow at the same
rate, so there is no dominant direction and quoting $\lambda_1$ adds nothing over the common value.
We do not need $\lambda_2,\dots$: they are the same number. That is why only one curve was shown.

**Is this interesting or generic?** Mostly generic, and it is worth being explicit about why. A
transformer block with a residual connection is the identity plus a correction, $h\mapsto h+f(h)$. If
the correction's derivative is small at our position — and it is, since the middle blocks barely
change the perturbation — then all singular values of the block Jacobian sit near 1 and the product is
close to a multiple of the identity. So the flat spectrum is a *property of residual architectures*,
not a discovery about language models. It is a check that passes, and it tells us that any
non-trivial tangent structure must come from the blocks where the gain is not 1: the first block and
the last ten.

**What we should report instead.** The per-block singular values, which localise the gain:

![block spectrum](figures/fits/block_spectrum.png)

**Panel (a).** $\sigma_1,\sigma_2,\sigma_4$ of **one block's** Jacobian at each depth (median over
tokens). **Panel (b).** $\sigma_1/\sigma_8$ within one block: above 1 means the block itself is
anisotropic.

This is the measurement that explains the depth map: the gain is concentrated in specific blocks
rather than spread evenly, and the question "which blocks" is answerable and model-specific. The
cross-model version is the next section.

"""

s = s[:i0] + new + s[i1:]
open(p, "w", encoding="utf-8").write(s)
print("replaced", len(new))
