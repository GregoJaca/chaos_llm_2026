p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()

A = r"""## 12.1c What the transport kernel is, and why it is the right object

**The definition.** Let $K(p,t)$ be the response at generated token $t$ when the perturbation is
applied **only** at prompt position $p$:
$$K(p,t)=\frac{\|\delta h_{t,L}\|}{r}\quad\text{for a perturbation at }p\text{ alone.}$$
It is a matrix: 65 prompt positions $\times$ 300 generated tokens (per model, per prompt, per layer).

**Why this and not the usual response.** Our normal experiment perturbs *every* prompt position at
once. The measured response is then, to first order,
$$R(t)=\sum_{p} K(p,t)\,u_p,$$
i.e. a weighted sum of the kernel over the positions the perturbation touches. So $K$ is the
**Green's function** of the system: once you know it, every prompt-perturbation experiment is a
superposition. It is the object that answers "where does the response at this token come from?".

**What the first measurement shows (DeepSeek, one prompt).**
- **A persistent background** from the first few positions: $p\lesssim5$ influence *every* generated
  token, decaying only ×27 → ×6 over 300 tokens. These are the tokens the model keeps returning to
  (the start of the prompt acts like a constant reference).
- **A travelling front**: for later $p$ the response peaks at a generated token that moves with $p$,
  with slope 1. A prompt token matters most to the continuation around the matching offset, and much
  less to tokens far before or after it.
- **Everything else is small**: outside the front and the background, $K$ falls 1–2 orders of
  magnitude.

**What the 8-prompt × 5-model run will give.**
1. **Two decay laws.** Fit the background $B(t)=\langle K\rangle_{p\le5}(t)$ and the front amplitude
   $A(p)=\max_t K(p,t)$ separately, with the curve-fitting tool (power law vs exponential). The
   background decay rate is a new, model-specific number: it says how long the prompt start keeps
   influencing the text.
2. **The reach of a token**: the distance $|t-(p+c)|$ at which $K$ falls below 1% of its peak. This
   turns "attention window" from a mask parameter into an effective, measured range.
3. **Universality**: does the normalised profile $K(p,t)/A(p)$ look the same for all prompts of a
   model? For all models?
4. **The model difference.** The far-field behaviour (Part 9.2b: Phi's response at $t+20$ *grows×9*,
   DeepSeek's decays ×0.7) should appear here as a different background-versus-front balance. If Phi's
   background decays more slowly, or its front is wider, we will have located the mechanism of the
   model difference.
5. **A superposition test.** Sum $K(p,t)$ over all $p$ and compare with the ordinary all-position
   response. If they agree (up to the direction weights), the linear decomposition is validated and
   everything above is on solid ground. This is a free check from data we already have.

**Cost.** 65 positions × 8 directions × 300 tokens = 3.3 min per prompt (measured). 8 prompts × 5
models = 2.2 h, running now.

"""

s = s.replace("## 12.2 Four programmes, with costs", A + "## 12.2 Four programmes, with costs", 1)

E = r"""
## 12.5 On programme E (per-block rotation): not worth 6 h as proposed

You are right to be sceptical. As designed, E would re-measure $\sigma_{\max}$ at more positions and
confirm what three models already show: every block amplifies 3–6×, the top direction rotates, so the
product grows ~20 instead of $10^{18}$. That is a confirmation, not a new result.

The only version of E that would add something is a different question: **what are the rotating
directions?** Concretely, at each block, take the top *left* singular vector of the Jacobian and ask
how it aligns with (i) the residual stream $h$ at that position, (ii) the embedding of the next token,
(iii) the previous block's top vector. If the amplification always points along $h$ (radial), it
cannot change decisions; if it points along the next-token direction, it *is* the decision mechanism.
That is one experiment (~1 h on one model) and it would connect the rotation to the margin dynamics.

My recommendation: skip E as proposed; if you want it, fund the 1-hour alignment version instead.
"""

s = s.rstrip() + E
open(p, "w", encoding="utf-8").write(s)
print("ok")
