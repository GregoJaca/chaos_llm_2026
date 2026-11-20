p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
anchor = "## 12.2 Four programmes, with costs"
new = r"""## 12.1b First result of the transport programme (running now)

![kernel](figures/transport/kernel_deepseek.png)

**Data.** DeepSeek, one prompt ($P=65$), one generated continuation ($T=300$), $r=0.1$. For **each of
the 65 prompt positions separately**, 8 random directions, teacher-forced, all layers. The matrix is
$\|\delta h_{t,L}\|/r$ at the last layer as a function of (perturbed prompt position $p$, generated
token $t$); left panel, log colour.

**There are two components, and they are clean:**
1. **A persistent global term.** The first few prompt positions ($p\lesssim5$) influence *every*
   generated token, slowly decaying from ×27 to ×6 over 300 tokens. This is the part that survives at
   long distance.
2. **A travelling front.** For later positions the response peaks at a generated token that moves with
   $p$ (cyan line: peak $t=21$ at $p=0$, rising to $t=97$ at $p=64$, slope ≈1). A prompt token
   matters most to the generated tokens "opposite" it in the continuation, and less to tokens far
   before or after.

**Why this matters.** This is the first measurement that decomposes the response into "a constant
background from the prompt start" and "a moving local influence". The same-position chain of Part 9
sees neither. If the two components have different model signatures (Phi's far-field growth, DeepSeek's
decay), the model comparison finally has a mechanism rather than a list of numbers.

**Next.** Same kernel for 5 models × 8 prompts (queued), then fit the decay of each component with
distance and test whether the front speed (peak $t$ vs $p$) is universal (it is 1 by construction if
attention is shift-invariant; the interesting part is the *amplitude* profile).

""" + anchor
assert anchor in s
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
