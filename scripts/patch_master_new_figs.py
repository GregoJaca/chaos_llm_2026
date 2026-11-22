p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()

# --- 1. superposition: replace the single figure with the six views
old = "![superposition](figures/transport/superposition.png)"
new = r"""![superposition six views](figures/transport/superposition_six_views.png)

**Six views of the same test** (DeepSeek, 65 positions, 3 trials, last block, $W=100$).

**(a)** $\|S\|$ against $\|A\|$ for every token and trial: the points lie on the identity line over
three decades. **(b)** The ratio $\|S\|/\|A\|$ peaks at 1 with a spread of ±5–10%. **(c)** $\cos(A,S)$
is 0.99–1.00. **(d)** The error **relative to the state norm** is $10^{-4}$–$10^{-3}$: negligible for
the state. **(e)** The gain of the linear prediction, $\cos(A,S)\|S\|/\|A\|$, is 1.00 ± 0.05.
**(f)** The relative error falls from ~0.7 at the smallest amplitude (where the response is at the
float32 floor) to 0.05 at the largest."""

assert old in s
s = s.replace(old, new, 1)

# --- 2. layer probe temporal, in section 8 before the table
anchor2 = "**The consistent table** (same prompt, $r=0.3$, last layer, 20 directions, perturbation at $s=16$):"
new2 = r"""![layer probe temporal](figures/fits/layer_probe_temporal.png)

**Parameters.** One-position perturbation at $s=16$, $r=0.1$ ($r=0.3$ for Phi), injected at block 0;
6 directions; prompt `prompts.txt`, $W=100$; teacher-forced, 300 tokens. Solid: the response at the
**perturbed position**; dashed: 5 tokens later; dotted: 20 tokens later. Band = 10–90% over directions.

**What it shows.** At the perturbed position the response is ×1–3 in every model. Twenty tokens later
DeepSeek has decayed to ×0.05–0.7 while Qwen3-4B and Phi have **grown** to ×1–9. The transported
response, not the local one, is where the models differ.

""" + anchor2
assert anchor2 in s
s = s.replace(anchor2, new2, 1)

# --- 3. window token agreement, in section 9
anchor3 = "The law for the flush step is"
new3 = r"""![window token agreement](figures/veto/window_token_agreement_multi.png)

**Parameters.** DeepSeek, prompt of 65 tokens, no perturbation ($r=0$), free-running greedy generation,
300 tokens, windows 60–1000 compared with the unwindowed run ($W=4096$).

**(a)** The running fraction of tokens identical to the unwindowed text. **(b)** The first differing
token against $W$: it lies on $W-P$ (dashed) for every window. **(c)** The residual latency,
$t_{\rm first}-(W-P)$: 2–7 tokens (mean 5), the time needed for the lost context to change a decision.
**(d)** The surviving fraction of the 300 tokens against $W$; the best fit is a stretched rise
$1-\exp(-(W/237)^{2.0})$ ($R^2=0.999$). **Reading: the text is bit-identical for 300 tokens once
$W\gtrsim500$, i.e. the effective memory of this continuation is ~5–8× the prompt length.**

""" + anchor3
assert anchor3 in s
s = s.replace(anchor3, new3, 1)
open(p, "w", encoding="utf-8").write(s)
print("inserted 3 figures")
