p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()

# ---------- replace the six-views superposition block with the 3-panel version
start = "![superposition six views](figures/transport/superposition_six_views.png)"
i0 = s.index(start)
i1 = s.index("**Why 0.36 at the smallest radius")
new = r"""![superposition](figures/transport/superposition_3.png)

**What exactly is measured.** Three trials. In each trial one unit vector $u\in\mathbb{R}^{d}$ is drawn
and two runs are made on the same prompt (`prompts.txt`, $P=65$ tokens, window $W=100$, teacher-forced,
300 generated tokens, block $L$ = last):
1. **joint run**: $r\,u$ added to **every** prompt position; the response is
   $A_t=h'_t-h_t\in\mathbb{R}^{d}$ for each generated token $t$;
2. **single-position runs**: $r\,u$ added to **one** prompt position $p$ at a time; the response is
   $B_{p,t}$; the sum is $S_t=\sum_{p=0}^{64}B_{p,t}$.

If the model were linear in the input, $A_t=S_t$ exactly. The two vectors have the same dimension and
are compared component-wise; $\|A\|$, $\|S\|$ are their Euclidean norms.

**(a)** $\|S\|$ against $\|A\|$ for every token and trial: the points lie on the identity line over three
decades. **(b)** The ratio $\|S\|/\|A\|$ peaks at 1 with a spread of ±5–10%. **(c)** The relative error
falls as the amplitude grows; it is large (0.5–0.7) only where $\|A\|\lesssim10^{-2}$, i.e. where the
response is at the float32 resolution of the difference — not a physical failure.

"""
s = s[:i0] + new + s[i1:]

# ---------- add the flip-probability figure
anchor = "**Where the rule stops.** Above $r\\approx0.3$"
flip = r"""![flip fraction models](figures/fits/flip_fraction_models.png)

**What exactly is measured.** For each of **8 prompts**, **192 random unit directions**, and four radii,
the model generates greedily for 300 tokens with the perturbation on the prompt, and we record whether
the generated token sequence differs from the unperturbed run at least once. Panel (b) excludes flips
that occur at a step whose *unperturbed* margin is ≤0.02: those are quantisation ties where any
perturbation flips the token, and they dominate the raw counts (panel a). Band = 25–75% across prompts;
error bars = 95% confidence interval on the pooled fraction (Wilson), $n=1536$ direction–prompt pairs
per point.

**Tie-corrected flip fraction at $r=0.1\to1.0$:**

| model | 0.1 | 0.3 | 0.6 | 1.0 |
|---|---|---|---|---|
| DeepSeek-1.5B | 0.2% | 3.4% | 10.5% | 23.7% |
| Qwen3.5-2B | 3.6% | 20.1% | 45.8% | 71.9% |
| Qwen3-4B | 16.8% | 19.9% | 24.7% | 38.3% |
| Qwen2.5-3B | 30.4% | 33.7% | 42.6% | 53.0% |
| Phi-4-mini | 56.0% | 65.6% | 65.5% | 66.6% |

**Reading.** The ordering is the same as the amplification ordering (DeepSeek ≪ Qwen3.5 < Qwen3-4B <
Qwen2.5 < Phi), and the *shape* differs: DeepSeek starts near zero and rises with $r$; Phi is already at
56% at $r=0.1$ and saturates — it is fragile everywhere, not more fragile at larger $r$.

""" + anchor
assert anchor in s
s = s.replace(anchor, flip, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched master: superposition 3 panels + flip fraction")
