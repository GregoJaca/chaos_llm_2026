p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
anchor = "### N12. Gallery: flux-tube variants, all layers"
new = r"""### N13. Window spectroscopy: reading the context one token at a time

![wspec](figures/wspec_map.png)

**Data.** DeepSeek, prompt of $P=65$ tokens, **133 windows** $W=60\ldots139$ in steps of 1 and
$W=140\ldots400$ in steps of 5, one radius $r=0.1$, 4 directions, 200 generated tokens, layer 7.
This is the experiment proposed as P3 in Part 7, already run.

**Panel (a).** Response at fixed $t$ as a function of $W$. Two regimes, and the boundary is exactly
predicted:
- for $W<P+t$ the curve fluctuates strongly — every extra token of context changes the response,
  sometimes by a factor 1.5, in *both* directions;
- for $W\ge P+t$ the curve is **exactly flat**: at $t=20$ from $W=85$, at $t=60$ from $W=125$, at
  $t=100$ from $W=165$, at $t=150$ from $W=215$. Each equals $P+t$.

The flat point is the moment the window stops truncating the prompt: for $W\ge P+t$ the generated
token sees all $P$ prompt tokens. This is the cleanest confirmation of the $W-P$ mechanism in the
whole document, and it is a direct measurement, not an inference.

**Panel (b).** The relative change in the response caused by adding **one** prompt token,
$|\Delta\|\delta h\||/\|\delta h\|$ for $\Delta W=1$, placed at the prompt position it removes
($p=P+t-W$).

**Panel (c).** The same quantity collapsed over $t$: which prompt positions matter. Position 0
dominates (median 25% change), positions 8–14 are almost inert (1%), the rest sit at ~10%. The 20–80%
band is one order of magnitude wide, i.e. the importance of a token depends strongly on *where* in
the generation you are.

**Why this is useful.** It is a causal probe that uses no ablation: the window removes one token at a
time, reversibly and exactly. It converts the window from a nuisance into an instrument.

### N12. Gallery: flux-tube variants, all layers"""
assert anchor in s
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("inserted")
