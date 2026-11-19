p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
anchor = "### N3. The nonlinearity at large $r$ is a U-shaped (concave) law"
new = r"""### N2d. Where a kick lands: the position scan (five models)

![probe position](figures/fits/probe_position_law.png)

**Data.** A vector of norm $r$ (0.1 for DeepSeek, Qwen3-4B, Qwen2.5-3B, Qwen3.5; 0.3 for Phi) is
added to the residual stream at **block 0** at one generated position $s$, then the model is run
teacher-forced. 6 directions per point; 12 positions from 4 to 100. The plotted quantity is
$$A(s)=\frac{1}{r}\max_{1\le\tau\le50}\|\delta h_{s+\tau,L}\|$$
— the largest downstream response at the last layer within 50 tokens of the kick, divided by the kick
size. It has no units.

| model | median $A$ | min–max | max/min | shape |
|---|---|---|---|---|
| Qwen3-4B | 6.1 | 5.3–7.1 | 1.3 | flat |
| Qwen2.5-3B | 4.6 | 4.5–4.8 | 1.07 | flat |
| Qwen3.5-2B | 0.50 | 0.45–1.07 | 2.4 | flat after $s=8$ |
| Phi-4-mini | 8.3 | 1.7–17.1 | 10 | rise to a maximum at $s\approx40$, dip, rise |
| DeepSeek-1.5B | 0.11 | 0.05–0.76 | 15 | contraction + one spike at $s=16$ |

**Reading.** The *mean* amplification separates the models (Phi ×8, Qwen3 ×6, Qwen2.5 ×4.6, DeepSeek
×0.1, Qwen3.5 ×0.5), but the *structure* separates them just as sharply. Three models have a
position-independent gain: whatever the token, a kick is amplified by the same factor. Two have hot
spots: DeepSeek at $s=16$ and Phi at $s\approx40$.

**The Phi hot spot is the same event as the $t\approx40$ burst.** The kick at $s=40$ is amplified ×17,
the largest value anywhere in the scan, and the prompt-perturbation response of the same model bursts
at $t\approx40$ (N6b). Two independent experiments therefore localise the same instability to the same
place in the continuation.

**Consequence for the "amplification factor" language.** There is no single number per model. What is
reproducible across models is the *plateau* value; what differs is whether the field is flat or has
gates. Any mechanism has to explain both.

**Fitting.** Because three curves are flat and two are non-monotone with a single maximum, no single
functional family describes all five: the BIC winner is a constant/gaussian for the flat ones and a
gaussian with $R^2=0.46$ for Phi — i.e. the fits are not the point here, the shapes are.

""" + anchor
assert anchor in s
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("inserted")
