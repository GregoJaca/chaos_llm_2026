p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()

old = r"""![perturbation vs position](figures/fits/perturbation_vs_position.png)

**What is plotted.** A one-vector perturbation at **one generated position** $s$. For each direction,
take the largest response at the **last layer** in the 50 tokens after $s$; plot the median over
directions (line) and the 10–90% range (band), divided by $r$. 22 positions, generated-token
positions, $r=0.1$ ($r=0.3$ for Phi).

**What it shows.** Phi rises from ×30 at $s=4$ to ×130 at $s=40$ and falls back; DeepSeek stays below
×2.5 with a maximum at $s=16$; Qwen3-4B (×11–18), Qwen2.5 (×8–10) and Qwen3.5 (×1–3) are flat. The
model difference is in the *level* and in whether there are hot spots."""

new = r"""![heatmap](figures/positions/posscan_A_heatmap.png)

**Parameters.** One vector of norm $r=0.1$ ($r=0.3$ for Phi) at **one generated position** $s$,
injected at block 0, read at the last block $L$; 8 random directions, median over directions; the
prompt is `prompts.txt` (65 tokens), teacher-forced, 300 generated tokens; 22 positions from 4 to 100.

**What is plotted.** The response $\log_{10}\|\delta h_{s+\tau,L}\|/r$ as a function of the position $s$
and the lag $\tau$ (0–50 tokens after $s$).

**What it shows — and it corrects the earlier reading.** For Phi there is a **bright diagonal**: a
perturbation at position $s$ is amplified when the continuation reaches a particular *absolute* token,
so the peak lag shrinks as $s$ grows. DeepSeek shows horizontal bands (the response depends on the
position, not on the lag) and a global decay.

![peak absolute](figures/positions/peak_is_absolute.png)

**Panel (a).** The lag of the peak against the position. **Panel (b).** The absolute token of the peak,
$s+\tau^*$. For early positions it is **flat** at a model-specific value (DeepSeek ≈20–45, Qwen3-4B
≈20, Qwen2.5-3B ≈27, Phi ≈45, Qwen3.5 ≈42); for late positions it follows the position.

**Consequence.** The "hot spot at $s\approx40$" in the max-over-lag plot was a projection artefact: a
perturbation at $s=40$ peaks at lag ≈5, i.e. at the same absolute token where a perturbation at $s=16$
peaks at lag ≈29. **There is one amplification gate per continuation, not one per position.** The model
difference is the *time* of that gate (or its absence, for DeepSeek) and its amplitude, not a position."""

assert old in s
s = s.replace(old, new, 1)

# add relative-response figure with bands to 5.1
old2 = r"""**Answer.** Only **Phi-4-mini** makes the perturbation more relevant: its relative size grows ×20
through the depth. DeepSeek and Qwen3-4B peak at ×4 and ×3 near mid-depth and then decline; Qwen3.5
stays within ×1.2. This is the quantity to quote when the question is "is the perturbation important
for the state", because it cancels the residual stream's own growth (×18–30)."""
new2 = r"""![relative response](figures/fits/relative_response_models.png)

**Parameters.** Prompt perturbation, $r=0.1$ ($r=0.3$ for Phi), 10–20 directions, generated token
$t=50$; line = median over directions, band = 25–75%; all saved layers.

**Answer.** Only **Phi-4-mini** makes the perturbation more relevant: its relative size grows ×20
through the depth. DeepSeek peaks at ×3 near layer 13 and then declines; Qwen3-4B, Qwen2.5 and Qwen3.5
stay within ×1.5–2 of their initial value. This is the quantity to quote when the question is "is the
perturbation important for the state", because it cancels the residual stream's own growth (×18–30)."""
assert old2 in s
s = s.replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched master: position section + relative response")
