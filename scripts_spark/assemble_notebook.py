"""Assemble notebook_big.md from Part 0 + the two analysis sections, applying
the window-semantics corrections (the first 14B runs executed before the
transformers>=4.5x layer_types fix: their sliding window was silently NOT
applied, so they are full-attention / W=4096-semantics runs).
"""
import re

HERE = "notebook_big"

part0 = open(f"{HERE}/sections/part0.md", encoding="utf-8").read()
roadmap = open(f"{HERE}/sections/part0b_roadmap.md", encoding="utf-8").read()
floor = open(f"{HERE}/sections/section_floor.md", encoding="utf-8").read()
part1 = open(f"{HERE}/sections/part1_core.md", encoding="utf-8").read()
a1 = open(f"{HERE}/sections/section_a1_deep.md", encoding="utf-8").read()
xs = open(f"{HERE}/sections/section_cross_scale.md", encoding="utf-8").read()

CALL = """
> **Correction on the attention window (read this first).** Every 14B result in
> Parts 1 and 2 was produced before we discovered that transformers 4.5x gates
> the sliding window on a `layer_types` config field that stock Qwen2.5 configs
> never set. The intended `window=100` was therefore silently ignored: **these
> runs used full attention** (equivalent to W=4096 within the 365-token horizon).
> The data itself is valid and internally consistent (baseline and perturbed
> passes share the same attention pattern; the free-run validation matched it),
> but wherever a sentence below says "window 100" for the 14B, read "full
> attention". A rerun with the window genuinely active is in progress; its
> numbers will replace the affected comparisons. This also fully explains why
> the 14B shows no response surge at t=35 in Part 2: with full attention there
> is nothing to flush.
"""

a1 = a1.replace(
    "| attention window | 100 tokens (`set_window(model, 100)`) |",
    "| attention window | **full attention** (the window-100 setting was silently "
    "inactive; see the correction note above) |")
a1 = a1.replace(
    "“Window 100” matters for interpretation: once $65+t>100$, the earliest prompt "
    "positions fall outside the attention window. The perturbation is applied only to "
    "prompt positions, so at late steps its influence reaches the current token indirectly, "
    "through the states of generated tokens that were computed while the prompt was still "
    "visible. The late-time response is therefore not a direct prompt-embedding effect.",
    "Because this run used full attention, the prompt stays visible at every generated "
    "step: the late-time response is direct prompt influence, attenuated only by the "
    "network's own mixing, not by any attention window cutting the prompt off.")
a1 = a1.replace(
    "- The window is 100 tokens, so after $t\\approx35$ the prompt is progressively "
    "outside the attention window; late-time responses are propagated through "
    "generated-token states. The layer curves mix “depth growth” with “prompt leaving "
    "the window.”",
    "- This run used full attention (see the correction note), so no part of the decay is "
    "attributable to the prompt leaving an attention window. The true-window run will "
    "separate that effect.")

xs = xs.replace(
    "| attention window | $W=100$ | $W=100$ |",
    "| attention window | $W=100$ (active, transformers 4.50 on x11) | **full attention** "
    "(window setting inactive; see the correction note) |")
xs = xs.replace(
    "| flush ratio, last state ($\\overline{\\lVert\\delta h\\rVert/r}_{36..40}$ over "
    "$\\overline{\\lVert\\delta h\\rVert/r}_{30..34}$) | 1.70 | 1.11 | — |",
    "| flush ratio, last state ($\\overline{\\lVert\\delta h\\rVert/r}_{36..40}$ over "
    "$\\overline{\\lVert\\delta h\\rVert/r}_{30..34}$) | 1.70 | 1.11 | — |")
xs = xs.replace(
    "## Summary",
    "## Summary\n\n"
    "> Same correction note as Part 1: the 14B numbers here are full-attention runs "
    "(window inactive). Everything except the flush-ratio row is a same-prompt, "
    "same-radius comparison and stands as stated; the flush row is explained by the "
    "missing window, and the margin comparison is partially confounded by the window "
    "regime (on the 1.5B the window is known to change the margin spectrum). The "
    "true-window 14B rerun will close both gaps.", 1)

out = (part0 + "\n\n" + roadmap + "\n\n" + floor + "\n\n" + part1 + "\n\n"
       + CALL + "\n\n" + a1 + "\n\n---\n\n" + xs + "\n")
open(f"{HERE}/notebook_big.md", "w", encoding="utf-8").write(out)
print("assembled", len(out), "chars")
