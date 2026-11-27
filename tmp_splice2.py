"""Splice the rewritten section F into notebook/update.md safely.

Strategy: split the F section into blocks at the '![Fig.' markers, keep the blocks
we want (in order), renumber them, and insert the new text.  No regex spanning
paragraphs, so nothing can be swallowed by accident.
"""
import io
import re

P = "notebook/update.md"
s = io.open(P, encoding="utf-8").read()
newF = io.open("notebook/_sectionF.md", encoding="utf-8").read()

f_start = s.index("# F. Soft loop and the gate law")
g_start = s.index("# G. Linearity and precision")
head, fsec, tail = s[:f_start], s[f_start:g_start], s[g_start:]

# split into blocks: prefix text, then (figure header, body) pairs
parts = re.split(r"(?m)^(?=!\[Fig\. )", fsec)
prefix, blocks = parts[0], parts[1:]

keep_order = [
    ("Fig. F2b", "Fig. F7b"),   # feedback exponent
    ("Fig. F3b", "Fig. F8b"),   # live density sorted
    ("Fig. F3c", "Fig. F8c"),   # live density fit
    ("Fig. F4b", "Fig. F9b"),   # gate law lin-log
    ("Fig. F5b", "Fig. F9d"),   # gate law lin-log per model
    ("Fig. F4", "Fig. F9"),     # gate law log-log
    ("Fig. F5", "Fig. F9c"),    # gate law per model
    ("Fig. F7", "Fig. F10"),    # gate geometry
    ("Fig. F8", "Fig. F11"),    # rates by model
    ("Fig. F9", "Fig. F12"),    # old sampling figure (kept for reference)
    ("Fig. F10", "Fig. F13"),   # rank / entropy
    ("Fig. F11", "Fig. F14"),   # d2pipe
    ("Fig. F13", "Fig. F15"),   # metric vs T
    ("Fig. F14", "Fig. F16"),   # FSLE
    ("Fig. F15", "Fig. F17"),   # staircase
    ("Fig. F16", "Fig. F18"),   # endpoint rates
    ("Fig. F17", "Fig. F19"),   # finite-time exponent
    ("Fig. F18", "Fig. F20"),   # power spectrum
    ("Fig. F20", "Fig. F21"),   # intermittency
]

by_label = {}
for b in blocks:
    m = re.match(r"!\[(Fig\. [^\]]+)\]", b)
    if m:
        by_label[m.group(1)] = b

out = [newF]
missing = []
for old, new in keep_order:
    if old not in by_label:
        missing.append(old)
        continue
    b = by_label[old]
    b = b.replace(f"![{old}]", f"![{new}]", 1)
    b = b.replace(f"**{old} ", f"**{new} ", 1)
    if old == "Fig. F2b":
        b = """![Fig. F7](update_figures/pack_loop_metric_M.png)

**Fig. F7 — mean log amplification.** $M=\\langle\\log(a_t/a_1)\\rangle_t$ over the whole run, median over directions, at $T=1$; bars are closed loop, replay, CRN and independent. **Data:** `results/softloop*` / `results/longloop*`, 300-3000 steps. **Read:** the closed loop has $M=1$-$7$ nats; replay and CRN sit at 0 or below; independent sampling is near 0. **It integrates the whole trajectory**, so it does not depend on where the curve saturates, but it is still horizon-dependent.

""" + b
    out.append(b)

io.open(P, "w", encoding="utf-8").write(head + "".join(out) + tail)
print("kept", len(keep_order) - len(missing), "blocks; missing:", missing)
print("new length", len(head) + len("".join(out)) + len(tail))
