"""Splice the rewritten section F into notebook/update.md, renumbering the figures
that follow so the labels stay unique and in order."""
import io
import re

P = "notebook/update.md"
s = io.open(P, encoding="utf-8").read()
newF = io.open("notebook/_sectionF.md", encoding="utf-8").read()

start = s.index("# F. Soft loop and the gate law")
end = s.index("![Fig. F1](update_figures/pack_softloop_closed_all.png)")
# keep everything from the old F1 onwards for now; we will renumber below
s = s[:start] + newF + "\n" + s[end:]

# ---- renumber the surviving old figures so labels stay unique
# order of appearance after the new section
ren = [
    ("Fig. F1", "Fig. F1"),      # placeholder, handled by dropping the duplicate below
]

# drop the now-duplicated blocks that the new section replaces
def drop_block(text, label):
    """remove '![Fig. X](...)' + the following bold caption paragraph"""
    pat = re.compile(r"!\[" + re.escape(label) + r"\]\([^)]*\)\n\n\*\*.*?\*\*\n\n", re.S)
    return pat.sub("", text, count=1)

s = drop_block(s, "Fig. F1")        # old F1: now in the new section
s = drop_block(s, "Fig. F2")        # old metric_M -> we re-add it as F7 below
s = drop_block(s, "Fig. F19")       # old selfreg (new F3)
s = drop_block(s, "Fig. F19b")      # old selfreg big (new F3b)
s = drop_block(s, "Fig. F21")       # old temperature (new F6)

# renumber the rest (do it in one pass over the placeholder tokens)
mapping = {
    "Fig. F2b": "Fig. F7b",   # feedback
    "Fig. F3b": "Fig. F8b",   # live density sorted
    "Fig. F3c": "Fig. F8c",   # live density fit
    "Fig. F4b": "Fig. F9b",
    "Fig. F5b": "Fig. F9d",
    "Fig. F4":  "Fig. F9",
    "Fig. F5":  "Fig. F9c",
    "Fig. F7":  "Fig. F10",
    "Fig. F8":  "Fig. F11",
    "Fig. F9":  "Fig. F12",
    "Fig. F10": "Fig. F13",
    "Fig. F11": "Fig. F14",
    "Fig. F13": "Fig. F15",
    "Fig. F14": "Fig. F16",
    "Fig. F15": "Fig. F17",
    "Fig. F16": "Fig. F18",
    "Fig. F17": "Fig. F19",
    "Fig. F18": "Fig. F20",
    "Fig. F20": "Fig. F21",
}
# protect against chained replacement by using placeholders
for i, (old, new) in enumerate(mapping.items()):
    s = s.replace(f"![{old}]", f"![@@{i}@@]")
    s = s.replace(f"**{old} ", f"**@@{i}@@ ")
for i, (old, new) in enumerate(mapping.items()):
    s = s.replace(f"@@{i}@@", new)

io.open(P, "w", encoding="utf-8").write(s)
print("spliced; length", len(s))
