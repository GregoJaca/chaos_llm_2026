import io
import re

P = "analysis_update/pack_loop_core.py"
s = io.open(P, encoding="utf-8").read()

# use the shared, correct discovery + pooling from looputil
s = s.replace("from cache import *\nfrom style import plt, save",
              "from cache import *\nfrom looputil import TAGS, LEGACY, gather, pairs\nfrom style import plt, save")

# drop the local duplicate definitions (TAGS exists in looputil; keep ROOTS etc out)
start = s.find("TAGS = {")
end = s.find("summary = {}")
if start != -1 and end != -1:
    s = s[:start] + s[end:]

io.open(P, "w", encoding="utf-8").write(s)
print("rewired pack_loop_core to looputil")
