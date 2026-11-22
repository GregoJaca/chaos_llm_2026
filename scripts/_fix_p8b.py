src = open("notebook/controls.md", encoding="utf-8").read()
p8 = src.find("## Part 8")
p4body = src.find("**Data.** In-batch finite-difference")
assert p8 != -1 and p4body != -1 and p8 < p4body, (p8, p4body)
part8 = src[p8:p4body]
# Part 8 block should end with a '---' separator line; strip trailing separator
part8 = part8.rstrip()
if part8.endswith("---"):
    part8 = part8[:-3].rstrip()
# rebuild: everything before Part 8 + restored Part 4 heading + body onwards
src2 = src[:p8] + "## Part 4 \u2014 The state-level story: measured, not just proved\n\n" + src[p4body:]
if not src2.endswith("\n"):
    src2 += "\n"
src2 += "\n---\n\n" + part8 + "\n"
open("notebook/controls.md", "w", encoding="utf-8", newline="\n").write(src2)
import re
for m in re.finditer(r"^## Part.*$", src2, re.M):
    print(m.group(0)[:60])
