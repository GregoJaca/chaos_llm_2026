src = open("notebook/controls.md", encoding="utf-8").read()
start = src.find("## Part 8")
end = src.find("## Part 4 \u2014 The state-level story")
assert start != -1 and end != -1 and start < end, (start, end)
part8 = src[start:end]
src2 = src[:start] + src[end:]
src2 = src2.replace("---\n\n---\n", "---\n")
if not src2.endswith("\n"):
    src2 += "\n"
src2 += "\n" + part8.rstrip() + "\n"
open("notebook/controls.md", "w", encoding="utf-8", newline="\n").write(src2)
print("moved Part 8 to the end; length", len(src2))
