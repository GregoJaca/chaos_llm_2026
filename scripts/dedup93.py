p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
lines = open(p, encoding="utf-8").read().split("\n")
idx = [i for i, l in enumerate(lines) if l.startswith("## 9.3 The spectrum is flat")]
print("duplicate headings at", idx)
if len(idx) == 2:
    start = idx[1]
    end = next(i for i, l in enumerate(lines) if l.startswith("## 9.3b"))
    del lines[start:end]
    open(p, "w", encoding="utf-8").write("\n".join(lines))
    print("removed", end - start, "lines")
else:
    print("nothing to do")
