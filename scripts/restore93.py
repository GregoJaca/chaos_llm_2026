import subprocess

p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
cur = open(p, encoding="utf-8").read()
old = subprocess.check_output(["git", "-C", r"C:\Users\hjaca\research\chaos_llm_2026",
                               "show", "HEAD:notebook/notebook.md"], text=True, encoding="utf-8")

lines = old.split("\n")
i0 = next(i for i, l in enumerate(lines) if l.startswith("## 9.3 The spectrum is flat"))
i1 = next(i for i, l in enumerate(lines) if l.startswith("## 9.3b"))
sec = "\n".join(lines[i0:i1])
print("recovered", len(sec), "chars")

# insert before the 9.3b heading in the current file
marker = "## 9.3b Where each model does its stretching"
assert marker in cur
cur = cur.replace(marker, sec + "\n" + marker, 1)
open(p, "w", encoding="utf-8").write(cur)
print("restored")
