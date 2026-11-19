p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
for a, b in [
    (r"a(t+10)", r"\|\delta h_{t+10,\ell}\|"),
    (r"a(t-10)", r"\|\delta h_{t-10,\ell}\|"),
    (r"a(t,27)/r", r"\|\delta h_{t,27}\|/r"),
]:
    print(a, s.count(a))
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
import re
print("remaining a(t:", len(re.findall(r"a\(t", s)))
for m in re.finditer(r"a\(t[^)]*\)", s):
    print("  ", s[m.start():m.end()])
