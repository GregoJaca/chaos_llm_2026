p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()

fixes = [
    (r"\lambd\|\delta h_{t}\|", r"\lambda(t)"),
    (r"\Lambd\|\delta h_{t}\|", r"\Lambda(t)"),
    (r"a(t,\ell+1)", r"\|\delta h_{t,\ell+1}\|"),
    (r"a(t,27)=\|\delta h_{t,27}\|_2", r"\|\delta h_{t,27}\|"),
    (r"a(t,\ell+1)/a(t,\ell)", r"\|\delta h_{t,\ell+1}\|/\|\delta h_{t,\ell}\|"),
]
for a, b in fixes:
    n = s.count(a)
    s = s.replace(a, b)
    print(f"{a!r}: {n}")

open(p, "w", encoding="utf-8").write(s)

# report leftovers
import re
for pat in [r"a\(t", r"\\lambd\\", r"\\Lambd\\", r"q\(t"]:
    hits = re.findall(pat, s)
    print("leftover", pat, len(hits))
