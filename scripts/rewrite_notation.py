import re

p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
orig = s

pairs = [
    (r"a(t,\ell)", r"\|\delta h_{t,\ell}\|"),
    (r"a(t,1)", r"\|\delta h_{t,1}\|"),
    (r"a(t-1)", r"\|\delta h_{t-1}\|"),
    (r"a(t)", r"\|\delta h_{t}\|"),
    (r"a(1)", r"\|\delta h_{1}\|"),
    (r"a(0)", r"\|\delta h_{0}\|"),
    (r"q(t,\ell)", r"\|\delta h_{t,\ell}\|/\|h_{t,\ell}\|"),
    (r"a_{\ell}(t)", r"\|\delta h_{t,\ell}\|"),
    (r"a_{t,\ell}", r"\|\delta h_{t,\ell}\|"),
]
for a, b in pairs:
    n = s.count(a)
    s = s.replace(a, b)
    print(f"{a!r} -> {b!r}: {n}")

# collapse accidental double slashes from repeated rewrites
s = s.replace(r"\\|", r"\|")
s = s.replace(r"$\\|", r"$\|")
open(p, "w", encoding="utf-8").write(s)
print("changed", len(orig) != len(s))
