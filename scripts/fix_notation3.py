p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
for a, b in [
    (r"$$a(t,\ell;r)=r\,G(t,", r"$$\|\delta h_{t,\ell}\|(r)=r\,G(t,"),
    (r"median over directions of $a(t,27)$", r"median over directions of $\|\delta h_{t,27}\|$"),
]:
    print(a, s.count(a))
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
