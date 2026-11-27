import json
import io
import numpy as np

D = json.load(open("analysis_update/pack_loop_core.json"))
order = ["Mamba-130m", "DeepSeek-1.5B", "DeepSeek-14B", "Qwen3-14B", "Phi-4",
         "Phi-4-mini", "Pythia-410m", "Qwen3-4B", "Qwen3.5-9B", "randQwen",
         "Qwen2.5-3B", "Qwen3.5-2B"]
rows = ["| model | directions | steps | $L_\\infty$ (nats) | $\\tau$ (steps) | $R^2$ | $\\times$ at end |",
        "|---|---|---|---|---|---|---|"]
for m in order:
    d = D.get(m)
    if not d:
        rows.append(f"| {m} | – | – | – | – | – | (no matched pair yet) |")
        continue
    L = d.get("L_inf", float("nan"))
    tau = d.get("tau", float("nan"))
    r2 = d.get("sat_r2", float("nan"))
    fac = np.exp(d.get("L_end", np.nan))
    rows.append("| %s | %d | %d | %.1f | %.0f | %.2f | %s |" % (
        m, d.get("n_dirs", 0), d.get("steps", 0), L, tau, r2,
        ("%s" % format(int(round(fac)), ",")).replace(",", "\\,") if np.isfinite(fac) else "–"))
table = "\n".join(rows)

P = "notebook/update.md"
s = io.open(P, encoding="utf-8").read()
start = s.index("| model | directions | steps |")
end = s.index("The $R^2$ column matters")
s = s[:start] + table + "\n\n" + s[end:]
io.open(P, "w", encoding="utf-8").write(s)
print(table)
