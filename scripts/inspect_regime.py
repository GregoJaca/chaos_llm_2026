import json
import numpy as np

rows = json.load(open(r"C:\Users\hjaca\research\chaos_llm_2026\results\figures_big\regime_rows.json"))
print("total rows", len(rows))
print("prompts", sorted(set(r["prompt"] for r in rows)))
print("windows", sorted(set(r["W"] for r in rows)))
for w in sorted(set(r["W"] for r in rows)):
    sel = [r for r in rows if r["W"] == w]
    pre = [r["pre"] for r in sel if np.isfinite(r["pre"])]
    post = [r["post"] for r in sel if np.isfinite(r["post"])]
    print(f"W={w:5d} n={len(sel):4d} pre_n={len(pre):4d} pre_mean={np.mean(pre) if pre else float('nan'):+.4f} "
          f"post_n={len(post):4d} post_mean={np.mean(post) if post else float('nan'):+.4f}")
