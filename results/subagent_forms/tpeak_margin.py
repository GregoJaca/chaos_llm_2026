import numpy as np, os, json
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"
d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
pc = list(np.load(os.path.join(OUT, "t1_per_curve.npy"), allow_pickle=True))
m = d["margins"].astype(np.float64)
L27 = [row for row in pc if row["L"] == 27 and row["r"] <= 0.31]
tp = np.array([row["t_peak"] for row in L27])
print("t_peak (L=27, r<=0.3): median", np.median(tp), "IQR", np.quantile(tp, [.25, .75]),
      "range", tp.min(), tp.max())
# margin rank at t_peak vs uniform
med_m = np.median(m)
ranks = []
for t0 in tp:
    ranks.append((m <= m[int(t0)]).mean())
print("margin percentile at t_peak: median", np.median(ranks), " (0.5 = typical)")
print("margin at t=77:", m[75:80])
# margin percentiles at fixed spike times 16, 55-60, 75-85
for tt in [16, 57, 77, 85, 123, 143]:
    print(f"t={tt}: margin={m[tt]:.3f} (percentile {(m <= m[tt]).mean():.2f})")
# top-10 lowest margin tokens
o = np.argsort(m)[:10]
print("10 lowest-margin t:", sorted(o.tolist()), "values:", np.round(np.sort(m)[:10], 3))
json.dump(dict(tpeak_med=float(np.median(tp)), tpeak_iqr=[float(np.quantile(tp, .25)), float(np.quantile(tp, .75))],
               margin_pct_at_peak=float(np.median(ranks))), open(os.path.join(OUT, "tpeak_margin.json"), "w"))
