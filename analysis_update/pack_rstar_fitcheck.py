import os, sys, json
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kstest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

def surv(x, k, b):
    return 1 / (1 + (x / k) ** b)

ORDER = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Phi-4-mini"]
rows = {}
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
for n in ORDER:
    r = np.sort(rstar(n)["r"])
    S = 1 - np.arange(1, len(r) + 1) / len(r)
    ok = S > 0
    x, y = r[ok], S[ok]
    p0 = [np.median(r), 3]
    pl, _ = curve_fit(surv, x, y, p0=p0, maxfev=20000)
    # log-space fit
    def logsurv(xx, k, b):
        return np.log(surv(xx, k, b))
    pg, _ = curve_fit(logsurv, x, np.log(y), p0=p0, maxfev=20000)
    pred_l = surv(x, *pl); pred_g = surv(x, *pg)
    r2_l = 1 - ((y - pred_l) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2_g = 1 - ((np.log(y) - np.log(pred_g)) ** 2).sum() / ((np.log(y) - np.log(y).mean()) ** 2).sum()
    ks = kstest(r, lambda u: 1 - surv(u, *pg)).statistic
    rows[n] = dict(k_log=float(pg[0]), b_log=float(pg[1]), r2_lin=float(r2_l),
                   r2_log=float(r2_g), ks=float(ks), k_lin=float(pl[0]), b_lin=float(pl[1]))
    if n in ("DeepSeek-1.5B", "Qwen3-4B"):
        ax = axes[0] if n == "DeepSeek-1.5B" else axes[1]
        ax.loglog(x, y, ".", ms=2.5, color=COL[n], label="measured")
        ax.loglog(x, pred_l, "-", lw=1.2, color="#1f77b4", label=f"linear fit $R^2$={r2_l:.3f}")
        ax.loglog(x, pred_g, "--", lw=1.4, color="#d62728", label=f"log fit $R^2_{{\\log}}$={r2_g:.3f}")
        ax.set_xlabel(r"$r_*$")
        ax.set_ylabel(r"$P(r_*>x)$")
        ax.set_title(n, loc="left")
        ax.legend(fontsize=7.2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_rstar_fit_check.png")
json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_rstar_fit_check.json"), "w"), indent=1)
for n, v in rows.items():
    print(n, v)

