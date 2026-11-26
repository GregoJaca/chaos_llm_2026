import os, sys, json
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

TAGS = {"deepseek": "DeepSeek-1.5B", "phi4mini": "Phi-4-mini", "qwen25_3b": "Qwen2.5-3B",
        "qwen3_4b": "Qwen3-4B", "qwen3_5": "Qwen3.5-2B"}

def ecdf(x):
    x = np.sort(x)
    return x, np.arange(1, len(x) + 1) / len(x)

def loglog_surv(x, k, b):
    return 1.0 / (1.0 + (x / k) ** b)

stats = {}
fig_rows = []
for tag, name in TAGS.items():
    rs, ts, ms = [], [], []
    distinct, topshare = [], []
    for i in range(1, 6):
        z = np.load(os.path.join(HERE, f"results/rstar/{tag}/p{i}/a1.npz"), allow_pickle=True)
        m = z["margins"].astype(float)
        rc = z["r_c"][0].astype(float)
        rc[:, m <= 0.02] = np.inf
        j = np.argmin(rc, axis=1)
        r = rc[np.arange(rc.shape[0]), j]
        ok = np.isfinite(r)
        rs.append(r[ok]); ts.append(j[ok]); ms.append(m[j[ok]])
        distinct.append(len(np.unique(r[ok])))
        c = np.bincount(j[ok], minlength=len(m))
        topshare.append(float(np.sort(c)[-5:].sum() / ok.sum()))
    r = np.concatenate(rs); t = np.concatenate(ts); mm = np.concatenate(ms)
    # fit survival function
    r_sorted, F = ecdf(r)
    S = 1 - F
    good = S > 0
    p0 = [np.median(r), 3.0]
    try:
        popt, _ = curve_fit(loglog_surv, r_sorted[good], S[good], p0=p0, maxfev=20000)
        pred = loglog_surv(r_sorted[good], *popt)
        r2 = 1 - np.sum((S[good] - pred) ** 2) / np.sum((S[good] - S[good].mean()) ** 2)
    except Exception:
        popt, r2 = [np.nan, np.nan], np.nan
    stats[name] = dict(n=int(len(r)), median=float(np.median(r)),
                       p10=float(np.percentile(r, 10)), p90=float(np.percentile(r, 90)),
                       k=float(popt[0]), b=float(popt[1]), r2=float(r2),
                       distinct_per_prompt=distinct, top5_share=topshare,
                       r=r, t=t, m=mm)
    print(f"{name:16s} n={len(r)} med={np.median(r):.3f} distinct/prompt={distinct} "
          f"top5={np.mean(topshare):.2f} fit k={popt[0]:.2f} b={popt[1]:.2f} R2={r2:.3f}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))

ax = axes[0, 0]
for name, s in stats.items():
    x, F = ecdf(s["r"])
    ax.plot(x, 1 - F, color=COL[name], lw=1.6, label=f"{name} (median {s['median']:.2f})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"critical radius $r_*$  (perturbation norm that flips the first token)")
ax.set_ylabel(r"$P(r_*>x)$")
ax.set_title("(a) Fragility distribution: a staircase, ordered by model", loc="left")
ax.legend(fontsize=7.2)
ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
s = stats["DeepSeek-1.5B"]
sc = ax.scatter(s["r"], s["t"], c=s["m"], s=12, cmap="viridis", vmin=0, vmax=0.3)
ax.set_xscale("log")
ax.set_xlabel(r"$r_*$  (log scale)")
ax.set_ylabel(r"controlling token step  $t_*=\arg\min_t r_c(t,u)$")
ax.set_title("(b) DeepSeek: $r_*$ is set by a handful of low-margin steps", loc="left")
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(r"baseline margin $m_{t_*}$ (nats)")
ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 0]
for name in TAGS.values():
    s = stats[name]
    ax.barh(name, np.mean(s["top5_share"]), color=COL[name], alpha=0.85)
    ax.text(np.mean(s["top5_share"]) + 0.01, name,
            f"{np.mean(s['top5_share']):.0%}", va="center", fontsize=8)
ax.set_xlim(0, 1.08)
ax.set_xlabel("fraction of directions whose first flip is decided by the 5 most frequent steps")
ax.set_title("(c) Five steps control 70-95% of all flips", loc="left")
ax.grid(alpha=0.3, axis="x")

ax = axes[1, 1]
for name in ["DeepSeek-1.5B", "Qwen3-4B"]:
    s = stats[name]
    x, F = ecdf(s["r"])
    S = 1 - F
    ax.plot(x, S, ".", ms=2, color=COL[name], label=f"{name} measured")
    xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
    ax.plot(xx, loglog_surv(xx, s["k"], s["b"]), "-", color=COL[name], lw=1.4,
            label=f"log-logistic fit, $R^2$={s['r2']:.2f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$r_*$  (log scale)")
ax.set_ylabel(r"$P(r_*>x)$")
ax.set_title("(d) A 2-parameter CDF fits the staircase; the steps remain visible", loc="left")
ax.legend(fontsize=7.2)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Where the smooth $r_*$ distribution comes from: a few decision steps and a quantised direction set",
             fontsize=12.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_rstar_ecdf.png")

json.dump({n: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in s.items()} for n, s in stats.items()},
          open(os.path.join(HERE, "analysis_update", "rstar_numbers.json"), "w"), indent=1)
print("saved rstar_numbers.json")
