import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Qwen3.5-9B",
          "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen"]
D = {}
for n in MODELS:
    xs, ys = [], []
    for T, z, m in soft_runs(n):
        if "m_ref" not in z or "sig_read" not in z:
            continue
        mm = z["m_ref"].astype(float); s = z["sig_read"].astype(float)
        ok = np.isfinite(mm) & np.isfinite(s) & (s > 0)
        xs.append(mm[ok] / T); ys.append(s[ok])
    if not xs:
        continue
    x = np.concatenate(xs); y = np.concatenate(ys)
    sel = x > 1
    A = np.polyfit(x[sel], np.log(y[sel]), 1)
    G = np.exp(A[1])
    ident = G / 4 / np.cosh(x / 2) ** 2
    resid = np.abs(np.log(y[sel]) - np.log(ident[sel]))
    D[n] = dict(x=x, y=y, slope=A[0], G=G, resid=np.median(resid))

fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax = axes[0]
for n, d in D.items():
    idx = np.arange(0, len(d["x"]), max(1, len(d["x"]) // 4000))
    ax.scatter(d["x"][idx], d["y"][idx], s=3, alpha=0.25, color=COL[n], label=n)
xx = np.linspace(0, 8, 300)
ax.plot(xx, 2.0 / 4 / np.cosh(xx / 2) ** 2, "k--", lw=1.0, label="identity, $G=2$")
ax.set_yscale("log")
ax.set_xlim(0, 8)
ax.set_ylim(1e-7, 3)
ax.set_xlabel(r"$m_t/T$  (linear scale)")
ax.set_ylabel(r"$\sigma_{\max}(\Sigma_p)$  (log scale)")
ax.set_title("(a) Gate law, lin-log", loc="left")
ax.legend(fontsize=6.0, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
names = list(D)
y = np.arange(len(names))
for i, n in enumerate(names):
    d = D[n]
    ax.barh(i, d["resid"], color=COL[n], alpha=0.85)
    ax.text(d["resid"] + 0.02, i, f"G={d['G']:.1f}", va="center", fontsize=6.6)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.4)
ax.set_xlabel(r"median $|\log\sigma_{\max}-\log(G/4\cosh^2(m/2T))|$ for $m/T>1$")
ax.set_title("(b) Distance from the two-token identity", loc="left")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "pack_gate_linlog.png")

# per-model lin-log
fig, axes = plt.subplots(3, 3, figsize=(14.4, 11.0))
for ax, n in zip(axes.flat, MODELS):
    if n not in D:
        ax.axis("off"); continue
    d = D[n]
    idx = np.arange(0, len(d["x"]), max(1, len(d["x"]) // 3000))
    ax.scatter(d["x"][idx], d["y"][idx], s=2, alpha=0.25, color=COL[n])
    xx = np.linspace(0, min(8, d["x"].max()), 300)
    ax.plot(xx, d["G"] / 4 / np.cosh(xx / 2) ** 2, "r-", lw=1.2,
            label=r"identity, $G$=%.1f" % d["G"])
    ax.plot(xx, np.exp(d["slope"] * xx + np.log(d["G"])), "k--", lw=1.0,
            label=r"tail slope %.2f" % d["slope"])
    ax.set_yscale("log")
    ax.set_ylim(1e-7, max(2, np.percentile(d["y"], 99.5) * 2))
    ax.set_xlabel(r"$m_t/T$", fontsize=7.6)
    ax.set_ylabel(r"$\sigma_{\max}$", fontsize=7.6)
    ax.set_title(n, fontsize=8.6)
    ax.legend(fontsize=5.8); ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Gate law per model, lin-log; the red curve is the identity with the fitted $G$", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "pack_gate_linlog_per_model.png")
json.dump({n: dict(slope=float(D[n]["slope"]), G=float(D[n]["G"]), resid=float(D[n]["resid"])) for n in D},
          open(os.path.join(HERE, "analysis_update", "pack_gate_linlog.json"), "w"), indent=1)
print("done")
