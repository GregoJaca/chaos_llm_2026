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
    for T, z, meta in soft_runs(n):
        if "m_ref" not in z or "sig_read" not in z:
            continue
        m = z["m_ref"].astype(float); s = z["sig_read"].astype(float)
        ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
        xs.append(m[ok] / T); ys.append(np.log(s[ok]))
    if not xs:
        continue
    x = np.concatenate(xs); y = np.concatenate(ys)
    sel = x > 1
    A = np.polyfit(x[sel], y[sel], 1)
    r2 = 1 - ((y[sel] - np.polyval(A, x[sel])) ** 2).sum() / ((y[sel] - y[sel].mean()) ** 2).sum()
    # randQwen: also fit x>3
    A3 = None
    if n == "randQwen":
        s3 = x > 3
        A3 = np.polyfit(x[s3], y[s3], 1)
        r23 = 1 - ((y[s3] - np.polyval(A3, x[s3])) ** 2).sum() / ((y[s3] - y[s3].mean()) ** 2).sum()
    else:
        r23 = None
    D[n] = dict(x=x, y=y, slope=A[0], icpt=A[1], G=np.exp(A[1]), r2=r2,
                A3=A3, r23=r23, n=int(sel.sum()))
    print(n, "slope", round(A[0], 3), "G", round(np.exp(A[1]), 2), "R2", round(r2, 3))

# ---------------------------------------------------------------- 1. log-log all
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
ax = axes[0]
for n, d in D.items():
    idx = np.arange(0, len(d["x"]), max(1, len(d["x"]) // 4000))
    ax.scatter(d["x"][idx], np.exp(d["y"][idx]), s=3, alpha=0.25, color=COL[n], label=n)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$m_t/T$")
ax.set_ylabel(r"$\sigma_{\max}(\Sigma_p)$")
ax.set_title("(a) Readout gain vs margin/temperature, log-log", loc="left")
ax.legend(fontsize=6.2, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
names = list(D.keys())
y = np.arange(len(names))
for i, n in enumerate(names):
    d = D[n]
    ax.errorbar(d["slope"], i, xerr=0.03, fmt="o", color=COL[n], ms=6)
    ax.text(0.12, i, f"G={d['G']:.1f}, $R^2$={d['r2']:.2f}", fontsize=6.6, va="center", color=COL[n])
ax.axvline(-1, color="k", ls="--", lw=1.0)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.4)
ax.set_xlabel(r"slope of $\log\sigma_{\max}$ vs $m_t/T$ for $m_t/T>1$")
ax.set_title("(b) Slope is $-1$ for every model", loc="left")
ax.set_xlim(-1.2, 0.3); ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "pack_gate_law_loglog.png")

# ---------------------------------------------------------------- 2. per model fits
fig, axes = plt.subplots(3, 3, figsize=(14.4, 11.0))
for ax, n in zip(axes.flat, MODELS):
    if n not in D:
        ax.axis("off"); continue
    d = D[n]
    idx = np.arange(0, len(d["x"]), max(1, len(d["x"]) // 3000))
    ax.scatter(d["x"][idx], np.exp(d["y"][idx]), s=2, alpha=0.25, color=COL[n])
    xx = np.logspace(np.log10(max(0.2, d["x"].min())), np.log10(d["x"].max()), 200)
    ax.plot(xx, np.exp(d["icpt"] + d["slope"] * xx), "k-", lw=1.1,
            label=r"fit $m/T>1$: %.2f, $R^2$=%.2f" % (d["slope"], d["r2"]))
    ax.plot(xx, d["G"] / 4 / np.cosh(xx / 2) ** 2, "-", color="#d62728", lw=1.0,
            label=r"identity $G/4\cosh^{-2}(m/2T)$")
    if d["A3"] is not None:
        ax.plot(xx, np.exp(d["A3"][1] + d["A3"][0] * xx), "r--", lw=1.2,
                label=r"fit $m/T>3$: %.2f, $R^2$=%.2f" % (d["A3"][0], d["r23"]))
    ax.axvline(1, color="#888888", ls=":", lw=0.8)
    ax.axvline(3, color="#888888", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(n, fontsize=8.6)
    ax.set_xlabel(r"$m_t/T$", fontsize=7.6)
    ax.set_ylabel(r"$\sigma_{\max}(\Sigma_p)$", fontsize=7.6)
    ax.legend(fontsize=5.8)
    ax.grid(True, which="both", alpha=0.25)
fig.suptitle("The gate law per model: dashed vertical lines at $m/T=1$ and $3$; the random control is fitted both ways", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "pack_gate_law_per_model.png")

# ---------------------------------------------------------------- 3. sigma distributions
bins = np.array([0.05, 0.15, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0])
labels = []
data = []
for a, b in zip(bins[:-1], bins[1:]):
    vals = []
    for n in ["DeepSeek-1.5B", "Qwen3-4B", "Phi-4-mini", "randQwen"]:
        if n not in D:
            continue
        d = D[n]
        s = (d["x"] >= a) & (d["x"] < b)
        vals.extend(np.exp(d["y"][s]).tolist())
    if vals:
        data.append(vals); labels.append(f"{a:g}-{b:g}")
fig, ax = plt.subplots(figsize=(11.0, 4.8))
bp = ax.boxplot(data, patch_artist=True, showfliers=False)
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor(plt.cm.viridis(i / max(len(data) - 1, 1))); b.set_alpha(0.75)
ax.set_yscale("log")
ax.set_xticklabels(labels)
ax.set_xlabel(r"$m_t/T$ bin")
ax.set_ylabel(r"$\sigma_{\max}(\Sigma_p)$")
ax.set_title("Distribution of the readout gain at fixed $m_t/T$, four models pooled", loc="left")
ax.grid(True, which="both", alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_gate_sigma_distribution.png")

json.dump({n: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in d.items() if k not in ("x", "y")} for n, d in D.items()},
          open(os.path.join(HERE, "analysis_update", "pack_gate.json"), "w"), indent=1)
