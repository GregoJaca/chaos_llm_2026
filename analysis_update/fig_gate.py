import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Qwen3.5-9B",
          "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen"]

data = {}
for n in MODELS:
    xs, ys, ts = [], [], []
    for rr in softloop_runs(n, mode="loop"):
        z = rr["data"]
        if "m_ref" not in z or "sig_read" not in z:
            continue
        m = z["m_ref"].astype(float)
        s = z["sig_read"].astype(float)
        ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
        if not ok.any():
            continue
        xs.append(m[ok] / rr["T"]); ys.append(np.log(s[ok]))
        ts.append(np.full(ok.sum(), rr["T"]))
    if not xs:
        continue
    x = np.concatenate(xs); y = np.concatenate(ys); T = np.concatenate(ts)
    sel = x > 1.0
    slope, icpt, r2 = loglog_slope(x[sel], y[sel])
    # bootstrap CI on the tail slope
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(300):
        idx = rng.integers(0, sel.sum(), sel.sum())
        bs.append(np.polyfit(x[sel][idx], y[sel][idx], 1)[0])
    data[n] = dict(x=x, y=y, T=T, slope=slope, icpt=icpt, r2=r2,
                   lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                   G=float(np.exp(icpt)), n=int(sel.sum()),
                   xmed=float(np.median(x)))
    print(f"{n:16s} n={sel.sum():5d} slope={slope:+.3f} [{data[n]['lo']:+.3f},{data[n]['hi']:+.3f}] "
          f"G={data[n]['G']:.2f} R2={r2:.3f} x_med={np.median(x):.2f}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.4))

# (a) DeepSeek: the raw cloud, colour = T
ax = axes[0, 0]
d = data["DeepSeek-1.5B"]
idx = np.arange(0, len(d["x"]), 3)
sc = ax.scatter(d["x"][idx], d["y"][idx], c=np.log10(d["T"][idx]), s=4, alpha=0.35, cmap="viridis")
xx = np.logspace(-1.3, 2, 400)
for G, ls in [(2.0, "-"), (6.0, "--")]:
    ax.plot(xx, np.log(G) - np.log(4) - 2 * np.log(np.cosh(xx / 2)), color="#d62728", ls=ls, lw=1.3,
            label=f"two-token identity, $G$={G:g}")
ax.axhline(np.log(2.0 / 4), color="#d62728", lw=0.7, ls=":")
ax.set_xscale("log")
ax.set_xlim(0.05, 100)
ax.set_ylim(-8, 1.5)
ax.set_xlabel(r"margin-to-temperature ratio  $m_t/T$  (log scale)")
ax.set_ylabel(r"$\log\sigma_{\max}(\Sigma_p)$")
ax.set_title("(a) DeepSeek-1.5B, all temperatures: the softmax curve", loc="left")
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(r"$\log_{10} T$")
ax.legend(fontsize=7.0, loc="lower left")
ax.grid(True, which="both", alpha=0.3)

# (b) per-model tail slope
ax = axes[0, 1]
labels = [n for n in MODELS if n in data]
yv = np.arange(len(labels))
for i, n in enumerate(labels):
    d = data[n]
    col = COL.get(n, "k")
    ax.errorbar(d["slope"], i, xerr=[[d["slope"] - d["lo"]], [d["hi"] - d["slope"]]],
                fmt="o", color=col, ms=6, capsize=3, lw=1.6)
    ax.text(0.18, i, f"G={d['G']:.1f}  $R^2$={d['r2']:.2f}", va="center", fontsize=6.8, color=col)
ax.axvline(-1, color="k", ls="--", lw=1.0)
ax.set_yticks(yv); ax.set_yticklabels(labels, fontsize=7.8)
ax.set_xlabel(r"slope of $\log\sigma_{\max}$ vs $m_t/T$ for $m_t/T>1$  (theory: $-1$)")
ax.set_title("(b) Slope is $-1$ for every model, including the random control", loc="left")
ax.set_xlim(-1.25, 0.25)
ax.grid(alpha=0.3, axis="x")

# (c) binned median and IQR: collapse across models
ax = axes[1, 0]
bins = np.array([0.1, 0.2, 0.35, 0.6, 1.0, 1.7, 3.0, 5.0, 9.0])
for n in ["DeepSeek-1.5B", "Qwen3-4B", "Phi-4-mini", "randQwen"]:
    if n not in data:
        continue
    d = data[n]
    cx, med, lo, hi = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = (d["x"] >= a) & (d["x"] < b)
        if s.sum() < 8:
            continue
        cx.append(np.sqrt(a * b)); med.append(np.median(d["y"][s]))
        lo.append(np.percentile(d["y"][s], 25)); hi.append(np.percentile(d["y"][s], 75))
    ax.plot(cx, med, "-o", color=COL[n], ms=4, label=n)
    ax.fill_between(cx, lo, hi, color=COL[n], alpha=0.18, lw=0)
ax.set_xscale("log")
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10"])
ax.set_xlabel(r"$m_t/T$  (log scale)")
ax.set_ylabel(r"median $\log\sigma_{\max}$  (band = IQR)")
ax.set_title("(c) The same curve across architectures; random control agrees", loc="left")
ax.legend(fontsize=7.4)
ax.grid(True, which="both", alpha=0.3)

# (d) distribution of sigma at fixed x bins (DeepSeek)
ax = axes[1, 1]
d = data["DeepSeek-1.5B"]
edges = [0.0, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0]
vals, labs = [], []
for a, b in zip(edges[:-1], edges[1:]):
    s = (d["x"] >= a) & (d["x"] < b)
    if s.sum() >= 10:
        vals.append(d["y"][s]); labs.append(f"{a:g}-{b:g}\n({s.sum()})")
bp = ax.boxplot(vals, patch_artist=True, showfliers=False)
for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(plt.cm.viridis(i / max(len(vals) - 1, 1)))
    box.set_alpha(0.75)
ax.set_xticklabels(labs, fontsize=7)
ax.set_xlabel(r"$m_t/T$ bin  (number of steps)")
ax.set_ylabel(r"$\log\sigma_{\max}(\Sigma_p)$")
ax.set_title("(d) Distribution of the readout gain at fixed $m_t/T$ (DeepSeek)", loc="left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("The gate law: slope $-1$ is the softmax identity; the measured content is $G_{\\rm eff}$", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_gate_law.png")

json.dump({n: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in d.items() if k not in ("x", "y", "T")}
           for n, d in data.items()},
          open(os.path.join(HERE, "analysis_update", "gate_numbers.json"), "w"), indent=1)
print("saved gate_numbers.json")

