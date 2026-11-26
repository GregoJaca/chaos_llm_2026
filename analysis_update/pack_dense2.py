import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

PAIRS = {
    "DeepSeek-1.5B": ("results/local_data/w100_layers/a1.npz", "results/update/dense_ds15/a1.npz"),
    "Qwen2.5-3B": ("results/update/fp32_qwen25_a1.npz", "results/update/dense_qwen25/a1.npz"),
    "Qwen3.5-2B": ("results/update/fp32_qwen35_a1.npz", "results/update/dense_qwen35/a1.npz"),
}

def load_curve(path):
    z = np.load(os.path.join(HERE, path), allow_pickle=True)
    r = z["radii"].astype(float)
    y = z["dh_norm"][:, :, -1, 50].astype(float)
    med = np.median(y, axis=1); lo, hi = np.percentile(y, [25, 75], axis=1)
    return r, med, lo, hi

def rstar_curve(path):
    z = np.load(os.path.join(HERE, path), allow_pickle=True)
    r = z["radii"].astype(float); m = z["margins"].astype(float)
    med, lo, hi = [], [], []
    for i in range(len(r)):
        rc = z["r_c"][i].copy(); rc[:, m <= 0.02] = np.inf
        v = np.min(rc, axis=1); v = v[np.isfinite(v)]
        med.append(np.median(v))
        bs = [np.median(np.random.default_rng(0).choice(v, len(v))) for _ in range(200)]
        lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    return r, np.array(med), np.array(lo), np.array(hi)

fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))
for ax, (n, (ps, pd)) in zip(axes, PAIRS.items()):
    rs, ms, ls, hs = load_curve(ps); rd, md, ld, hd = load_curve(pd)
    ax.plot(rs, ms, "o--", ms=4, color=COL[n], alpha=0.55, label="sparse simplex")
    ax.fill_between(rs, ls, hs, color=COL[n], alpha=0.1, lw=0)
    ax.plot(rd, md, "o-", ms=5, color=COL[n], label="dense Gaussian")
    ax.fill_between(rd, ld, hd, color=COL[n], alpha=0.2, lw=0)
    sel = (rs >= 0.03) & (rs <= 0.3); b1 = np.polyfit(np.log(rs[sel]), np.log(ms[sel]), 1)[0]
    sel2 = (rd >= 0.03) & (rd <= 0.3); b2 = np.polyfit(np.log(rd[sel2]), np.log(md[sel2]), 1)[0]
    ax.text(0.03, 0.95, r"$\beta_{\rm sparse}$=%.2f, $\beta_{\rm dense}$=%.2f" % (b1, b2),
            transform=ax.transAxes, fontsize=7.2, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\|\delta h_{50,L}\|$")
    ax.set_title(n, fontsize=9)
    ax.legend(fontsize=6.8); ax.grid(True, which="both", alpha=0.3)
fig.suptitle("Sparse vs dense directions: same slope, a constant shift; both linear over 0.03-0.3", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "pack_dense_slopes.png")

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
ax = axes[0]
for n, (ps, pd) in PAIRS.items():
    rs, ms, ls, hs = rstar_curve(ps); rd, md, ld, hd = rstar_curve(pd)
    ax.plot(rs, ms, "o--", ms=4, color=COL[n], alpha=0.55)
    ax.fill_between(rs, ls, hs, color=COL[n], alpha=0.1, lw=0)
    ax.plot(rd, md, "o-", ms=5, color=COL[n], label=n + " dense")
    ax.fill_between(rd, ld, hd, color=COL[n], alpha=0.2, lw=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"probe radius $r$")
ax.set_ylabel(r"median $r_*$")
ax.set_title("(a) $r_*$ vs probe radius (95% bootstrap band)", loc="left")
ax.legend(fontsize=6.8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
data, labels = [], []
for n, (ps, pd) in PAIRS.items():
    z = np.load(os.path.join(HERE, pd), allow_pickle=True)
    i = 0  # r=0.03
    m = z["margins"].astype(float); rc = z["r_c"][i].copy(); rc[:, m <= 0.02] = np.inf
    v = np.min(rc, axis=1); v = v[np.isfinite(v)]
    data.append(v); labels.append(n)
bp = ax.violinplot(data, showextrema=False, widths=0.8)
for i, b in enumerate(bp["bodies"]):
    b.set_facecolor(COL[labels[i]]); b.set_alpha(0.55)
for i, n in enumerate(labels):
    ax.plot(i + 1, np.median(data[i]), "o", color="white", mec=COL[n], mew=2, ms=6)
ax.set_yscale("log")
ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"$r_*$ at $r=0.03$ (dense)")
ax.set_title("(b) Dense $r_*$ distributions", loc="left")
ax.grid(True, which="both", alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_rstar_dense2.png")
print("done")
