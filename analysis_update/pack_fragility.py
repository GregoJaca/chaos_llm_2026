import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

ORDER = all_names()
RTAG = {v: k for k, v in RSTAR_SMALL.items()}
S = {n: rstar(n) for n in ORDER}
for n in ORDER:
    print(n, "n", len(S[n]["r"]), "median", np.median(S[n]["r"]))

# ---------------------------------------------------------------- 1. ECDF + PDF
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
ax = axes[0]
for n in ORDER:
    x, F = ecdf(S[n]["r"])
    ax.plot(x, 1 - F, color=COL[n], lw=1.6, label=n)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"critical radius $r_*$  (perturbation norm)")
ax.set_ylabel(r"$P(r_*>x)$")
ax.set_title("(a) Fragility distribution, all models", loc="left")
ax.legend(fontsize=6.4, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
bins = np.linspace(-2.5, 1.5, 55)
for n in ORDER:
    r = S[n]["r"]; r = r[np.isfinite(r) & (r > 0)]
    ax.hist(np.log10(r), bins=bins, histtype="step", lw=1.4, density=True,
            color=COL[n], label=n)
ax.set_xlabel(r"$\log_{10} r_*$")
ax.set_ylabel(r"probability density  $p(\log_{10}r_*)$")
ax.set_title("(b) Density of the critical radius", loc="left")
ax.legend(fontsize=6.2, ncol=2); ax.grid(alpha=0.3)

ax = axes[2]
data = [np.log10(S[n]["r"][np.isfinite(S[n]["r"])]) for n in ORDER]
bp = ax.violinplot(data, showextrema=False, widths=0.85)
for i, b in enumerate(bp["bodies"]):
    b.set_facecolor(COL[ORDER[i]]); b.set_alpha(0.55)
for i, n in enumerate(ORDER):
    med = np.median(data[i])
    ax.plot(i + 1, med, "o", color="white", mec=COL[n], mew=2, ms=6)
ax.set_xticks(range(1, len(ORDER) + 1))
ax.set_xticklabels([n.replace("DeepSeek", "DS") for n in ORDER], rotation=25, ha="right", fontsize=7.4)
ax.set_ylabel(r"$\log_{10} r_*$")
ax.set_title("(c) Same, as distributions (white dot = median)", loc="left")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_rstar_all.png")

# ---------------------------------------------------------------- 2. decomposition
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
ax = axes[0]
for n in ORDER:
    d = S[n]
    ok = np.isfinite(d["r"]) & np.isfinite(d["m"]) & (d["m"] > 0)
    ax.scatter(d["m"][ok], d["r"][ok], s=3, alpha=0.25, color=COL[n], label=n)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"margin at the controlling step  $m_{t_*}$  (nats)")
ax.set_ylabel(r"$r_*$")
ax.set_title(r"(a) $r_*=m_{t_*}/s_{t_*}$: the margin sets the scale", loc="left")
ax.legend(fontsize=6.2, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for n in ORDER:
    d = S[n]
    ok = np.isfinite(d["r"]) & np.isfinite(d["s"]) & (d["s"] > 0)
    ax.scatter(d["s"][ok], d["r"][ok], s=3, alpha=0.25, color=COL[n])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"local sensitivity at the controlling step  $s_{t_*}=|\delta m_{t_*}|/r$")
ax.set_ylabel(r"$r_*$")
ax.set_title(r"(b) The spread of $r_*$ at fixed margin is $1/s$", loc="left")
ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
labels, vals = [], []
for n in ORDER:
    d = S[n]
    ok = np.isfinite(d["m"]) & np.isfinite(d["r"]) & (d["m"] > 0) & (d["r"] > 0)
    vals.append(np.corrcoef(np.log(d["m"][ok]), np.log(d["r"][ok]))[0, 1])
    labels.append(n)
y = np.arange(len(labels))
ax.barh(y, vals, color=[COL[n] for n in labels], alpha=0.85)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.4)
ax.set_xlim(0, 1.05); ax.set_xlabel(r"Pearson $r(\log m_{t_*},\log r_*)$")
ax.set_title("(c) Margin explains most of the fragility", loc="left")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "pack_rstar_decomposition.png")

# ---------------------------------------------------------------- 3. decoupling
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.6))
for i, n in enumerate(ORDER[:8]):
    ax = axes.flat[i]
    gs, rs = [], []
    if n in SMALL:
        for pi in range(1, 6):
            z = load(f"results/rstar/{RTAG[n]}/p{pi}/a1.npz")
            r0 = float(z["radii"][0])
            G = z["dh_norm"][0, :, -1, 50].astype(float) / r0
            m = z["margins"].astype(float)
            rc = z["r_c"][0].astype(float); rc[:, m <= 0.02] = np.inf
            rr = np.min(rc, axis=1)
            ok = np.isfinite(rr) & (rr > 0) & np.isfinite(G) & (G > 0)
            gs.append(G[ok]); rs.append(rr[ok])
    else:
        for tag in big_tags(n):
            z = big(n, tag)
            G = z["G_u"].astype(float)
            m = z["margins"].astype(float)
            r0 = float(z["radii"][0])
            rc = np.where(z["dm"] / r0 < 0, m[None] * r0 / np.maximum(-z["dm"], 1e-30), np.inf)
            rc[:, m <= 0.02] = np.inf
            rr = np.min(rc, axis=1)
            ok = np.isfinite(rr) & (rr > 0) & np.isfinite(G) & (G > 0)
            gs.append(G[ok]); rs.append(rr[ok])
    g = np.concatenate(gs); r = np.concatenate(rs)
    ax.scatter(g, r, s=4, alpha=0.35, color=COL[n])
    rho = np.corrcoef(np.log(g), np.log(r))[0, 1]
    ax.set_title(f"{n}\n" + r"$r=%.3f$" % rho, fontsize=8.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$G(u)=\|\delta h_L\|/r$", fontsize=8)
    ax.set_ylabel(r"$r_*(u)$", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
for j in range(len(ORDER), 8):
    axes.flat[j].axis("off")
fig.suptitle("Bulk gain and discrete fragility are decoupled: $r_*$ is not predicted by $G$", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_rstar_vs_gain.png")

# ---------------------------------------------------------------- 4. bottleneck
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.2))
for i, n in enumerate(ORDER[:8]):
    ax = axes.flat[i]
    d = S[n]
    j = d["t"][np.isfinite(d["t"])].astype(int)
    c = np.bincount(j, minlength=300)
    top = np.argsort(c)[-5:]
    share = c[top].sum() / max(c.sum(), 1)
    ax.bar(np.arange(len(c)), c, width=1.0, color=COL[n])
    ax.set_title(f"{n}: top-5 = {share:.0%}", fontsize=8.4)
    ax.set_xlabel("controlling step $t_*$", fontsize=7.6)
    ax.set_ylabel("directions", fontsize=7.6)
    ax.set_xlim(0, 120)
    ax.grid(alpha=0.25)
for j in range(len(ORDER), 8):
    axes.flat[j].axis("off")
fig.suptitle("A few decision steps control most of the flips (first 120 generated tokens shown)", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_rstar_bottleneck.png")

# save numbers
out = {n: dict(median=float(np.median(S[n]["r"])),
               p10=float(np.percentile(S[n]["r"], 10)),
               p90=float(np.percentile(S[n]["r"], 90)),
               n=int(len(S[n]["r"]))) for n in ORDER}
json.dump(out, open(os.path.join(HERE, "analysis_update", "pack_rstar.json"), "w"), indent=1)
