"""The depth-propagation story, with the three metrics on the same footing.

For one token t, layer l:
    A_l  = ||delta h_l||          absolute perturbation norm          (dh_norm)
    S_l  = ||h_l||                reference state norm                (norm_h0)
    R_l  = A_l / S_l              RELATIVE perturbation ("significance")

Identity used everywhere:
    R_l / R_1  =  (A_l / A_1) / (S_l / S_1)
so the relative growth is the absolute growth divided by the residual-stream norm
growth.  A perturbation can grow in absolute terms simply because the residual
stream grows; only R says whether it keeps up with the signal.

The tangent-map probe (run_depth2) gives a third, independent object:
    cum_l = log sigma_max(J_l ... J_0)   linearised amplification of an
                                          infinitesimal vector at one position
    g_l   = ||J_l v||                    per-block local gain
cum_l <= sum_l log g_l because the optimal direction rotates between blocks.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

CTRL = {"Mamba-130m": "results_big_small/ctrl_Mamba_130m.npz",
        "Pythia-410m": "results_big_small/ctrl_Pythia_410m.npz",
        "randQwen": "results_big_small/ctrl_randQwen.npz"}
ORDER = all_names() + list(CTRL)
R = 0.1

DEP = {"DeepSeek-1.5B": "results/depth2/deepseek/depth.npz",
       "DeepSeek-14B": "results_big_small/results_big/ds14/depth_w100/depth.npz",
       "Qwen2.5-3B": "results/depth2/qwen25_3b/depth.npz",
       "Qwen3-4B": "results/depth2/qwen3_4b/depth.npz",
       "Qwen3-14B": "results_big_small/results_big/qwen314/depth_w0/depth.npz",
       "Qwen3.5-2B": "results/depth2/qwen3_5/depth.npz",
       "Phi-4-mini": "results/depth2/phi4mini/depth.npz",
       "Phi-4": "results_big_small/results_big/phi4/depth_w0/depth.npz"}


def load_model_data(n, r=R):
    """med/lo/hi of A and S, plus the per-direction relative percentiles."""
    if n in SMALL:
        z = a1(n)
        ri = int(np.argmin(np.abs(z["radii"] - r)))
        dh = z["dh"][ri].astype(float)        # (dir, layer, T)
        hn = z["hn"].astype(float)            # (layer, T)
        med = np.median(dh, axis=0); lo, hi = np.percentile(dh, [25, 75], axis=0)
        rel = dh / np.maximum(hn, 1e-30)[None]
        rmed = np.median(rel, axis=0); rlo, rhi = np.percentile(rel, [25, 75], axis=0)
        lay = z["layers"].astype(float)
    elif n in BIG:
        z = big(n)
        med = z["dh_med"].astype(float); lo = z["dh_lo"].astype(float); hi = z["dh_hi"].astype(float)
        hn = z["norm_h0"].astype(float)
        rel = med / np.maximum(hn, 1e-30)
        rmed, rlo, rhi = rel, lo / np.maximum(hn, 1e-30), hi / np.maximum(hn, 1e-30)
        lay = z["layers"].astype(float)
    else:
        z = load(CTRL[n])
        ri = int(np.argmin(np.abs(z["radii"] - r)))
        med = z["dh_med"][ri].astype(float); lo = z["dh_lo"][ri].astype(float); hi = z["dh_hi"][ri].astype(float)
        hn = z["norm_h0"].astype(float)
        rel = med / np.maximum(hn, 1e-30)
        rmed, rlo, rhi = rel, lo / np.maximum(hn, 1e-30), hi / np.maximum(hn, 1e-30)
        lay = z["layers"].astype(float)
    return dict(A=med, Alo=lo, Ahi=hi, S=hn, R=rmed, Rlo=rlo, Rhi=rhi, lay=lay, T=med.shape[-1])


DATA = {}
for n in ORDER:
    try:
        DATA[n] = load_model_data(n)
    except Exception as e:
        print("skip", n, e)

# ---------------------------------------------------------------- P1: the three metrics, two models
fig, axes = plt.subplots(2, 4, figsize=(17.2, 7.6))
for row, n in enumerate(["DeepSeek-1.5B", "Qwen3-14B"]):
    d = DATA[n]
    t = min(50, d["T"] - 1)
    x = np.arange(len(d["lay"]))

    ax = axes[row, 0]
    ax.semilogy(x, d["A"][:, t], "o-", ms=3, color="#d62728")
    ax.set_ylabel(r"$A_\ell=\|\delta h_\ell\|$")
    ax.set_title(f"{n}: (a) absolute perturbation", loc="left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[row, 1]
    ax.semilogy(x, d["S"][:, t], "o-", ms=3, color="#1f77b4")
    ax.set_ylabel(r"$S_\ell=\|h_\ell\|$")
    ax.set_title(f"(b) reference state norm", loc="left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[row, 2]
    ax.semilogy(x, d["R"][:, t], "o-", ms=3, color="#2ca02c")
    ax.set_ylabel(r"$R_\ell=\|\delta h_\ell\|/\|h_\ell\|$")
    ax.set_title("(c) relative perturbation = (a)/(b)", loc="left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[row, 3]
    p = DEP.get(n)
    if p and os.path.exists(os.path.join(HERE, p)):
        z = np.load(os.path.join(HERE, p), allow_pickle=True)
        lam = z["lam"].astype(float); L = lam.shape[1]
        cum = lam * (np.arange(L) + 1)[None, :, None]
        cm = np.median(cum, axis=(0, 2))
        g = z["local_gain"].astype(float)
        sg = np.log(np.maximum(np.median(g, axis=0), 1e-30))
        ax.plot(np.arange(L), cm, "-", color="#9467bd", lw=1.5, label=r"$\log\sigma_{\max}(\prod J)$")
        ax.plot(np.arange(L), np.cumsum(sg), "--", color="#8c564b", lw=1.2,
                label=r"$\sum_\ell \log g_\ell$")
        ax.set_ylabel("log amplification (tangent map)")
        ax.legend(fontsize=6.4, loc="upper left")
    ax.set_title("(d) linearised: cum vs sum of local gains", loc="left", fontsize=9)
    ax.grid(alpha=0.3)
    for j in range(4):
        axes[row, j].set_xlabel("block $\\ell$", fontsize=7.6)
        axes[row, j].tick_params(labelsize=7)
fig.suptitle("What each depth metric measures (token $t=50$, $r=0.1$): absolute, state norm, relative, and linearised", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "pack_depth_story.png")

# ---------------------------------------------------------------- P2: relative growth, all models
fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.0))
ax = axes[0]
for n in ORDER:
    d = DATA.get(n)
    if d is None:
        continue
    t = min(50, d["T"] - 1)
    Rn = d["R"][:, t]
    if Rn[1] <= 0:
        continue
    x = (d["lay"] - d["lay"][0]) / (d["lay"][-1] - d["lay"][0])
    ax.plot(x[1:], Rn[1:] / Rn[1], "o-", ms=2.6, lw=1.3, color=COL[n],
            ls="--" if n in CTRL else "-", label=n)
ax.axhline(1, color="k", ls=":", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel(r"normalized depth  $(\ell-\ell_1)/(\ell_L-\ell_1)$")
ax.set_ylabel(r"$R_\ell/R_1$   (relative perturbation, layer 1 = 1)")
ax.set_title("(a) Does the perturbation keep up with the state?", loc="left")
ax.legend(fontsize=5.8, ncol=3); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
names, vals, lo_, hi_ = [], [], [], []
for n in ORDER:
    d = DATA.get(n)
    if d is None:
        continue
    t = min(50, d["T"] - 1)
    r1 = d["R"][1, :]
    rL = d["R"][-1, :]
    tmax = min(60, d["T"])
    keep = np.zeros(d["T"], bool); keep[2:tmax] = True
    ok = (r1 > 1e-9) & (rL > 1e-9) & keep
    if ok.sum() < 5:
        continue
    ratio_tok = rL[ok] / r1[ok]
    names.append(n)
    vals.append(float(np.median(ratio_tok)))
    lo_.append(float(np.percentile(ratio_tok, 10)))
    hi_.append(float(np.percentile(ratio_tok, 90)))
    # keep the t=50 value for the JSON
order = np.argsort(vals)
y = np.arange(len(names))
for i, k in enumerate(order):
    ax.barh(i, vals[k], color=COL[names[k]], alpha=0.85,
            hatch="//" if names[k] in CTRL else None)
    ax.plot([lo_[k], hi_[k]], [i, i], color="k", lw=1.0)
ax.axvline(1, color="k", lw=1.0)
ax.set_yticks(y); ax.set_yticklabels([names[k] + ("  (control)" if names[k] in CTRL else "") for k in order],
                                     fontsize=7.4)
ax.set_xscale("log")
ax.set_xlabel(r"$R_L/R_1$ (last / first block); bar = 10--90% over tokens")
ax.set_title("(b) Net relative change across the stack", loc="left")
ax.grid(True, which="both", alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "pack_depth_relative_all.png")

json.dump({n: dict(R_ratio=float(vals[names.index(n)]),
                   lo=float(lo_[names.index(n)]), hi=float(hi_[names.index(n)]),
                   A_ratio=float(DATA[n]["A"][-1, min(50, DATA[n]["T"] - 1)] /
                                 max(DATA[n]["A"][1, min(50, DATA[n]["T"] - 1)], 1e-30)),
                   S_ratio=float(DATA[n]["S"][-1, min(50, DATA[n]["T"] - 1)] /
                                 max(DATA[n]["S"][1, min(50, DATA[n]["T"] - 1)], 1e-30)))
           for n in names},
          open(os.path.join(HERE, "analysis_update", "pack_depth_relative_all.json"), "w"), indent=1)

# ---------------------------------------------------------------- P3: per-model relative profiles with IQR
fig, axes = plt.subplots(4, 4, figsize=(16.4, 12.6))
for ax, n in zip(axes.flat, ORDER):
    d = DATA.get(n)
    if d is None:
        ax.axis("off"); continue
    t = min(50, d["T"] - 1)
    Rn, Rlo, Rhi = d["R"][:, t], d["Rlo"][:, t], d["Rhi"][:, t]
    if Rn[1] <= 0:
        ax.axis("off"); continue
    x = (d["lay"] - d["lay"][0]) / (d["lay"][-1] - d["lay"][0])
    ax.plot(x[1:], Rn[1:] / Rn[1], "o-", ms=2.6, lw=1.4, color=COL[n])
    ax.fill_between(x[1:], Rlo[1:] / Rn[1], Rhi[1:] / Rn[1], color=COL[n], alpha=0.18, lw=0)
    ax.axhline(1, color="k", ls=":", lw=0.9)
    ax.set_yscale("log")
    ax.set_title(n + ("  (control)" if n in CTRL else ""), fontsize=8.6)
    ax.set_xlabel(r"$\ell/(L-1)$", fontsize=7.4)
    ax.set_ylabel(r"$R_\ell/R_1$", fontsize=7.4)
    ax.tick_params(labelsize=6.8)
    ax.grid(True, which="both", alpha=0.25)
for j in range(len(ORDER), 16):
    axes.flat[j].axis("off")
fig.suptitle("Relative perturbation vs depth, every model and control (band = IQR over directions, $t=50$)", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "pack_depth_relative_grid.png")

# ---------------------------------------------------------------- P4: the propagation ladder (layer x token)
MODS = ["DeepSeek-1.5B", "Qwen3-14B", "Phi-4-mini", "randQwen"]
fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.6))
for ax, n in zip(axes, MODS):
    d = DATA.get(n)
    if d is None:
        ax.axis("off"); continue
    R = d["R"][1:]                                   # (layer-1, T)
    scale = np.median(d["R"][1][d["R"][1] > 0]) if np.any(d["R"][1] > 0) else 1.0
    Z = np.log10(np.maximum(R / scale, 1e-6))
    vmin, vmax = np.percentile(Z, [2, 98])
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=(0, d["T"], d["lay"][1], d["lay"][-1]))
    ax.set_xlabel("generated token $t$")
    ax.set_ylabel("layer $\\ell$")
    ax.set_title(n, fontsize=9)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}(R_\ell(t))$", fontsize=7.4)
fig.suptitle("Relative perturbation $\\|\\delta h_\\ell\\|/\\|h_\\ell\\|$ over the whole (token, layer) plane", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "pack_depth_ladder.png")
print("done")
for n in names:
    d = DATA[n]; t = min(50, d["T"] - 1)
    print("%-14s R_L/R_1=%5.2f  A_L/A_1=%7.1f  S_L/S_1=%7.1f" % (
        n, d["R"][-1, t] / max(d["R"][1, t], 1e-30),
        d["A"][-1, t] / max(d["A"][1, t], 1e-30),
        d["S"][-1, t] / max(d["S"][1, t], 1e-30)))
