import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

DEPTH = {
    "DeepSeek-1.5B": "results/depth2/deepseek/depth.npz",
    "DeepSeek-14B": "results_big_small/results_big/ds14/depth_w100/depth.npz",
    "Qwen2.5-3B": "results/depth2/qwen25_3b/depth.npz",
    "Qwen3-4B": "results/depth2/qwen3_4b/depth.npz",
    "Qwen3-14B": "results_big_small/results_big/qwen314/depth_w0/depth.npz",
    "Qwen3.5-2B": "results/depth2/qwen3_5/depth.npz",
    "Phi-4-mini": "results/depth2/phi4mini/depth.npz",
    "Phi-4": "results_big_small/results_big/phi4/depth_w0/depth.npz",
}
ORDER = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Phi-4-mini",
         "DeepSeek-14B", "Qwen3-14B", "Phi-4"]

D = {}
for n, p in DEPTH.items():
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    lam = z["lam"].astype(float)          # (tok, L, k) = cum/(l+1)
    L = lam.shape[1]
    ell = np.arange(L)
    cum = lam * (ell + 1)[None, :, None]
    cum_tok = np.median(cum, axis=2)      # (tok, L)
    D[n] = dict(lam=lam, cum=np.median(cum, axis=(0, 2)),
                cum_lo=np.percentile(cum_tok, 25, axis=0),
                cum_hi=np.percentile(cum_tok, 75, axis=0),
                local=np.median(z["local_gain"].astype(float), axis=0),
                L=L, tok=z["tokens"].astype(int))

# ---------------------------------------------------------------- 1. cum + local
fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))
ax = axes[0]
for n in ORDER:
    x = np.arange(D[n]["L"]) / (D[n]["L"] - 1)
    ax.plot(x, D[n]["cum"], color=COL[n], label=n)
    ax.fill_between(x, D[n]["cum_lo"], D[n]["cum_hi"], color=COL[n], alpha=0.12, lw=0)
ax.set_xlabel("normalized depth  $\\ell/(L-1)$")
ax.set_ylabel(r"$\log$ product gain  $\mathrm{cum}_\ell$")
ax.set_title("(a) Depth gain, all models (band = IQR over probe tokens)", loc="left")
ax.legend(fontsize=6.4, ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
for n in ORDER:
    ax.plot(np.arange(D[n]["L"]), D[n]["local"], color=COL[n], label=n)
ax.axhline(1, color="k", ls=":", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("block index $\\ell$")
ax.set_ylabel(r"local gain $g_\ell=\langle\|J_\ell v_i\|\rangle_i$")
ax.set_title("(b) Block 0 vs the rest", loc="left")
ax.legend(fontsize=6.4, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
for n in ORDER:
    g = D[n]["local"]
    vals = [g[0], np.median(g[len(g)//4:3*len(g)//4]), np.median(g[-max(1,len(g)//10):])]
    ax.plot([0, 1, 2], vals, "o-", color=COL[n], label=n)
ax.set_yscale("log")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["block 0", "middle", "last 10%"])
ax.set_ylabel("local gain")
ax.set_title("(c) The profile is not a single exponential", loc="left")
ax.legend(fontsize=6.2, ncol=2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_depth_all.png")

# ---------------------------------------------------------------- 2. per-token
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
for i, n in enumerate(ORDER):
    ax = axes.flat[i]
    z = np.load(os.path.join(HERE, DEPTH[n]), allow_pickle=True)
    lam = z["lam"].astype(float)
    L = lam.shape[1]
    cum = lam * (np.arange(L) + 1)[None, :, None]
    for ti in range(cum.shape[0]):
        ax.plot(np.arange(L), np.median(cum[ti], axis=1), lw=1.0, alpha=0.8)
    ax.set_title(f"{n} ({cum.shape[0]} tokens)", fontsize=8.5)
    ax.set_xlabel("block $\\ell$", fontsize=7.6)
    ax.set_ylabel(r"$\mathrm{cum}_\ell$", fontsize=7.6)
    ax.grid(alpha=0.25)
fig.suptitle("Depth gain for every probe token: the shape is stable, the scale is not", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_depth_multitoken.png")

# ---------------------------------------------------------------- 3. relative response
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
for i, n in enumerate(ORDER):
    ax = axes.flat[i]
    if n in SMALL:
        z = a1(n); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
        t = min(50, z["rel"].shape[-1] - 1)
        med = z["rel"][ri, :, t]
        lo, hi = np.percentile(z["dh"][ri, :, :, t], [25, 75], axis=0) / np.maximum(z["hn"][:, t], 1e-30)
        base = max(med[1], 1e-30)
        med, lo, hi = med[1:] / base, lo[1:] / base, hi[1:] / base
        x = np.arange(len(med)) / max(len(med) - 1, 1)
    else:
        z = big(n)
        med = z["rel_med"][:, 50]
        med = med[1:] / max(med[1], 1e-30)
        x = np.arange(len(med)) / max(len(med) - 1, 1)
        lo = hi = None
    ax.plot(x, med, color=COL[n])
    if lo is not None:
        ax.fill_between(x, lo, hi, color=COL[n], alpha=0.2, lw=0)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(n, fontsize=8.5)
    ax.set_xlabel("$\\ell/(L-1)$", fontsize=7.6)
    ax.set_ylabel(r"$\|\delta h_\ell\|/\|h_\ell\|$ / layer 1", fontsize=7.6)
    ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Relative response: does the perturbation become more important with depth?", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_depth_relative.png")

# ---------------------------------------------------------------- 4. fits (exploration)
def fit_models(x, y):
    """Return dict name -> (pred, k)."""
    out = {}
    n = len(x)
    # linear in x
    A = np.polyfit(x, y, 1); out["linear"] = (np.polyval(A, x), 2)
    # log
    A = np.polyfit(np.log(x + 1), y, 1); out["log"] = (np.polyval(A, np.log(x + 1)), 2)
    # sqrt
    A = np.polyfit(np.sqrt(x + 1), y, 1); out["sqrt"] = (np.polyval(A, np.sqrt(x + 1)), 2)
    # transient + linear: a + b*x + c*exp(-x/tau), grid over tau
    best = None
    for tau in np.linspace(0.5, 20, 60):
        X = np.column_stack([np.ones(n), x, np.exp(-x / tau)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        rss = np.sum((y - pred) ** 2)
        if best is None or rss < best[0]:
            best = (rss, pred, 4, tau)
    out["transient+linear"] = (best[1], best[2])
    return out

rows = []
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
for i, n in enumerate(ORDER):
    ax = axes.flat[i]
    x = np.arange(D[n]["L"], dtype=float)
    y = D[n]["cum"]
    fits = fit_models(x, y)
    for k, (pred, kk) in fits.items():
        rss = np.sum((y - pred) ** 2)
        bic = len(y) * np.log(max(rss / len(y), 1e-30)) + kk * np.log(len(y))
        rows.append((n, k, bic, 1 - rss / np.sum((y - y.mean()) ** 2)))
    ax.plot(x, y, "k.", ms=3, label="data")
    ax.plot(x, fits["linear"][0], "-", lw=1.2, label="linear in $\\ell$")
    ax.plot(x, fits["transient+linear"][0], "--", lw=1.4, label="transient + linear")
    ax.set_title(n, fontsize=8.5)
    ax.set_xlabel("block $\\ell$", fontsize=7.6)
    ax.set_ylabel(r"$\mathrm{cum}_\ell$", fontsize=7.6)
    ax.grid(alpha=0.25)
    if i == 0:
        ax.legend(fontsize=5.8)
fig.suptitle("What can be fitted? A block-0 transient plus a near-linear tail (AIC comparison below)", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_depth_fits.png")

json.dump([dict(model=a, form=b, bic=float(c), r2=float(d)) for a, b, c, d in rows],
          open(os.path.join(HERE, "analysis_update", "pack_depth_fits.json"), "w"), indent=1)
print("done")
