"""Post-batch analysis: window regimes and stable chaos (run on the GPU host)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures_big"
os.makedirs(OUT, exist_ok=True)


def local_rate(A, k=10):
    """A: [dirs, T] -> [dirs, T-2k] local rates."""
    t = np.arange(k, A.shape[1] - k)
    return np.log(np.maximum(A[:, t + k], 1e-12) / np.maximum(A[:, t - k], 1e-12)) / (2 * k), t


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


# ---------------- regimes ----------------
rows = []
for d in sorted(glob.glob("results/big/regime/*_w*")):
    f = os.path.join(d, "a1.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f, allow_pickle=True)
    name = os.path.basename(d); prompt, W = name.split("_w"); W = int(W)
    L = list(z["layers"]); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    P = int(z["prompt_ids"].shape[0])
    A = np.maximum(z["dh_norm"][ri, :, L.index(27), :], 1e-12)
    lam, t = local_rate(A)
    pre = (t < (W - P) - 5)
    post = (t >= (W - P) + 5)
    for di in range(A.shape[0]):
        rows.append(dict(prompt=prompt, W=W, P=P, dir=di,
                         pre=float(lam[di, pre].mean()) if pre.any() else np.nan,
                         post=float(lam[di, post].mean()) if post.any() else np.nan))

if rows:
    json.dump(rows, open(os.path.join(OUT, "regime_rows.json"), "w"), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, title in [(axs[0], "pre", "pre-clip rate (t < W-P-5)"),
                           (axs[1], "post", "post-clip rate (t > W-P+5)")]:
        wins = sorted(set(r["W"] for r in rows))
        med, lo, hi = [], [], []
        for w in wins:
            v = [r[key] for r in rows if r["W"] == w and np.isfinite(r[key])]
            if not v:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            m = np.mean(v); ci = boot_ci(v)
            med.append(m); lo.append(ci[0]); hi.append(ci[1])
        ax.errorbar(wins, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                    fmt="o-", capsize=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("window $W$")
        ax.set_ylabel("mean $\\lambda_{10}$ (nats/token)")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "regime_rates.png"), dpi=190)
    print("regime rows", len(rows))

# ---------------- stable chaos ----------------
rows = []
for d in sorted(glob.glob("results/big/chaos/*_r*")):
    f = os.path.join(d, "free_run.npz")
    if not os.path.exists(f):
        continue
    fr = np.load(f, allow_pickle=True)
    name = os.path.basename(d); prompt, r = name.split("_r"); r = float(r)
    for row in fr["obs"]:
        rows.append(dict(prompt=prompt, r=r, obs=[int(x) for x in row]))

if rows:
    json.dump(rows, open(os.path.join(OUT, "chaos_rows.json"), "w"), indent=1)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    prompts = sorted(set(x["prompt"] for x in rows))
    for p in prompts:
        rr = sorted(set(x["r"] for x in rows if x["prompt"] == p))
        frac = []
        for r in rr:
            obs = [o for x in rows if x["prompt"] == p and x["r"] == r for o in x["obs"]]
            frac.append(np.mean([o >= 0 for o in obs]))
        ax.semilogx(rr, frac, "o-", label=p)
    ax.set_xlabel("perturbation radius $r$")
    ax.set_ylabel("fraction of directions that flip")
    ax.set_title("Flip fraction vs radius across prompts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chaos_flip_fraction.png"), dpi=190)
    print("chaos rows", len(rows))
