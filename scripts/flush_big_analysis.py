"""Flush law across prompts and windows (run on the GPU host, data in results/big/flush)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results/big"
OUT = "results/figures_big"
os.makedirs(OUT, exist_ok=True)

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "flush", "*_w*"))):
    f = os.path.join(d, "a1.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f, allow_pickle=True)
    meta = json.load(open(os.path.join(d, "meta.json")))
    L = list(z["layers"]); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    P = int(z["prompt_ids"].shape[0]); D = 1536
    pmax = int((z["subspace"] // D).max())
    name = os.path.basename(d)
    prompt = name.split("_w")[0]
    W = int(name.split("_w")[1])
    for l in L:
        A = z["dh_norm"][ri, :, L.index(l), :]
        ts = []
        for di in range(A.shape[0]):
            zz = np.where(A[di] <= 0)[0]
            if len(zz):
                ts.append(int(zz[0]))
        t = int(np.median(ts)) if ts else -1
        rows.append(dict(prompt=prompt, W=W, P=P, pmax=pmax, layer=int(l), t=t,
                         pred1=pmax + (W - P), pred_l=pmax + (W - P) + (int(l) - 1) * (W - 1)))

np.savez(os.path.join(OUT, "flush_summary.npz"),
         **{k: np.array([r[k] for r in rows]) for k in rows[0]})
json.dump(rows, open(os.path.join(OUT, "flush_rows.json"), "w"), indent=1)

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))

ax = axs[0]
for prompt in sorted(set(r["prompt"] for r in rows)):
    sel = [r for r in rows if r["prompt"] == prompt and r["t"] > 0]
    ax.scatter([r["pred_l"] for r in sel], [r["t"] for r in sel], s=22,
               label=f"{prompt} (P={sel[0]['P']})")
lim = [50, 900]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted $t^*_\\ell = p_{last}+(W-P)+(\\ell-1)(W-1)$")
ax.set_ylabel("measured first-zero step")
ax.set_title("(a) The law across prompts and windows")
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

ax = axs[1]
res = [r["t"] - r["pred_l"] for r in rows if r["t"] > 0]
ax.hist(res, bins=20, color="tab:blue")
ax.set_xlabel("measured minus predicted step")
ax.set_ylabel("count")
ax.set_title(f"(b) Residuals: median {np.median(res):.0f}, 90% within "
             f"{np.percentile(np.abs(res), 90):.0f}")
ax.grid(alpha=0.25)

ax = axs[2]
for prompt in sorted(set(r["prompt"] for r in rows)):
    for W in sorted(set(r["W"] for r in rows if r["prompt"] == prompt)):
        sel = [r for r in rows if r["prompt"] == prompt and r["W"] == W and r["t"] > 0]
        if len(sel) < 2:
            continue
        base = sel[0]["t"]
        ax.plot([r["layer"] - sel[0]["layer"] for r in sel],
                [(r["t"] - base) / (W - 1) for r in sel], "o-", ms=3, lw=0.8, alpha=0.8)
ax.plot([0, 5], [0, 5], "k--", lw=1)
ax.set_xlabel("layer index $\\ell-\\ell_0$")
ax.set_ylabel("$(t^*_\\ell-t^*_{\\ell_0})/(W-1)$")
ax.set_title("(c) The cascade: one window per layer")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "flush_law_all.png"), dpi=190)
print("rows", len(rows), "residual median", float(np.median(res)),
      "p90", float(np.percentile(np.abs(res), 90)))
