"""The flux-tube radius r* over the full direction space (1536 directions) + survival of flips."""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from chaos_llm.curvefit import fit_curve

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures", "fsle")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})
NUM = {}

d = np.load(os.path.join(LD, "all_dirs.npz"), allow_pickle=True)
rc = d["r_c"][0]                       # [1536, 300]
m = d["margins"].astype(float)
rstar = rc.min(axis=1)
ok = np.isfinite(rstar) & (rstar > 0)
rs = np.sort(rstar[ok])
ccdf = 1 - np.arange(len(rs)) / len(rs)
NUM["n_dirs"] = int(ok.sum())
NUM["rstar_q"] = np.quantile(rstar[ok], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()

# ---- fits of the CCDF
fits = {}
for key, yv in [("ccdf", ccdf)]:
    r = fit_curve(rs, yv)
    fits[key] = r
    NUM["ccdf_best"] = r.best.name
    NUM["ccdf_params"] = dict(zip(r.best.pnames, r.best.params.tolist()))
    NUM["ccdf_r2"] = r.best.r2
    NUM["ccdf_second"] = r.fits[1].name
    NUM["ccdf_dbic"] = float(r.fits[1].bic - r.best.bic)

fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axs[0]
ax.loglog(rs, ccdf, ".", ms=3, color="0.35", label=f"data (n={ok.sum()})")
xx = np.geomspace(rs.min(), rs.max(), 400)
r = fits["ccdf"]
ax.loglog(xx, np.maximum(r.best.predict(xx), 1e-4), "-", lw=1.8, color="tab:red",
          label=f"best: {r.best.name}")
ax.loglog(xx, np.maximum(r.fits[1].predict(xx), 1e-4), "--", lw=1.2, color="tab:blue",
          label=f"2nd: {r.fits[1].name}")
ax.set_xlabel(r"$r_*$"); ax.set_ylabel(r"$P(r_*>x)$")
ax.set_title("(a) CCDF of the flip radius, 1536 directions")
ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")

ax = axs[1]
ax.hist(np.log10(rs), bins=45, color="tab:blue", alpha=0.85)
ax.set_xlabel(r"$\log_{10} r_*$"); ax.set_ylabel("directions")
ax.set_title("(b) Three decades of flip radii")
ax.grid(alpha=0.25)

ax = axs[2]
# r* = r_ref / |u| at the argmin, because r_c = m/(|dm|/r) = r*m/|dm|
dm0 = d["dm"][0].astype(float)
u = dm0 / np.maximum(m, 1e-9)
i_min = np.argmin(rc, axis=1)
mu = u[np.arange(u.shape[0]), i_min]
ax.loglog(rstar[ok], np.abs(mu[ok]), ".", ms=3, color="tab:green")
r_ref = float(d["radii"][0])
xx2 = np.geomspace(rs.min(), rs.max(), 50)
ax.loglog(xx2, r_ref / xx2, "k--", lw=1.0, label=fr"$|u|=r_{{\rm ref}}/r_*$, $r_{{\rm ref}}={r_ref}$")
ax.set_xlabel(r"$r_*$"); ax.set_ylabel(r"$|u_t|$ at the argmin")
ax.set_title(r"(c) $r_*$ is the inverse of the margin coordinate")
ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rstar_1536.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved rstar_1536")

# ---- survival of free-run flips (DeepSeek, 10 prompts x 20 directions)
paths = glob.glob(os.path.join(LD, "models", "qwen3_4b", "chaos", "*_r*", "free_run.npz"))
ds = sorted(glob.glob(os.path.join(ROOT, "results", "big4", "chaos", "*", "free_run.npz")))
if ds:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for p in ds[:6]:
        z = np.load(p, allow_pickle=True)
        obs = z["obs"].ravel()
        name = os.path.basename(os.path.dirname(p))
        t = np.arange(0, 301)
        S = np.array([(obs < 0).mean() if False else (obs >= tt).mean() if (obs >= 0).any() else 1.0
                      for tt in t])
        S = np.array([np.mean((obs < 0) | (obs >= tt)) for tt in t])
        ax.step(t, S, where="post", lw=1.1, label=name)
    ax.set_xlabel("generated token $t$"); ax.set_ylabel("survival $S(t)$")
    ax.set_title("Fraction of directions that have not flipped yet")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flip_survival.png"), dpi=185, bbox_inches="tight")
    plt.close(fig)
    print("saved flip_survival", len(ds))

json.dump(NUM, open(os.path.join(ROOT, "results", "rstar_numbers.json"), "w"), indent=1)
print(json.dumps(NUM, indent=1)[:900])
