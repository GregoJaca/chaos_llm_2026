"""Observed vs threshold-predicted survival curves, with free y-axes.

S_r(t) = fraction of the 20 directions whose first flip is later than t.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
T = 300


def load_obs(radii_wanted):
    """Return {r: array of first-flip steps (T=inf if never)}."""
    out = {}
    for fname, key in [("fr_w100_mid.npz", "obs"), ("fr_w100_extra.npz", "obs"), ("fr_w100_large.npz", "obs")]:
        p = os.path.join(B, fname)
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        for r, row in zip(d["radii"], d[key]):
            for rw in radii_wanted:
                if abs(float(r) - rw) < 1e-9:
                    out[rw] = np.asarray(row)
    return out


WANT = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
obs = load_obs(WANT)
d = np.load(os.path.join(B, "w100_layers", "a1.npz"), allow_pickle=True)
radii = d["radii"]; pred_T = d["pred_T"]
pred = {}
for rw in WANT:
    ri = int(np.argmin(np.abs(radii - rw)))
    pred[rw] = pred_T[ri]

fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True)
for ax, rw in zip(axs.flat, WANT):
    if rw not in obs:
        ax.set_visible(False)
        continue
    t = np.arange(T + 1)
    o = np.where(obs[rw] >= 0, obs[rw], np.inf)
    so = np.array([np.mean(o > tt) for tt in t])
    p = np.where(pred[rw] >= 0, pred[rw], np.inf)
    sp = np.array([np.mean(p > tt) for tt in t])
    ax.step(t, so, where="post", color="tab:blue", lw=1.6, label="observed (20 directions)")
    ax.step(t, sp, where="post", color="tab:red", lw=1.4, ls="--", label="threshold prediction")
    lo, hi = min(so.min(), sp.min()), max(so.max(), sp.max())
    pad = max(0.02, 0.06 * (hi - lo))
    ax.set_ylim(max(-0.02, lo - pad), min(1.03, hi + pad))
    ax.set_title(f"$r={rw:g}$", fontsize=10)
    ax.grid(alpha=0.25)
    if ax is axs.flat[0]:
        ax.legend(fontsize=8, loc="lower left")
for ax in axs[1]:
    ax.set_xlabel(r"generation step $t$")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$S_r(t)=P(T>t)$")
fig.suptitle("Survival curves: the threshold field predicts when trajectories leave the tube (free y-axes)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "stablechaos_survival.png"), dpi=190)
print("saved")
for rw in WANT:
    o = obs.get(rw)
    if o is None:
        continue
    print(f"r={rw:g}: flipped {int((o>=0).sum())}/20, mean T={(np.where(o>=0,o,T)).mean():.1f}")
