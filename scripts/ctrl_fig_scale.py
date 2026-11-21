"""Mamba scale check: 130m vs 370m vs 1.4b (790m pending) - does the control's
behaviour change with scale? Response, lambda, thresholds, obs/pred."""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "results", "ctrl_local", "mscale")
MAM130 = os.path.join(ROOT, "results", "ctrl_local", "mamba130")
OUT = os.path.join(ROOT, "notebook", "figures_controls")
os.makedirs(OUT, exist_ok=True)


def signed_rc(zz, r):
    ri = int(np.argmin(np.abs(zz["radii"] - r)))
    m = zz["margins"][None, :]
    neg = np.maximum(-zz["dm"][ri], 0.0)
    rc = np.where(neg > 0, m * float(zz["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    return rc, float(zz["radii"][ri])


z130 = np.load(os.path.join(MAM130, "a1i_txt.npz"))
z370 = np.load(os.path.join(MS, "mamba370_a1i_txt.npz"))
z14 = np.load(os.path.join(MS, "mamba14_a1i_txt.npz"))

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
cols = {"130m": "#0072B2", "370m": "#56B4E9", "1.4b": "#009E73"}
for z, name, ruse in [(z130, "130m", 0.1), (z370, "370m", 0.1), (z14, "1.4b", 0.1)]:
    ri = int(np.argmin(np.abs(z["radii"] - ruse)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    axs[0].plot(np.arange(len(a)), a, lw=1.2, color=cols[name], label=f"Mamba-{name}")
    k = 10
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    axs[1].plot(t, lam, lw=1.0, color=cols[name], label=name)
axs[0].set_yscale("log")
axs[0].set_xlabel(r"generated token $t$")
axs[0].set_ylabel(r"$a(t)/r$ at the last layer, $r=0.1$")
axs[0].set_title("(a) The control's response does not grow with scale:\nno size produces a persistent amplified response")
axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3, which="both")
axs[1].axhline(0, color="k", lw=0.8)
axs[1].set_xlabel(r"generated token $t$")
axs[1].set_ylabel(r"$\lambda_{10}(t)$ (nats/token)")
axs[1].set_title("(b) Stretching rate at three scales: fluctuating around zero,\nno sustained positive phase at any size")
axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c2b_scale.png"), dpi=190); plt.close(fig)

print("model  layers  margins(min/med)  r*(measured)  a/r t=1..t=299  lambda mean/max")
for z, name in [(z130, "130m"), (z370, "370m"), (z14, "1.4b")]:
    rc, rmeas = signed_rc(z, 1.0)
    ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    k = 10
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    print(f"{name:5s} {int(z['layers'][-1])+1:3d}    {z['margins'].min():.4g}/{np.median(z['margins']):.3g}    "
          f"{float(np.min(rc)):.3g} (r={rmeas:g})    {a[1]:.2f}->{a[299]:.2f}    {lam.mean():+.4f}/{lam.max():+.4f}")

# obs/pred for 370 and 1.4b where free runs exist
for z, name, root in [(z370, "370m", os.path.join(MS, "fr")),
                      (z14, "1.4b", os.path.join(MS, "fr"))]:
    for f in sorted(glob.glob(os.path.join(root, f"*{name.replace('.', '')}*" if False else "*.npz"))):
        base = os.path.basename(f)
        if not base.startswith("mamba" + name.replace(".", "")) and name != "370m":
            continue
        try:
            rr = float(base.split("_r")[-1][:-4])
        except ValueError:
            continue
        fr = np.load(f, allow_pickle=True)
        obs = fr["obs"][0]
        rc, _ = signed_rc(z, rr)
        hit = rc <= rr
        pred_T = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        both = (obs >= 0) & (pred_T >= 0)
        print(f"{name} r={rr}: obs {(obs>=0).sum()}/20 pred {(pred_T>=0).sum()} "
              f"exact {int((obs[both]==pred_T[both]).sum())} "
              f"FA {int(((pred_T>=0)&(obs<0)).sum())} miss {int(((pred_T<0)&(obs>=0)).sum())}")
