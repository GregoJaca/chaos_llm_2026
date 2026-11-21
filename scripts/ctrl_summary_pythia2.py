"""Pythia wide-radius obs/pred (measured at the target radius) + step0 + pythia1b numbers."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "results", "ctrl_local", "pythia410")

z = np.load(os.path.join(PY, "a1i_wide.npz"), allow_pickle=True)
print("wide radii:", z["radii"])

def signed_rc(zz, r):
    ri = int(np.argmin(np.abs(zz["radii"] - r)))
    m = zz["margins"][None, :]
    neg = np.maximum(-zz["dm"][ri], 0.0)
    rc = np.where(neg > 0, m * float(zz["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    return rc, float(zz["radii"][ri])

rc, rmeas = signed_rc(z, 0.19)
print("pythia410 signed r* (measured at r=0.19):", float(np.min(rc)))

tot = dict(exact=0, fa=0, miss=0, both=0, noflip=0, obs=0)
for R, tag in [(0.19, "frw_r019"), (0.3, "frw_r03"), (0.7, "frw_r07"), (1.4, "frw_r14"), (3, "frw_r3")]:
    fr = np.load(os.path.join(PY, tag + ".npz"), allow_pickle=True)
    obs = fr["obs"][0]
    rc, rmeas = signed_rc(z, R)
    hit = rc <= R
    pred_T = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    both = (obs >= 0) & (pred_T >= 0)
    exact = int((obs[both] == pred_T[both]).sum())
    fa = int(((pred_T >= 0) & (obs < 0)).sum())
    miss = int(((pred_T < 0) & (obs >= 0)).sum())
    tot["exact"] += exact; tot["fa"] += fa; tot["miss"] += miss
    tot["both"] += int(both.sum()); tot["noflip"] += int((obs < 0).sum()); tot["obs"] += int((obs >= 0).sum())
    print(f"r={R}: obs {(obs>=0).sum()}/20, pred {(pred_T>=0).sum()}, exact {exact}, FA {fa}, miss {miss}"
          + (f", flip steps {sorted(int(x) for x in obs[obs>=0])}" if (obs >= 0).any() else ""))
print("TOTAL:", tot)

# response and lambda at each model, wide
for name, fn, r in [("pythia410", "a1i_wide.npz", 0.3), ("pythia1b", "pythia1b_a1i.npz", 0.1),
                    ("pythia410_step0", "step0_a1i.npz", 0.1)]:
    zz = np.load(os.path.join(PY, fn), allow_pickle=True)
    ri = int(np.argmin(np.abs(zz["radii"] - r)))
    a = np.median(zz["dh_norm"][ri, :, -1] / zz["radii"][ri], axis=0)
    k = 10
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    rc, _ = signed_rc(zz, zz["radii"][ri])
    print(f"{name}: a/r t=1 {a[1]:.3f} t=100 {a[100]:.3f} t=299 {a[299]:.3f}; "
          f"lam mean {lam.mean():+.4f} max {lam.max():+.4f}; "
          f"margins min {zz['margins'].min():.3g} med {np.median(zz['margins']):.3g}; "
          f"r* {float(np.min(rc)):.3g}")
