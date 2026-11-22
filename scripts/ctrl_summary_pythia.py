"""Pythia-410 (attention, Pile) vs Mamba-130 (SSM, Pile) vs DeepSeek: key numbers."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "results", "ctrl_local", "pythia410", "a1i_txt_a1.npz")
MAM = os.path.join(ROOT, "results", "ctrl_local", "mamba130", "a1i_txt.npz")
DS = os.path.join(ROOT, "results", "local_data_ds", "w100_layers", "a1.npz")

z = np.load(PY, allow_pickle=True)
m = np.load(MAM)
d = np.load(DS)
print("pythia410 radii:", z["radii"])
print("margins: min=%.4g q10=%.3g med=%.3g" % (z["margins"].min(),
      np.quantile(z["margins"], .1), np.median(z["margins"])))

def resp(z, r):
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    return a

for name, zz, r in [("pythia410", z, 0.1), ("mamba130", m, 0.1), ("deepseek", d, 0.1)]:
    a = resp(zz, r)
    k = 10
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    print(f"{name}: a/r t=1 {a[1]:.3f} t=50 {a[50]:.3f} t=100 {a[100]:.3f} t=299 {a[299]:.3f}; "
          f"lam mean {lam.mean():+.4f} max {lam.max():+.4f}")

def signed_rstar(zz, r):
    ri = int(np.argmin(np.abs(zz["radii"] - r)))
    mm = zz["margins"][None, :]
    neg = np.maximum(-zz["dm"][ri], 0.0)
    rc = np.where(neg > 0, mm * float(zz["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    return float(rc.min())

print("signed r*: pythia410 %.4g (r=0.1), mamba txt %.4g (r=1), ds %.4g (r=0.1)" % (
    signed_rstar(z, 0.1), signed_rstar(m, 1.0), signed_rstar(d, 0.1)))

# free runs obs vs pred (pythia)
for rr in [0.3, 1.4, 3]:
    f = os.path.join(ROOT, "results", "ctrl_local", "pythia410", f"fr_txt_r{rr}.npz")
    if not os.path.exists(f):
        continue
    fr = np.load(f, allow_pickle=True)
    obs = fr["obs"][0]
    ri = int(np.argmin(np.abs(z["radii"] - rr)))
    mm = z["margins"][None, :]
    neg = np.maximum(-z["dm"][ri], 0.0)
    rc = np.where(neg > 0, mm * float(z["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    hit = rc <= rr
    pred_T = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    both = (obs >= 0) & (pred_T >= 0)
    print(f"pythia r={rr}: obs flips {(obs>=0).sum()}/20, pred {(pred_T>=0).sum()}, "
          f"exact {(obs[both]==pred_T[both]).sum()}, FA {((pred_T>=0)&(obs<0)).sum()}, "
          f"miss {((pred_T<0)&(obs>=0)).sum()}")
    if (obs >= 0).any():
        print("   flip steps:", sorted(int(x) for x in obs[obs >= 0]))
