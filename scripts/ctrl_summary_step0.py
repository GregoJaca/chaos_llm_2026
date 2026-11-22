"""Step0 (random-init Pythia-410m) free-run scorecard against the signed rule."""
import glob
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S0 = os.path.join(ROOT, "results", "ctrl_local", "pythia410")
z = np.load(os.path.join(S0, "step0_a1i.npz"), allow_pickle=True)

def signed_rc(zz, r):
    ri = int(np.argmin(np.abs(zz["radii"] - r)))
    m = zz["margins"][None, :]
    neg = np.maximum(-zz["dm"][ri], 0.0)
    rc = np.where(neg > 0, m * float(zz["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    return rc, float(zz["radii"][ri])

rc, rmeas = signed_rc(z, 0.1)
print("step0 signed r* (measured at r=0.1):", float(np.min(rc)))
tot = dict(exact=0, fa=0, miss=0, obs=0, noflip=0)
for f in sorted(glob.glob(os.path.join(S0, "step0_fr", "*.npz"))):
    fr = np.load(f, allow_pickle=True)
    rr = float(np.atleast_1d(fr["radii"])[0])
    obs = fr["obs"][0]
    rc, _ = signed_rc(z, rr)
    hit = rc <= rr
    pred_T = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    both = (obs >= 0) & (pred_T >= 0)
    exact = int((obs[both] == pred_T[both]).sum())
    fa = int(((pred_T >= 0) & (obs < 0)).sum())
    miss = int(((pred_T < 0) & (obs >= 0)).sum())
    tot["exact"] += exact; tot["fa"] += fa; tot["miss"] += miss
    tot["obs"] += int((obs >= 0).sum()); tot["noflip"] += int((obs < 0).sum())
    steps = sorted(int(x) for x in obs[obs >= 0])
    print(f"r={rr:g}: obs {(obs>=0).sum()}/20 pred {(pred_T>=0).sum()} exact {exact} FA {fa} miss {miss}"
          + (f" steps {steps}" if steps else ""))
print("TOTAL:", tot)
