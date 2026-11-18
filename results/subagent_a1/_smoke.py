import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common

D = common.load("w100")
print("radii:", D["radii"])
print("small idx:", np.where(D["small"])[0], "large idx:", np.where(D["large"])[0])
print("floors:", dict(zip(D["layers"].tolist(), np.round(D["floor"], 6).tolist())))
print("flush:", D["flush"])
print("nG at L27: min/median over t:", D["nG"][-1].min(), np.median(D["nG"][-1]))
G = D["G"]
for li, l in enumerate(D["layers"]):
    if l == 0:
        continue
    print(f"L{l:2d} G(0)={G[li,0]:8.3f} G(12)={G[li,12]:8.3f} G(50)={G[li,50]:8.3f} "
          f"G(80)={G[li,80]:8.3f} G(120)={G[li,120]:8.3f} G(250)={G[li,250]:8.3f} "
          f"nG250={D['nG'][li,250]}")

# per-direction gains at t=0 (mean over small r), each layer
dh, radii = D["dh"], D["radii"]
small = np.where(D["small"])[0]
for li, l in enumerate(D["layers"]):
    if l == 0:
        continue
    g = np.array([np.mean(dh[small, idir, li, 0] / radii[small]) for idir in range(20)])
    print(f"L{l:2d} gain t=0 per dir: min={g.min():.3f} med={np.median(g):.3f} "
          f"max={g.max():.3f} (argmax dir {g.argmax()})")

# collapse CV per layer: (a) all 7 small radii on pre-flush t<=80; (b) safe radii
for li, l in enumerate(D["layers"]):
    if l == 0:
        continue
    for tag, idxs, trange in [("all7, t<=80", small, range(0, 80)),
                              ("safe4, all t", np.where(D["safe"])[0], range(300))]:
        cvs = []
        for t in trange:
            vals = [np.median(dh[ir, :, li, t]) / radii[ir] for ir in idxs]
            vals = np.array(vals)
            if vals.mean() > 0:
                cvs.append(vals.std() / vals.mean())
        cvs = np.array(cvs)
        print(f"L{l:2d} CV[{tag:14s}] median={np.median(cvs)*100:5.1f}%  "
              f"90pct={np.percentile(cvs,90)*100:5.1f}%")

# floor-hit diagnostic: where does dh(r=1e-3) exceed the linear prediction r*G?
for li, l in enumerate(D["layers"]):
    if l == 0:
        continue
    pred = 0.001 * D["G"][li]
    obs = D["dhm"][0, li]
    ok = np.isfinite(pred) & (pred > 0)
    ratio = obs[ok] / pred[ok]
    bad = ratio > 1.5
    tbad = np.where(ok)[0][bad]
    frac = bad.mean()
    t0 = tbad.min() if len(tbad) else None
    print(f"L{l:2d} floor-hit (dh(r=1e-3) > 1.5*r*G): frac_t={frac:.2f}, first t={t0}")
