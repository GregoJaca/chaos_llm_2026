import numpy as np, os, json
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"
z = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = z["radii"]; layers = list(z["layers"]); dh = z["dh_norm"].astype(np.float64)
Lfit = [3, 7, 14, 21, 27]
Laxis = [layers.index(L) for L in Lfit]
def slope(x, y):
    m = (np.asarray(y) > 0)
    X = np.column_stack([np.ones(m.sum()), np.log(np.asarray(x)[m])])
    yl = np.log(np.asarray(y)[m])
    b = np.linalg.lstsq(X, yl, rcond=None)[0]
    r = yl - X @ b
    return b[1], 1 - (r @ r) / (((yl - yl.mean()) ** 2).sum())
rng = np.random.default_rng(11)
res = {}
for r in [0.01, 0.1, 0.3, 1.0, 2.0]:
    ri = int(np.argmin(np.abs(radii - r)))
    A1 = np.array([np.median(dh[ri, :, ai, 1]) for ai in Laxis])
    p, r2 = slope(Lfit, A1)
    bs = []
    for b in range(2000):
        idx = rng.integers(0, dh.shape[1], dh.shape[1])
        Ab = np.array([np.median(dh[ri, idx, ai, 1]) for ai in Laxis])
        bs.append(slope(Lfit, Ab)[0])
    lo, hi = np.quantile(bs, [.025, .975])
    p2, r22 = slope(Lfit[1:], A1[1:])
    res[r] = dict(A1=[float(v) for v in A1], p=float(p), lo=float(lo), hi=float(hi), r2=float(r2),
                  p_noL3=float(p2), r2_noL3=float(r22))
    print(f"r={r:g}: A(1)={np.round(A1,4)}  p={p:+.3f} [{lo:+.3f},{hi:+.3f}] R2={r2:.3f} | noL3 p={p2:+.3f} R2={r22:.3f}")
# window_fixed at t=150
for W in [100, 150, 200, 300, 1000]:
    zz = np.load(os.path.join(BASE, "window_fixed", f"w{W}", "a1.npz"))
    dw = zz["dh_norm"].astype(np.float64); Rs = list(zz["radii"]); Ls = list(zz["layers"])
    ri = Rs.index(0.1)
    for tp in [1, 150]:
        A1 = np.array([np.median(dw[ri, :, Ls.index(L), tp]) for L in Ls])
        p, r2 = slope(Ls, A1)
        print(f"W={W} t={tp}: A={np.round(A1,4)} p={p:+.3f} R2={r2:.3f}")
json.dump(res, open(os.path.join(OUT, "depth_quick.json"), "w"), indent=1)
