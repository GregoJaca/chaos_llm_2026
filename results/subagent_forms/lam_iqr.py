import numpy as np, os, json
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"
d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"].astype(np.float64)
T = dh.shape[-1]
ri = int(np.argmin(np.abs(radii - 0.1)))
out = {}
for L in [7, 14, 27]:
    li = int(np.where(layers == L)[0][0])
    y = np.maximum(dh[ri, :, li, :], 1e-300)
    lam = np.log(y / y[:, 1:2]) / np.maximum(np.arange(T), 1)
    out[L] = {}
    for t in [10, 30, 50, 70, 90, 110, 150, 200, 299]:
        v = lam[:, t]
        out[L][t] = dict(med=float(np.median(v)), q1=float(np.quantile(v, .25)), q3=float(np.quantile(v, .75)))
    print(f"L={L}: " + "  ".join(
        f"t={t}: {out[L][t]['med']:+.4f} [{out[L][t]['q1']:+.4f},{out[L][t]['q3']:+.4f}]" for t in [30, 70, 110, 200]))
json.dump(out, open(os.path.join(OUT, "lam_iqr.json"), "w"), indent=1)
