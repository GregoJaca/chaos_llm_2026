"""Per-radius table from t1 per-curve fits (rise/decay/tau vs r, incl. r>=0.5)."""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"
pc = np.load(os.path.join(OUT, "t1_per_curve.npy"), allow_pickle=True)
pc = list(pc)
# extend with radii > 0.3 at L=27 (refit quickly)
import fitlib
from fitlib import fit_segments, find_peak
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"].astype(np.float64)
for ri, r in enumerate(radii):
    if r <= 0.31: continue
    for di in range(20):
        y = dh[ri, di, -1, :]
        t = np.arange(300); m = (t >= 1) & (y > 0)
        tpk = find_peak(t[m], y[m])
        seg = fit_segments(t[m], y[m], tpk)
        row = dict(r=float(r), d=di, L=27, t_peak=int(tpk), best_bic=None,
                   rise_lam=seg.get("rise", {}).get("params", {}).get("lam", np.nan),
                   rise_lam_se=seg.get("rise", {}).get("se", {}).get("lam", np.nan),
                   decay_lam=seg.get("decay-exp", {}).get("params", {}).get("lam", np.nan),
                   decay_lam_se=seg.get("decay-exp", {}).get("se", {}).get("lam", np.nan),
                   decay_tau=seg.get("decay-expfloor", {}).get("params", {}).get("tau", np.nan),
                   decay_tau_se=seg.get("decay-expfloor", {}).get("se", {}).get("tau", np.nan),
                   decay_pow_b=np.nan)
        pc.append(row)
rows = [x for x in pc if x["L"] == 27]
print("r      n   t*    rise_lam[CI]                decay_lam[CI]              tau_expfloor[CI]")
table = {}
for r in sorted(set(x["r"] for x in rows)):
    sub = [x for x in rows if x["r"] == r]
    def bm(key):
        v = np.array([x[key] for x in sub if np.isfinite(x[key])])
        rng = np.random.default_rng(0)
        bs = np.median(v[rng.integers(0, len(v), (2000, len(v)))], axis=1)
        return np.median(v), *np.quantile(bs, [.025, .975])
    m_r = bm("rise_lam"); m_d = bm("decay_lam"); m_t = bm("decay_tau"); m_p = bm("t_peak")
    table[float(r)] = dict(n=len(sub), tpeak=float(m_p[0]), rise=[float(x) for x in m_r],
                            decay=[float(x) for x in m_d], tau=[float(x) for x in m_t])
    print(f"{r:6g} {len(sub):3d} {m_p[0]:4.0f} {m_r[0]:+.4f}[{m_r[1]:+.4f},{m_r[2]:+.4f}] "
          f"{m_d[0]:+.4f}[{m_d[1]:+.4f},{m_d[2]:+.4f}] {m_t[0]:6.1f}[{m_t[1]:.1f},{m_t[2]:.1f}]")
json.dump(table, open(os.path.join(OUT, "t5_radius_table.json"), "w"), indent=1)
