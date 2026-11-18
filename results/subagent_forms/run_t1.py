"""Task 1: per-(radius<=0.3, direction, layer) functional-form comparison for a(t).

Fits six full-curve models in log space, AIC/BIC winner counts, plus segment
(rise/decay) fits with uncertainties.  Saves results to fits_t1.npz / fits_t1.json.
"""
import os, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitlib import fit_all_full, fit_segments, find_peak, boot_stat_ci

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"

d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"].astype(np.float64)
r_sel = np.where(radii <= 0.31)[0]
l_sel = [3, 7, 14, 21, 27]
l_idx = [int(np.where(layers == L)[0][0]) for L in l_sel]
T = dh.shape[-1]

MODELS = ["exp", "power", "exp+floor", "lognormal", "pw-logt", "two-exp"]
res = {m: dict(aic=[], bic=[], r2=[], k=[]) for m in MODELS}
per_curve = []   # rows: r, d, l, t_peak, rise lam, se, decay lam, se, tau, ...
t0 = time.time()
nfail = 0
for ri in r_sel:
    for di in range(dh.shape[1]):
        for L, li in zip(l_sel, l_idx):
            y = dh[ri, di, li, :]
            tmax = 289 if L == 3 else T - 1          # layer 3 tail zeros at t>=290
            tt = np.arange(0, T); m = (tt >= 1) & (tt <= tmax) & (y > 0)
            t, yv = tt[m], y[m]
            try:
                fits = fit_all_full(t, yv)
            except Exception as e:
                nfail += 1; continue
            for mm in MODELS:
                f = fits[mm]
                res[mm]["aic"].append(f["aic"] if np.isfinite(f["aic"]) else np.inf)
                res[mm]["bic"].append(f["bic"] if np.isfinite(f["bic"]) else np.inf)
                res[mm]["r2"].append(f["r2"] if np.isfinite(f.get("r2", np.nan)) else np.nan)
                res[mm]["k"].append(f["k"])
            tpk = find_peak(t, yv)
            seg = fit_segments(t, yv, tpk)
            row = dict(r=float(radii[ri]), d=int(di), L=int(L), t_peak=int(tpk if tpk >= 1 else 1),
                       best_bic=min(MODELS, key=lambda mm: fits[mm]["bic"]))
            for key, src in [("rise_lam", "rise"), ("decay_lam", "decay-exp"),
                             ("decay_tau", "decay-expfloor"), ("decay_pow_b", "decay-power")]:
                f = seg.get(src)
                if f is None:
                    row[key] = np.nan; row[key + "_se"] = np.nan
                else:
                    if src == "decay-expfloor":
                        row[key] = f["params"]["tau"] if f["params"] else np.nan
                        row[key + "_se"] = f["se"]["tau"] if (f.get("se") and f["se"].get("tau")) else np.nan
                    else:
                        row[key] = f["params"].get("lam", f["params"].get("b"))
                        se = f.get("se") or {}
                        row[key + "_se"] = se.get("lam", se.get("b"))
            # decay-segment model comparison (exp vs expfloor vs power)
            row["decay_best_bic"] = min([k2 for k2 in ("decay-exp", "decay-expfloor", "decay-power") if k2 in seg],
                                       key=lambda k2: seg[k2]["bic"]) if len(seg) else None
            row["deltaBIC_expfloor_vs_exp"] = (seg["decay-expfloor"]["bic"] - seg["decay-exp"]["bic"]) if "decay-expfloor" in seg else np.nan
            per_curve.append(row)
print(f"fits done in {time.time()-t0:.1f}s, failures={nfail}, n_curves={len(per_curve)}")

# ---------- winner counts ----------
n = len(per_curve)
aic_arr = np.array(res["exp"]["aic"])  # just to get n
winners = {m: dict(aic=0, bic=0) for m in MODELS}
aic_mat = np.array([res[m]["aic"] for m in MODELS])   # [6, n]
bic_mat = np.array([res[m]["bic"] for m in MODELS])
for j in range(n):
    winners[MODELS[int(np.argmin(aic_mat[:, j]))]]["aic"] += 1
    winners[MODELS[int(np.argmin(bic_mat[:, j]))]]["bic"] += 1
print("winner counts (AIC):", {m: winners[m]["aic"] for m in MODELS})
print("winner counts (BIC):", {m: winners[m]["bic"] for m in MODELS})

med_r2 = {m: float(np.nanmedian(res[m]["r2"])) for m in MODELS}
print("median R^2:", {m: round(v, 3) for m, v in med_r2.items()})

# median R2 of the BIC winner per curve
r2_win = []
pc = per_curve
bic_ok = np.isfinite(bic_mat).all(axis=0)
for j in np.where(bic_ok)[0]:
    best = MODELS[int(np.argmin(bic_mat[:, j]))]
    r2_win.append(res[best]["r2"][j])
print("median R^2 of BIC winner:", np.nanmedian(r2_win))

# per-layer winner counts (BIC)
by_layer = {}
for row in pc:
    by_layer.setdefault(row["L"], []).append(row["best_bic"])
for L in sorted(by_layer):
    u, c = np.unique(by_layer[L], return_counts=True)
    print(f"layer {L}: " + ", ".join(f"{a}:{b}" for a, b in zip(u, c)))

# per-radius winner counts (BIC)
by_r = {}
for row in pc:
    by_r.setdefault(row["r"], []).append(row["best_bic"])
for r in sorted(by_r):
    u, c = np.unique(by_r[r], return_counts=True)
    print(f"r={r:g}: " + ", ".join(f"{a}:{b}" for a, b in zip(u, c)))

# ---------- segment parameter aggregates ----------
def agg(key, r, L, sel="d"):
    v = np.array([row[key] for row in pc if row["r"] == r and row["L"] == L and np.isfinite(row[key])])
    return v
print("\n=== segment rates (median [bootstrap 95% CI], per radius, layer 27) ===")
aggL = {}
for r in sorted(set(row["r"] for row in pc)):
    for L in [7, 14, 21, 27]:
        vr = agg("rise_lam", r, L); vd = agg("decay_lam", r, L); vt = agg("decay_tau", r, L); vp = agg("t_peak", r, L)
        if len(vr) < 5: continue
        lo_r, hi_r = np.quantile(np.random.default_rng(1).choice(vr, (2000, len(vr))).astype(float), [.025, .975], axis=0) if False else boot_stat_ci(vr)[1:3]
        lo_d, hi_d = boot_stat_ci(vd)[1:3] if len(vd) >= 5 else (np.nan, np.nan)
        lo_t, hi_t = boot_stat_ci(vt)[1:3] if len(vt) >= 5 else (np.nan, np.nan)
        aggL[(r, L)] = dict(n=len(vr), rise_med=float(np.median(vr)), rise_lo=float(lo_r), rise_hi=float(hi_r),
                            decay_med=float(np.median(vd)), decay_lo=float(lo_d), decay_hi=float(hi_d),
                            tau_med=float(np.median(vt)), tau_lo=float(lo_t), tau_hi=float(hi_t),
                            tpeak_med=float(np.median(vp)))
        if L == 27:
            a = aggL[(r, L)]
            print(f"r={r:6g} L=27 n={a['n']:3d}: rise={a['rise_med']:+.4f} [{a['rise_lo']:+.4f},{a['rise_hi']:+.4f}]  "
                  f"decay={a['decay_med']:+.4f} [{a['decay_lo']:+.4f},{a['decay_hi']:+.4f}]  tau={a['tau_med']:.1f} [{a['tau_lo']:.1f},{a['tau_hi']:.1f}]  t*={a['tpeak_med']:.0f}")

# identifiability: SE/param ratios
rel_se = {}
for key in ["rise_lam", "decay_lam"]:
    v = np.array([abs(row[key] / row[key + "_se"]) for row in pc if np.isfinite(row[key]) and np.isfinite(row[key + "_se"]) and row[key + "_se"] > 0])
    rel_se[key] = (float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75)))
print("\n|param/SE| medians (identifiability):", {k: tuple(round(x, 1) for x in v) for k, v in rel_se.items()})

np.save(os.path.join(OUT, "t1_per_curve.npy"), np.array(pc, dtype=object), allow_pickle=True)
with open(os.path.join(OUT, "t1_summary.json"), "w") as f:
    json.dump(dict(n_curves=n, winners={m: winners[m] for m in MODELS},
                   median_r2=med_r2, median_r2_winner=float(np.nanmedian(r2_win)),
                   by_layer={str(k): {m: int(sum(1 for x in v if x == m)) for m in MODELS} for k, v in by_layer.items()},
                   by_radius={str(k): {m: int(sum(1 for x in v if x == m)) for m in MODELS} for k, v in by_r.items()},
                   aggL={f"{k[0]}_{k[1]}": v for k, v in aggL.items()},
                   identifiability=rel_se), f, indent=1)
print("saved t1_summary.json")
