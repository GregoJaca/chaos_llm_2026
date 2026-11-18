"""Recompute only the MIAAFT-dependent entries of d2_results.json with the
fixed rank-match miiaft (cross-spectrum now preserved to ~0.1 rel err):
cloud MIAAFT C(eps) curves, their naive fits, their RQA, and the fidelity
probe. Data-side entries (curves, runs, ER widths, shuffles, phase) are
untouched. Regenerates fig2.
"""
import json
import numpy as np
from common_d2c import (PROMPTS, OUT, load, distm, c_eps_curve, eps_grid_from_med,
                        naive_best_fit, cloud_miiaft, miiaft, rqa_from_D,
                        pair_values, save_json, rng_for)
import exp1_d2 as e1

with open(OUT / "d2_results.json") as f:
    res = json.load(f)

N_IA = 5
for name in PROMPTS:
    old_fid = res[name].get("miiaft_crosscov_relerr", [1.0])
    if np.median(old_fid) < 0.6:
        print(f"{name[:34]:36s} already patched (fid med {np.median(old_fid):.3f}), skipping",
              flush=True)
        continue
    X, _ = load(name)
    Xc = X - X.mean(0)
    med = res[name]["med_dist"]
    eps = eps_grid_from_med(med)
    U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
    Ytop = U[:, :50] * sv[:50]
    ia_curves, ia_nbf, rqa_ia = [], [], []
    for s_i in range(N_IA):
        Zi = cloud_miiaft(Xc, rng_for(name, 200 + s_i))
        Di = distm(Zi)
        Cia = c_eps_curve(Di, eps, 0)
        ia_curves.append(list(Cia))
        ia_nbf.append(naive_best_fit(eps, Cia))
        rqa_ia.append(rqa_from_D(Di))
    C1 = (Ytop[:-1].T @ Ytop[1:]) / (Ytop.shape[0] - 1)
    fid = []
    for s_i in range(3):
        Zt = miiaft(Ytop, rng_for(name, 400 + s_i), n_iter=25)
        C1s = (Zt[:-1].T @ Zt[1:]) / (Ytop.shape[0] - 1)
        fid.append(float(np.linalg.norm(C1s - C1) / np.linalg.norm(C1)))
    res[name]["ia_C"] = ia_curves
    res[name]["ia_nbf"] = ia_nbf
    res[name]["rqa_iaaft"] = rqa_ia
    res[name]["miiaft_crosscov_relerr"] = [round(f, 3) for f in fid]
    print(f"{name[:34]:36s} fid med {np.median(fid):.3f} "
          f"DET(iaaft, RR=10%) {rqa_ia[0][1]['det']:.3f}", flush=True)
    save_json(res, "d2_results.json")
save_json(res, "d2_results.json")
e1.make_fig2(res)
print("fig2 regenerated with fixed-MIAAFT entries")
