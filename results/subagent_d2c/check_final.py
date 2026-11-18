import json
import numpy as np

LOOPY = ["cybersecurity", "transformer_architecture", "history_of_computing",
         "renewable_energy_economics", "history_of_music", "ocean_currents",
         "sleep_and_memory", "urban_transportation"]
CLEAN = ["interstellar_travel", "childhood_experiences", "healthy_habits",
         "astronomy_observations", "evolution_of_language", "history_of_mathematics",
         "software_engineering_practices", "volcanoes_and_plate_tectonics"]

with open("pred_results.json") as f:
    pred = json.load(f)
with open("d2_results.json") as f:
    d2 = json.load(f)
with open("props_results.json") as f:
    pr = json.load(f)

# exp3 MIAAFT fidelity in PC-50
fid = np.array([np.mean(pred[n]["miiaft_crosscov_relerr"]) for n in pred])
print("exp3 PC-50 MIAAFT crosscov relerr: median %.3f min %.3f max %.3f" %
      (np.median(fid), fid.min(), fid.max()))

# gap_ridge and gap_iaaft for the clean 8 at h=8
gr = [pred[n]["gap_ridge_h8"] for n in CLEAN]
gia = [pred[n]["gap_iaaft_h8"] for n in CLEAN]
k1 = [pred[n]["nse_knn"][0] for n in CLEAN]
print(f"CLEAN 8: gap_ridge h8 med {np.median(gr):+.3f} max {np.max(gr):+.3f}; "
      f"gap_iaaft med {np.median(gia):+.3f} max {np.max(gia):+.3f}; knn h1 med {np.median(k1):+.3f}")
gr = [pred[n]["gap_ridge_h8"] for n in LOOPY]
k1 = [pred[n]["nse_knn"][0] for n in LOOPY]
k8 = [pred[n]["nse_knn"][3] for n in LOOPY]
rg8 = [pred[n]["nse_ridge"][3] for n in LOOPY]
print(f"LOOPY 8: gap_ridge h8 med {np.median(gr):+.3f} max {np.max(gr):+.3f}; "
      f"knn h1 med {np.median(k1):+.3f}; knn h8 med {np.median(k8):+.3f}; ridge h8 med {np.median(rg8):+.3f}")

# ridge vs surrogates (linear part second-order explained?)
rd = np.array([pred[n]["nse_ridge"][3] for n in pred])
rp = np.array([pred[n]["ridge_phase_mean"][3] for n in pred])
ri = np.array([pred[n]["ridge_iaaft_mean"][3] for n in pred])
print(f"ridge data - ridge phase (h8): med {np.median(rd-rp):+.3f} max {np.max(rd-rp):+.3f}")
print(f"ridge data - ridge MIAAFT (h8): med {np.median(rd-ri):+.3f} max {np.max(rd-ri):+.3f}")

# persistence + drift
pers1 = [pred[n]["nse_persist"][0] for n in pred]
pers8 = [pred[n]["nse_persist"][3] for n in pred]
drift = [pred[n]["nse_drift"][3] for n in pred]
print(f"persistence h1 med {np.median(pers1):+.3f} h8 med {np.median(pers8):+.3f}; "
      f"drift indicator h8 med {np.median(drift):+.3f} (mild nonstationarity)")

# h_half for the clean 8
hh = [(n, pred[n]["h_half_knn"], pred[n]["h_half_ridge"], pred[n]["h_half_iaaft_med"]) for n in CLEAN]
print("h_half (knn, ridge, MIAAFT) for clean 8:")
for n, a, b, c in hh:
    print(f"  {n[:34]:36s} {a} {b} {c}")

# DET vs best null
resid = []
for n in d2:
    g = lambda v: next(x["det"] for x in v if x["rr"] == 0.10)
    d = g(d2[n]["rqa_data"])
    ph = np.mean([g(v) for v in d2[n]["rqa_phase"]])
    ia = np.mean([g(v) for v in d2[n]["rqa_iaaft"]])
    resid.append(d - max(ph, ia))
resid = np.array(resid)
print(f"DET data - max(phase, MIAAFT): median {np.median(resid):+.3f}, "
      f"pos {(resid>0).sum()}/16, max {resid.max():+.3f}, min {resid.min():+.3f}")

# shuffle w50 deviation split
w50_loopy = [d2[n]["shuffle_w50_max_dlogC"] for n in LOOPY]
w50_clean = [d2[n]["shuffle_w50_max_dlogC"] for n in CLEAN]
print(f"time-shuffle w=50 max|dlC|: loopy med {np.median(w50_loopy):.2f} max {np.max(w50_loopy):.2f}; "
      f"clean med {np.median(w50_clean):.3f} max {np.max(w50_clean):.3f}")

# which naive fit is ER-legal
for n in d2:
    nb = d2[n]["cfgs"]["N1000_w0"]["nbf"]
    if nb and nb["C_lo"] >= 0.01 and nb["slope"] <= 6.0:
        print("ER-legal naive fit:", n, nb)

# token stats for the report: renewable & sleep unique tokens
for n in ["renewable_energy_economics", "sleep_and_memory", "cybersecurity"]:
    p = pr["props"][n]
    print(n, "unique_frac", p["unique_frac"], "H", p["entropy_bits"],
          "top_share", round(p["top_share"], 3), "loop", p["loop_dist"], "@", p["loop_P"])
