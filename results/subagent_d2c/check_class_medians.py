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
i8 = 3
for lab, names in (("LOOPY", LOOPY), ("CLEAN", CLEAN)):
    knn = np.array([pred[n]["nse_knn"][i8] for n in names])
    rg = np.array([pred[n]["nse_ridge"][i8] for n in names])
    ia = np.array([np.mean(pred[n]["knn_iaaft"][i8]) for n in names])
    ph = np.array([np.mean(pred[n]["knn_phase"][i8]) for n in names])
    sh = np.array([np.mean(pred[n]["knn_shuffle"][i8]) for n in names])
    gb = np.array([pred[n]["gap_best_surr_h8"] for n in names])
    print(f"{lab}: knn h8 [{knn.min():+.3f},{knn.max():+.3f}] med {np.median(knn):+.3f} | "
          f"ridge med {np.median(rg):+.3f} | MIAAFT [{ia.min():+.3f},{ia.max():+.3f}] med {np.median(ia):+.3f} | "
          f"phase [{ph.min():+.3f},{ph.max():+.3f}] med {np.median(ph):+.3f} | "
          f"shuffle med {np.median(sh):+.3f} | gap_best [{gb.min():+.3f},{gb.max():+.3f}]")
