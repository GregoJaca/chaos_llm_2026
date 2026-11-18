"""Inspect the additional free_run folders found in results/ (not in local_data).
These are raw run_thresholds.py outputs; check radii, obs/pred, sequences."""
import os

import numpy as np

import common2 as C
from common2 import FreeNPZ

for p in [
    r"C:\Users\hjaca\research\chaos_llm_2026\results\a1_ds_w100\free_run\free_run.npz",
    r"C:\Users\hjaca\research\chaos_llm_2026\results\a1_ds_w100\free_run_large\free_run.npz",
    r"C:\Users\hjaca\research\chaos_llm_2026\results\a1_ds_w200_large\free_run\free_run.npz",
    r"C:\Users\hjaca\research\chaos_llm_2026\results\a1_ds_w500_large\free_run\free_run.npz",
    r"C:\Users\hjaca\research\chaos_llm_2026\results\a1_cyber_w100\free_run\free_run.npz",
]:
    z = np.load(p, allow_pickle=True)
    print("=" * 70)
    print(os.path.basename(os.path.dirname(p)), "->", os.path.dirname(p))
    print("  radii:", z["radii"])
    obs = np.asarray(z["obs"])
    print("  obs flips per radius:", [int((obs[i] >= 0).sum())
                                      for i in range(obs.shape[0])])
    pred = np.asarray(z["pred"])
    print("  pred flips per radius:", [int((pred[i] >= 0).sum())
                                       for i in range(pred.shape[0])])
    nb = int(np.ceil(obs.shape[1] / 6))
    print("  batches:", len(list(z["gen_ids"])),
          "(expect", obs.shape[0] * nb, ")")
    # compare with the local_data records I already used
    if "w100\\free_run_large" in p:
        f = C.FreeNPZ(os.path.join(C.BASE, "fr_w100_large.npz"))
        print("  == fr_w100_large.npz obs:",
              (np.asarray(z["obs"]) == f.obs).all(),
              "pred:", (np.asarray(z["pred"]) == f.pred).all(),
              "radii:", (np.asarray(z["radii"]) == f.radii).all())
    if "w200_large" in p:
        import json
        j = json.load(open(os.path.join(C.BASE, "free_w200.json")))
        print("  == free_w200.json obs:", np.asarray(z["obs"]).tolist() == j["obs"])
    if "w500_large" in p:
        import json
        j = json.load(open(os.path.join(C.BASE, "free_w500.json")))
        print("  == free_w500.json obs:", np.asarray(z["obs"]).tolist() == j["obs"])
