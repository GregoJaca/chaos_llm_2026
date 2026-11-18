import glob, os
import numpy as np

OUT = "results/long_last"
os.makedirs(OUT, exist_ok=True)
for root in ["results/long_traj", "results/long_traj2", "results/long_traj3", "results/long_traj4"]:
    for path in sorted(glob.glob(os.path.join(root, "*.npz"))):
        d = np.load(path)
        if "h_last" not in d:
            continue
        name = os.path.basename(path)
        np.savez_compressed(os.path.join(OUT, name), h_last=d["h_last"].astype(np.float32),
                            gen_ids=d["gen_ids"])
        print(name, d["h_last"].shape)
