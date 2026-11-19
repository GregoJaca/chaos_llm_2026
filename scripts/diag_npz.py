import numpy as np
for p in ["results/long_traj/ocean_currents.npz", "results/long_traj_2k/ocean_currents.npz",
          "results/long_last/ocean_currents.npz"]:
    try:
        d = np.load(p)
        print(p, list(d.keys()), {k: d[k].shape for k in d.keys() if k != "h_last"})
    except Exception as e:
        print(p, "ERR", e)
