import numpy as np
import plots

a1 = plots.load_a1(r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers")
cos = np.asarray(a1["cos"], float)
for ri in [6, 9, 11]:
    for li in [0, 3, 7]:
        c = cos[ri, :, li, :]
        print(f"r={a1['radii'][ri]} L={a1['layers'][li]}: "
              f"median={np.nanmedian(c):.5f} p1={np.nanpercentile(c, 1):.5f} "
              f"min={np.nanmin(c):.4f}")
