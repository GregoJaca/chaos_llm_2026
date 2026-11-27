import numpy as np, os, sys
sys.path.insert(0, "analysis_update")
from cache import HERE
z = np.load(os.path.join(HERE, "results/depth2/deepseek/depth.npz"), allow_pickle=True)
print(z.files)
for k in z.files:
    print(k, z[k].shape, z[k].dtype)
