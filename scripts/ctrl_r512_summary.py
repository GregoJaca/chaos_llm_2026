import glob
import os
import numpy as np

for f in sorted(glob.glob(r"results\ctrl_local\mamba130\r512\*.npz")):
    z = np.load(f)
    m = z["margins"][None, :]
    neg = np.maximum(-z["dm"][0], 0.0)
    rc = np.where(neg > 0, m * 0.1 / np.maximum(neg, 1e-30), np.inf)
    rstar = float(np.min(rc))
    K = rc.shape[0]
    n01 = int((rc.min(axis=1) <= 0.1).sum())
    print("%-12s r*(512 dirs) = %.4g ; dirs with r*<=0.1: %d/%d ; margins min %.4g"
          % (os.path.basename(f)[:-4], rstar, n01, K, z["margins"].min()))
