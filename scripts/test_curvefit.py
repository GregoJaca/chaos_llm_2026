import sys
import numpy as np
sys.path.insert(0, "src")
from chaos_llm.curvefit import fit_curve

layers = np.arange(28)
lam = np.array([2.85, 1.42, 0.95, 0.74, 0.60, 0.49, 0.41, 0.35, 0.30, 0.26, 0.23,
                0.205, 0.185, 0.168, 0.154, 0.142, 0.132, 0.124, 0.117, 0.112, 0.107,
                0.103, 0.099, 0.096, 0.093, 0.090, 0.088, 0.085])
r = fit_curve(layers, lam)
print(r.summary())
print(r.table(10))
