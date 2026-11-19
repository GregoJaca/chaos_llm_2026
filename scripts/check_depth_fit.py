import sys
import numpy as np
sys.path.insert(0, "src")
from chaos_llm.curvefit import fit_curve

z = np.load(r"results\depth2\deepseek\depth.npz")
lam, L, tok = z["lam"], z["layers"], z["tokens"]
med = np.median(lam[:, :, 0], axis=0)
r = fit_curve(L[1:], med[1:])
print("n tokens", len(tok), "n points", len(L) - 1)
for f in r.fits[:6]:
    print(f"{f.name:22s} R2={f.r2:.6f} BIC={f.bic:8.2f} dBIC={f.bic-r.best.bic:7.2f}  {f.line(4)}")
logsig = med * (L + 1)
rr = fit_curve(L[1:], logsig[1:], families=["linear", "quadratic"], log_mode="native")
print("logsigma:", rr.best.name, rr.best.params, "se", rr.best.se, "R2", rr.best.r2)
