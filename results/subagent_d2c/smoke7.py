import numpy as np, time
from common_d2c import load
from exp3_predict import pc_scores, knn_nse, ridge_nse, kg_stats, h_half, HS

t0 = time.time()
X, _ = load("interstellar_travel")
Y, sv = pc_scores(X)
for h in HS:
    knn, pers, drift = knn_nse(Y, h)
    rid = ridge_nse(Y, h)
    print(f"h={h:2d}  knn {knn:+.3f}  ridge {rid:+.3f}  persist {pers:+.3f}  drift {drift:+.3f}")
print("h_half knn:", h_half([knn_nse(Y, h)[0] for h in HS]))
kg = kg_stats(Y, 8)
print("kg h=8: small", kg["small"], "overall", kg["overall"], "bins", np.round(kg["bins"], 3))
# surrogate sanity
from common_d2c import miiaft, phase_surrogate, rng_for
ia = miiaft(Y, rng_for("interstellar_travel", 1))
C1 = (Y[:-1].T @ Y[1:]) / 999
C1s = (ia[:-1].T @ ia[1:]) / 999
print("miiaft crosscov rel err:", round(float(np.linalg.norm(C1s - C1) / np.linalg.norm(C1)), 3))
print("miiaft marginal exact:", np.allclose(np.sort(ia, 0), np.sort(Y, 0)))
knn_s, _, _ = knn_nse(ia, 8)
rid_s = ridge_nse(ia, 8)
print(f"surrogate: knn {knn_s:+.3f} ridge {rid_s:+.3f}")
ph = phase_surrogate(Y, rng_for("interstellar_travel", 2))
knn_p, _, _ = knn_nse(ph, 8)
print(f"phase: knn {knn_p:+.3f}")
print(f"{time.time()-t0:.1f}s")
