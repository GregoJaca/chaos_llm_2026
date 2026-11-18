import numpy as np, time
from common_d2c import load, miiaft, naive_best_fit, distm, c_eps_curve, eps_grid_from_med, rng_for, pair_values

X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
Y = U[:, :999] * sv[:999]

t0 = time.time(); Z = miiaft(Y, rng_for("interstellar_travel", 200), n_iter=40)
print(f"miiaft(999 cols, 40 it): {time.time()-t0:.1f}s")

t0 = time.time(); Z = miiaft(Y[:, :50], rng_for("interstellar_travel", 200), n_iter=40)
print(f"miiaft( 50 cols, 40 it): {time.time()-t0:.1f}s")

D = distm(Xc)
med = float(np.median(pair_values(D, 0)))
eps = eps_grid_from_med(med)
C = c_eps_curve(D, eps, 0)
t0 = time.time(); nb = naive_best_fit(eps, C)
print(f"naive_best_fit(96-grid): {time.time()-t0:.2f}s -> {nb}")

t0 = time.time(); D2 = distm(Xc)
print(f"distm(1000x1536): {time.time()-t0:.2f}s")
