"""Bootstrap CIs for survival-fit timescales (task 7)."""
import os, json
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_t7 import load_fr, survival_from_obs, fit_survival   # reuse loaders
import warnings; warnings.filterwarnings("ignore")

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"

def flips_of(agree):
    mm = ~agree
    return np.where(mm.any(axis=1), np.argmax(mm, axis=1), -1)

cases = [("fr_w100_large.npz", 0.5), ("fr_w100_large.npz", 1.0), ("fr_w100_large.npz", 2.0),
         ("fr_childhood.npz", 0.3), ("fr_cyber.npz", 0.3)]
res = {}
rng = np.random.default_rng(7)
for fname, r in cases:
    fr = load_fr(fname)
    agree = fr[float(r)]
    fl = flips_of(agree); n = len(fl); T = agree.shape[1]
    t = np.arange(T)
    tau_bs, k_bs = [], []
    for b in range(1000):
        fb = fl[rng.integers(0, n, n)]
        S = survival_from_obs(fb, T)
        if (S > 0).sum() < 20: continue          # degenerate resample (no flips)
        f = fit_survival(t, S)
        if np.isfinite(f["exponential"]["tau"]): tau_bs.append(f["exponential"]["tau"])
        k_bs.append(f["weibull"]["k"])
    S = survival_from_obs(fl, T); f0 = fit_survival(t, S)
    lo, hi = np.quantile(tau_bs, [.025, .975]) if len(tau_bs) > 50 else (np.nan, np.nan)
    klo, khi = np.quantile(k_bs, [.025, .975]) if len(k_bs) > 50 else (np.nan, np.nan)
    res[f"{fname}_{r}"] = dict(n=n, n_flip=int((fl >= 0).sum()),
                               tau_exp=f0["exponential"]["tau"], tau_lo=float(lo), tau_hi=float(hi),
                               weibull_k=f0["weibull"]["k"], k_lo=float(klo), k_hi=float(khi),
                               n_boot_ok=len(tau_bs))
    print(f"{fname} r={r:g}: flips={int((fl>=0).sum())}/{n}, tau_exp={f0['exponential']['tau']:.0f} "
          f"[{lo:.0f},{hi:.0f}], weibull k={f0['weibull']['k']:.2f} [{klo:.2f},{khi:.2f}]")
json.dump(res, open(os.path.join(OUT, "t7_boot.json"), "w"), indent=1)
print("saved t7_boot.json")
