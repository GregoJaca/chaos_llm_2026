import numpy as np, os, collections
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"
pc = np.load(os.path.join(OUT, "t1_per_curve.npy"), allow_pickle=True)
cnt = collections.Counter(row["decay_best_bic"] for row in pc)
print("decay-segment BIC winners:", dict(cnt))
db = np.array([row["deltaBIC_expfloor_vs_exp"] for row in pc])
print("deltaBIC(expfloor-exp) median, q25, q75:", np.nanmedian(db), np.nanquantile(db, .25), np.nanquantile(db, .75))
print("frac expfloor better (dBIC<-2):", np.mean(db < -2), " frac exp better (dBIC>2):", np.mean(db > 2))
# by layer
for L in [3, 7, 14, 21, 27]:
    sub = [row for row in pc if row["L"] == L]
    c = collections.Counter(row["decay_best_bic"] for row in sub)
    print(f"L={L}: {dict(c)}")
# tau vs pure-exp tau at layer 27
for L in [7, 27]:
    v_tau = np.array([row["decay_tau"] for row in pc if row["L"] == L and np.isfinite(row["decay_tau"])])
    v_lam = np.array([row["decay_lam"] for row in pc if row["L"] == L and np.isfinite(row["decay_lam"])])
    print(f"L={L}: tau_floor med={np.median(v_tau):.1f}, tau_exp=1/|lam| med={1/np.median(np.abs(v_lam)):.1f}")
