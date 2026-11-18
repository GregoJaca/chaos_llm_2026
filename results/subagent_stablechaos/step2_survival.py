"""Step 2 (v2): decisive survival/threshold analysis + T(r) scaling for all prompts.
Adds MAD head-to-head (exp fit vs threshold), flip-count calibration, dh_norm decay.
Saves stats.json."""
import numpy as np, json, sys, pickle
sys.path.insert(0, r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_stablechaos")
from stablechaos_lib import *
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
a1 = load_a1(r"w100_layers\a1.npz")
margins, r_c, radii_a1, pred_T = a1["margins"], a1["r_c"], a1["radii"], a1["pred_T"]

FIELD = {0.05: 4, 0.1: 4, 0.2: 5, 0.3: 6, 0.5: 7, 1.0: 9, 2.0: 11}

datasets = {
    "main": [("fr_w100_mid.npz", "main_mid"), ("fr_w100_large.npz", "main_large")],
    "cyber": [("fr_cyber.npz", "cyber")],
    "childhood": [("fr_childhood.npz", "childhood")],
}
main_radii = []
for fn, tag in datasets["main"]:
    d = load_free(fn)
    for k in range(d["obs"].shape[0]):
        main_radii.append((tag, k, float(d["radii"][k])))
main_radii.sort(key=lambda x: x[2])

stats = {}
curves = {}
print("=" * 78)
print("A. Threshold-model vs observed, per (prompt, radius)")
print("=" * 78)

def analyze_one(d, k, r, field_i=None):
    T = d["obs"][k]; P = d["pred"][k]
    tg, Sobs = km_curve(T)
    tp, Sp = km_curve(P)          # threshold-model survival (saved per-direction pred_T)
    if field_i is not None:       # rebuild from the r_c field (main prompt) & cross-check
        (tf, Sf), Tf = threshold_survival(r_c[field_i], r)
        assert np.array_equal(Tf, P), (r, "field/pred mismatch")
    fits = fit_survival_forms(T)
    lam = fits["exp"]["lam"]
    Sexp = np.exp(-lam * tg)
    # bootstrap CI band for obs survival
    tb, Slo, Shi = bootstrap_S(T, nboot=1000, thin=10)
    out = dict(
        radius=r, n_flip=int((T >= 0).sum()), n_flip_pred=int((P >= 0).sum()),
        mean_T_cens=mean_restricted(T), mean_T_cens_pred=mean_restricted(P),
        mean_T_ci=[float(q) for q in np.quantile(bootstrap_mean_T(T, nboot=2000), [0.025, 0.975])],
        fits=fits,
        R2_threshold=r2_of_prediction(Sobs, Sp), sup_threshold=sup_dist(Sobs, Sp),
        MAD_threshold=mad(Sobs, Sp), MAD_exp=mad(Sobs, Sexp),
        R2_exp=fits["exp"]["R2"],
        exact_pred_match=int((T == P).sum()),
        both=int(((T >= 0) & (P >= 0)).sum()),
        obs_only=int(((T >= 0) & (P < 0)).sum()),
        pred_only=int(((T < 0) & (P >= 0)).sum()),
        later=int(((T >= 0) & (P >= 0) & (T > P)).sum()),
        earlier=int(((T >= 0) & (P >= 0) & (T < P)).sum()),
        Ts=sorted(T[T >= 0].tolist()), pred_Ts=sorted(P[P >= 0].tolist()),
    )
    return out, (tg, Sobs), (tp, Sp), Sexp, (tb, Slo, Shi)

for prompt in ["main", "cyber", "childhood"]:
    stats[prompt] = {}
    for fn, tag in datasets[prompt]:
        d = load_free(fn)
        for k in range(d["obs"].shape[0]):
            r = float(d["radii"][k])
            fi = FIELD.get(round(r, 3)) if prompt == "main" else None
            o, c_obs, c_pred, Sexp, band = analyze_one(d, k, r, fi)
            stats[prompt][f"r{r:g}"] = o
            curves[(prompt, r)] = (c_obs, c_pred, Sexp, band, o["Ts"], o["pred_Ts"])
            print(f"{prompt:10s} r={r:5.2f}: flips {o['n_flip']:2d}/20 (pred {o['n_flip_pred']:2d}) "
                  f"exactT {o['exact_pred_match']:2d}/20 | MAD thr {o['MAD_threshold']:.3f} vs exp {o['MAD_exp']:.3f} | "
                  f"R2 exp {o['R2_exp']:.2f} lam {lam if False else o['fits']['exp']['lam']:.4f} | "
                  f"pred_only {o['pred_only']} obs_only {o['obs_only']} later {o['later']} earlier {o['earlier']}")

print()
print("=" * 78)
print("B. T(r) scaling, main prompt (7 radii, 20 dirs each)")
print("=" * 78)
Ts_by_r, pred_by_r = {}, {}
for tag, k, r in main_radii:
    fn = "fr_w100_mid.npz" if tag == "main_mid" else "fr_w100_large.npz"
    d = load_free(fn)
    Ts_by_r[r] = d["obs"][k]; pred_by_r[r] = d["pred"][k]
rs = np.array([r for _, _, r in main_radii])
meanT = np.array([mean_restricted(Ts_by_r[r]) for r in rs])
meanT_pred = np.array([mean_restricted(pred_by_r[r]) for r in rs])

def fit_with_ci(kind, nboot=2000):
    pars = []
    for b in range(nboot):
        idx = rng.integers(0, 20, 20)
        y = np.array([mean_restricted(Ts_by_r[r][idx]) for r in rs])
        f = fit_T_vs_r(rs, y)
        pars.append(f[kind]["b"] if kind == "log" else f[kind]["alpha"])
    return np.array(pars)

f_all = fit_T_vs_r(rs, meanT)
boot_b = fit_with_ci("log"); boot_a = fit_with_ci("pow")
f_thr = r2_of_prediction(meanT, meanT_pred)
print(f"all 7 radii (censored mean, cap 300):")
print(f"  log-law  T=a-b ln r : a={f_all['log']['a']:.1f} b={f_all['log']['b']:.1f} "
      f"(CI {np.quantile(boot_b,.025):.1f}..{np.quantile(boot_b,.975):.1f}) R2={f_all['log']['R2']:.3f}")
print(f"  power    T=c r^-a   : c={f_all['pow']['c']:.1f} alpha={f_all['pow']['alpha']:.2f} "
      f"(CI {np.quantile(boot_a,.025):.2f}..{np.quantile(boot_a,.975):.2f}) R2={f_all['pow']['R2']:.3f}")
print(f"  threshold model (0 fitted params): R2={f_thr:.3f}  MAD={mad(meanT, meanT_pred):.1f} tokens"
      f"  (log-law MAD={mad(meanT, f_all['log']['a'] - f_all['log']['b']*np.log(rs)):.1f})")

# radii with flips only (0.2,0.3,0.5,1,2)
m = rs >= 0.2
f_flip = fit_T_vs_r(rs[m], meanT[m])
f_thr_flip = r2_of_prediction(meanT[m], meanT_pred[m])
print(f"radii with flips only (r>=0.2):")
print(f"  log-law  R2={f_flip['log']['R2']:.3f} (b={f_flip['log']['b']:.1f}) | power R2={f_flip['pow']['R2']:.3f} "
      f"(alpha={f_flip['pow']['alpha']:.2f}) | threshold R2={f_thr_flip:.3f}")

# flip fraction vs r
ff = np.array([(Ts_by_r[r] >= 0).mean() for r in rs])
ff_pred = np.array([(pred_by_r[r] >= 0).mean() for r in rs])
ff_ci = []
for r in rs:
    n = (Ts_by_r[r] >= 0).sum()
    lo = np.quantile([(rng.random(20) < n / 20).mean() for _ in range(500)], 0.025)
    ff_ci.append([float(max(0, np.quantile([(np.random.default_rng(b).random(20) * 20 < n).mean() for b in range(1)], 0.025) if False else 0)), 0])
# proper binomial CI
from scipy.stats import beta as beta_dist
ff_ci = []
for r in rs:
    n = int((Ts_by_r[r] >= 0).sum())
    lo = 0.0 if n == 0 else float(beta_dist.ppf(0.025, n, 20 - n + 1))
    hi = 1.0 if n == 20 else float(beta_dist.ppf(0.975, n + 1, 20 - n))
    ff_ci.append([lo, hi])
print("flip fraction obs vs pred:", list(zip(rs, ff.round(2), ff_pred.round(2))))

stats["T_vs_r"] = dict(radii=rs.tolist(), meanT=meanT.tolist(), meanT_pred=meanT_pred.tolist(),
                       meanT_ci=[[float(q) for q in np.quantile(bootstrap_mean_T(Ts_by_r[r], nboot=1000), [0.025, 0.975])] for r in rs],
                       log_all=f_all["log"], pow_all=f_all["pow"],
                       log_b_ci=[float(np.quantile(boot_b, 0.025)), float(np.quantile(boot_b, 0.975))],
                       pow_alpha_ci=[float(np.quantile(boot_a, 0.025)), float(np.quantile(boot_a, 0.975))],
                       R2_threshold_means=float(f_thr),
                       log_fliponly=f_flip["log"], pow_fliponly=f_flip["pow"], R2_threshold_fliponly=float(f_thr_flip),
                       flipfrac=ff.tolist(), flipfrac_pred=ff_pred.tolist(), flipfrac_ci=ff_ci)

print()
print("=" * 78)
print("C. Discreteness + margins")
print("=" * 78)
pooled = []
for r in rs:
    pooled += Ts_by_r[r][Ts_by_r[r] >= 0].tolist()
pooled = np.array(pooled)
vals, cnts = np.unique(pooled, return_counts=True)
ranks = {int(v): int(np.sum(margins < margins[v])) for v in vals}
min_rc = {}
for r in rs:
    fi = FIELD[round(r, 3)]
    v = np.where(np.isfinite(r_c[fi]), r_c[fi], np.inf).min()
    min_rc[f"{r:g}"] = float(v)
print(f"pooled n_flips={len(pooled)}, distinct steps={len(vals)}: {dict(zip(vals.tolist(), cnts.tolist()))}")
print(f"margin ranks of flip steps: {ranks}")
print(f"min r_c per radius: { {k: round(v,4) for k,v in min_rc.items()} }")
print(f"smallest margin {margins.min():.4f} at step {int(np.argmin(margins))}; margins[0]={margins[0]:.3f} (rank {int(np.sum(margins<margins[0]))}); margins[24]={margins[24]:.3f} (rank {int(np.sum(margins<margins[24]))})")
stats["discreteness"] = dict(n_flips=int(len(pooled)), distinct_steps=int(len(vals)),
                             steps=vals.tolist(), counts=cnts.tolist(), margin_ranks=ranks,
                             min_rc=min_rc, min_margin=float(margins.min()),
                             min_margin_step=int(np.argmin(margins)))
for prompt in ["cyber", "childhood"]:
    fn, tag = datasets[prompt][0]
    d = load_free(fn)
    pool = sorted(set(np.concatenate([d["obs"][k][d["obs"][k] >= 0] for k in range(d["obs"].shape[0])]).tolist()))
    pr = sorted(set([p for p in d["pred"].ravel().tolist() if p >= 0]))
    print(f"{prompt}: obs steps {pool}; pred steps {pr}")
    stats["discreteness"][prompt] = dict(obs_steps=pool, pred_steps=pr)

print()
print("=" * 78)
print("D. Teacher-forced perturbation norm dh_norm (amplification or decay?)")
print("=" * 78)
dh = np.asarray(a1.get("dh_norm") if False else np.load(BASE + r"\w100_layers\a1.npz", allow_pickle=True)["dh_norm"], float)
layers = np.load(BASE + r"\w100_layers\a1.npz", allow_pickle=True)["layers"]
# dh[i,d,layer,t]: norm of hidden-state perturbation at tracked layer, step t
prof = dh.mean(axis=1)  # [12, 8, 300] mean over directions
res = {}
for i in [2, 4, 7, 9, 11]:
    L = int(np.argmax(layers))  # last tracked layer index
    p = prof[i, L]
    res[f"r{radii_a1[i]:g}"] = dict(norm_t0=float(p[1]), norm_t10=float(p[10]), norm_t50=float(p[50]),
                                    norm_t100=float(p[100]), norm_t200=float(p[200]) if len(p) > 200 else None,
                                    growth_t50_over_t1=float(p[50] / p[1]) if p[1] > 0 else None)
    print(f"r={radii_a1[i]:5.2f} layer{layers[L]}: |dh| t=1: {p[1]:.4f}  t=10: {p[10]:.4f}  t=50: {p[50]:.4f}  "
          f"t=100: {p[100]:.4f}  growth(50/1)={p[50]/p[1]:.3f}")
# also mean over all tracked layers
p_all = prof[9].mean(axis=0)
print(f"r=1.0 mean-over-layers: t=1 {p_all[1]:.4f} t=10 {p_all[10]:.4f} t=50 {p_all[50]:.4f} t=100 {p_all[100]:.4f}")
stats["dh_norm"] = res

json.dump(stats, open(OUT + r"\stats.json", "w"), indent=1)
with open(OUT + r"\curves.pkl", "wb") as f:
    pickle.dump(curves, f)
print("\nsaved stats.json, curves.pkl")
