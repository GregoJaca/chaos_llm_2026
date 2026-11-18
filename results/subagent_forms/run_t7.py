"""Task 7: divergence curves from free-run token sequences.

(a) f_ident(t): fraction of directions whose token still equals the reference at
    position t, per (dataset, radius).  Fits: exp-to-floor, power-to-floor,
    linear-to-floor, on data with n=20 dirs (binomial/bootstrap CIs).
(b) survival S(t) = P(no flip yet): exponential (const hazard) vs Weibull.
(c) window comparison of free-run divergence from free_*.json flip records
    (w100 / w200 / w500): flip fractions and fitted survival scales.
"""
import os, json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitlib import _ols, _ic

warnings.filterwarnings("ignore")
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"

def load_fr(fname):
    z = np.load(os.path.join(BASE, fname), allow_pickle=True)
    radii = list(z["radii"]); obs = z["obs"]; gens = z["gen_ids"]; refs = z["ref_ids"]
    out = {}
    for r_idx, r in enumerate(radii):
        # chunks of 6 dirs; rows 1..6 of chunk k=r*4+c are dirs 6c..6c+5
        agree = []   # [n_valid_dirs, T] boolean
        for c in range(4):
            k = r_idx * 4 + c
            if k >= len(gens): break
            g = np.asarray(gens[k]); ref = np.asarray(refs[k])
            for j in range(1, g.shape[0]):
                d = 6 * c + (j - 1)
                if d >= 20: break
                agree.append(g[j] == ref)
        out[float(r)] = np.array(agree)     # [<=20, 300]
    return out

DATASETS = [("fr_w100_large.npz", "w100, prompt a1"), ("fr_w100_mid.npz", "w100, prompt a1 (mid r)"),
            ("fr_childhood.npz", "w100, prompt childhood"), ("fr_cyber.npz", "w100, prompt cyber")]
fr = {fname: load_fr(fname) for fname, name in DATASETS}

def fit_fident(t, f):
    """f in [0,1]; models: f = C+(1-C)exp(-t/tau) | C+(1-C)(1+t/t0)^-b | clip lin."""
    yl = f
    res = {}
    # exp to floor: linearize? do grid over tau with OLS on [exp(-t/tau), 1]
    best = None
    for tau in np.exp(np.linspace(np.log(3), np.log(3000), 80)):
        X = np.column_stack([np.exp(-t / tau), np.ones_like(t, float)])
        b, cov, rss, r2, n = _ols(X, yl)
        if best is None or rss < best[2]: best = (tau, b, rss)
    tau, b, rss = best
    C = float(np.clip(b[1], 0, 1)); A = float(b[0])
    r2 = 1 - rss / ((yl - yl.mean()) ** 2).sum()
    res["exp-floor"] = dict(tau=float(tau), C=C, A=A, r2=r2, bic=_ic(rss, len(t), 2)[1],
                            pred=A * np.exp(-t / tau) + C)
    # power to floor: grid t0
    best = None
    for t0 in np.exp(np.linspace(np.log(1), np.log(500), 60)):
        X = np.column_stack([(1 + t / t0) ** -1.0, np.ones_like(t, float)])  # fix b=1 grid over b?
        b, cov, rss, r2, n = _ols(X, yl)
        if best is None or rss < best[2]: best = (t0, b, rss)
    bestb = None
    for bb in np.linspace(0.3, 4, 20):     # now grid exponent b with t0
        for t0 in np.exp(np.linspace(np.log(1), np.log(500), 30)):
            X = np.column_stack([(1 + t / t0) ** (-bb), np.ones_like(t, float)])
            b, cov, rss, r2, n = _ols(X, yl)
            if bestb is None or rss < bestb[3]: bestb = (float(t0), float(bb), b, float(rss))
    t0, bb, b, rss = bestb
    C = float(np.clip(b[1], 0, 1)); A = float(b[0])
    r2 = 1 - rss / ((yl - yl.mean()) ** 2).sum()
    res["power-floor"] = dict(t0=float(t0), b=float(bb), C=C, A=A, r2=r2, bic=_ic(rss, len(t), 3)[1],
                              pred=A * (1 + t / t0) ** (-bb) + C)
    # linear to floor
    X = np.column_stack([np.maximum(1 - t / 300.0, 0), np.ones_like(t, float)])
    b, cov, rss, r2, n = _ols(X, yl)
    r2 = 1 - rss / ((yl - yl.mean()) ** 2).sum()
    res["linear-floor"] = dict(slope=float(b[0]), C=float(np.clip(b[1], 0, 1)), r2=r2,
                               bic=_ic(rss, len(t), 2)[1], pred=np.maximum(b[0] * (1 - t / 300.0) + b[1], 0))
    return res

def survival_from_obs(flip_times, T):
    """S(t) = frac of runs with no flip through t; -1 = never flips (censored at T)."""
    ft = np.asarray(flip_times, float)
    ft_c = np.where(ft < 0, T, ft)
    return 1 - np.array([(ft_c <= t).mean() for t in range(T)])

def fit_survival(t, S):
    """S = exp(-t/tau) | S=exp(-(t/tw)^k). t from 0..T-1."""
    yl = np.log(np.maximum(S, 1e-12))
    m = S > 0
    res = {}
    X = np.column_stack([t[m].astype(float), ])   # exp: log S = -t/tau
    b, cov, rss, r2, n = _ols(np.column_stack([np.ones(m.sum()), t[m].astype(float)]), yl[m])
    res["exponential"] = dict(tau=float(-1 / b[1]), r2=r2, bic=_ic(rss, n, 1)[1],
                              se_tau=float(abs(np.sqrt(cov[1, 1]) / b[1] ** 2)),
                              pred=np.exp(b[0] + b[1] * t))
    best = None
    for k in np.linspace(0.3, 4, 40):
        x = t[m].astype(float) ** k
        b2, cov2, rss2, r22, n2 = _ols(np.column_stack([np.ones(m.sum()), x]), yl[m])
        if best is None or rss2 < best[2]: best = (k, b2, rss2)
    k, b2, rss2 = best
    res["weibull"] = dict(k=float(k), tw=float(np.exp(-b2[0] / b2[1]) ** (1 / k)) if b2[1] != 0 else np.nan,
                          r2=1 - rss2 / ((yl[m] - yl[m].mean()) ** 2).sum(),
                          bic=_ic(rss2, m.sum(), 2)[1], pred=np.exp(b2[0] + b2[1] * t ** k))
    return res

results = {}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
axF, axS, axW = axs
cmap = plt.get_cmap("viridis")
print("=== f_ident(t) fits (n=20 dirs per radius unless noted) ===")
for fname, label in DATASETS:
    for r, agree in fr[fname].items():
        f = agree.mean(axis=0)               # [300]
        n = agree.shape[0]
        t = np.arange(len(f))
        # skip curves that never diverge
        if f.min() > 0.999:
            print(f"{label} r={r:g}: no divergence (f=1 throughout), n={n} -- no fit")
            results[f"{fname}_{r}"] = dict(n=n, no_divergence=True)
            continue
        fits = fit_fident(t, f)
        win = min(fits, key=lambda k: fits[k]["bic"])
        lo95 = np.maximum(1 - f, 1e-6) * 0  # placeholder
        # bootstrap CI on f(t)
        rng = np.random.default_rng(0)
        bs = np.array([agree[rng.integers(0, n, n)].mean(axis=0) for _ in range(1000)])
        flo, fhi = np.quantile(bs, [.025, .975], axis=0)
        print(f"{label} r={r:g} n={n}: winner={win} " +
              " ".join(f"{k}:R2={v['r2']:.3f},BIC={v['bic']:.0f}" for k, v in fits.items()) +
              f" | {win} params: " + str({k: round(v, 3) for k, v in fits[win].items() if k not in ('pred', 'r2', 'bic')}))
        results[f"{fname}_{r}"] = dict(n=n, winner=win,
                                       params={k: v for k, v in fits[win].items() if k not in ("pred",)},
                                       all_r2={k: v["r2"] for k, v in fits.items()})
        # survival
        mm = ~agree
        flips = np.where(mm.any(axis=1), np.argmax(mm, axis=1), -1)  # first mismatch per dir, -1 if none
        S = survival_from_obs(flips, len(f))
        sfits = fit_survival(t, S)
        results[f"{fname}_{r}"]["survival"] = {k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                                                for k, v in sfits.items()}
        # plot selected
        col = cmap(min(0.9, float(r) / 2.5 + 0.1))
        axF.plot(t, f, color=col, lw=1.2, label=f"{label.split(',')[-1].strip()} r={r:g}")
        axF.plot(t, fits[win]["pred"], "--", color=col, lw=.9, alpha=.8)

axF.set_xlabel("t (tokens)"); axF.set_ylabel("fraction identical tokens")
axF.set_ylim(-0.05, 1.05); axF.legend(fontsize=6, ncol=2); axF.grid(alpha=.3)
axF.set_title("token divergence f(t): best fit per (prompt, r)")

print("\n=== survival fits (representative) ===")
for key in ["fr_w100_large.npz_0.5", "fr_w100_large.npz_1.0", "fr_w100_large.npz_2.0",
            "fr_childhood.npz_0.3", "fr_w100_mid.npz_0.3"]:
    if key in results and "survival" in results.get(key, {}):
        print(key, {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()
                        if kk in ("tau", "k", "r2", "bic")} for k, v in results[key]["survival"].items()})
        fname = key.rsplit("_", 1)[0]; r = float(key.split("_")[-1])
        agree = fr[fname][r]
        mm = ~agree
        flips = np.where(mm.any(axis=1), np.argmax(mm, axis=1), -1)
        S = survival_from_obs(flips, agree.shape[1])
        axS.plot(np.arange(len(S)), S, lw=1.2, label=key.replace(".npz", ""))
axS.set_xlabel("t"); axS.set_ylabel("S(t) = P(no flip yet)")
axS.legend(fontsize=6); axS.grid(alpha=.3)
axS.set_title("free-run survival curves")

# ---- window comparison from json flip records ----
print("\n=== window comparison (free-run flips, prompt a1, 20 dirs/radius) ===")
wincomp = {}
for wname, fjson, radii_sel in [("w100", "free_w100_layers_large.json", None),
                                ("w200", "free_w200.json", None),
                                ("w500", "free_w500.json", None),
                                ("w100", "free_large.json", None)]:
    with open(os.path.join(BASE, fjson)) as f: j = json.load(f)
    for r, o in zip(j["radii"], j["obs"]):
        o = np.asarray(o)
        rec = dict(window=j.get("window", wname), n=len(o), n_flip=int((o >= 0).sum()),
                   frac_flip=float((o >= 0).mean()),
                   med_flip=float(np.median(o[o >= 0])) if (o >= 0).any() else None)
        wincomp[f"{fjson}_{r}"] = rec
        print(f"{fjson}: W={rec['window']} r={r:g}: flips {rec['n_flip']}/{rec['n']}"
              + (f", median flip t={rec['med_flip']:.0f}" if rec['med_flip'] is not None else ""))
# plot flip fraction vs radius for w100/w200/w500
marks = {"free_w100_layers_large.json": ("tab:blue", "o"), "free_w200.json": ("tab:orange", "s"),
         "free_w500.json": ("tab:green", "^"), "free_large.json": ("tab:red", "D")}
for fjson, (c, mk) in marks.items():
    rs = [float(k.split("_")[-1]) for k in wincomp if k.startswith(fjson)]
    fs = [wincomp[k]["frac_flip"] for k in wincomp if k.startswith(fjson)]
    lbl = fjson.replace("free_", "").replace(".json", "")
    axW.scatter(rs, fs, color=c, marker=mk, label=lbl)
    # bootstrap CI on fraction (n=20)
    for r in rs:
        k = f"{fjson}_{r}"; n = wincomp[k]["n"]; p = wincomp[k]["frac_flip"]
        se = np.sqrt(max(p * (1 - p), 1 / (4 * n)) / n)
        axW.errorbar([r], [p], yerr=[[1.96 * se], [1.96 * se]], color=c, fmt="none", capsize=3, alpha=.6)
axW.set_xscale("log"); axW.set_xlabel("r"); axW.set_ylabel("fraction of dirs flipped by t=300")
axW.set_ylim(-.05, 1.05); axW.legend(fontsize=7); axW.grid(alpha=.3)
axW.set_title("free-run divergence vs window (w100<w200~w500)")

fig.suptitle("Task 7: divergence of free-running token streams")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_divergence.png"), dpi=160); plt.close(fig)

with open(os.path.join(OUT, "t7_summary.json"), "w") as f:
    json.dump(dict(results={k: {kk: vv for kk, vv in v.items() if kk != "all_r2"} for k, v in results.items()},
                   window_comp=wincomp), f, indent=1)
print("saved fig_divergence.png, t7_summary.json")
