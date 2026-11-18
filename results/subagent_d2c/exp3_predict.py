"""Exp 3: predictability replication test on the 16 new trajectories.

kNN (simplex-style, k=12, inverse-distance weights) prediction of x_{t+h} from
x_t in PC-50 space; horizons h = 1, 2, 4, 8, 16, 32; train = first 700 states,
held-out test = last 300; Theiler W=32 (>= max h) so no test point can use a
temporally adjacent library state. NSE against the train-mean constant
predictor on the test segment.

Baselines: (i) ridge regression on the same PC-50 features (alpha picked on an
internal validation tail of the train segment), (ii) persistence (x_{t+h} =
x_t), (iii) constant. Surrogates: MIAAFT in PC-50 space (per-PC marginal exact,
per-PC spectrum ~exact, cross-spectrum approximate - fidelity reported),
multivariate phase (cross-spectrum exact, Gaussianized marginals), time-shuffle
(statics only). Each surrogate runs through the identical pipeline.

Continuation sharpness (Kaplan-Glass): median rc/r over the smallest-r decile
of pairs (i,j) with |i-j|>50, where r = ||x_i-x_j||, rc = ||x_{i+h}-x_{j+h}||,
h = 1, 8, 16; full ratio-vs-r curves saved. Skill horizon h_half = h where
NSE drops to NSE(1)/2.

Output: pred_results.json, fig3_predictability.png, fig4_sharpness.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from common_d2c import (PROMPTS, REP, OUT, load, distm, phase_surrogate, miiaft,
                        save_json, rng_for)

plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5,
                     "axes.titlesize": 9.0, "axes.titleweight": "bold",
                     "figure.constrained_layout.use": True})

HS = [1, 2, 4, 8, 16, 32]
SPLIT = 700
KNN_K = 12
THEILER = 32
NPC = 50
N_IA, N_PH, N_SH = 5, 5, 3


def pc_scores(X, npc=NPC):
    Xc = X - X.mean(0)
    U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :npc] * sv[:npc], sv


def knn_nse(Y, h, k=KNN_K, split=SPLIT, W=THEILER):
    T = Y.shape[0]
    lib = np.arange(0, split - h)
    te = np.arange(split, T - h)
    Xl, Yl = Y[lib], Y[lib + h]
    Xt, Yt = Y[te], Y[te + h]
    sql = np.einsum("ij,ij->i", Xl, Xl)
    sqt = np.einsum("ij,ij->i", Xt, Xt)
    D2 = sqt[:, None] + sql[None, :] - 2.0 * (Xt @ Xl.T)
    np.maximum(D2, 0, out=D2)
    Dm = np.sqrt(D2)
    Dm[np.abs(te[:, None] - lib[None, :]) <= W] = np.inf
    kk = min(k, Dm.shape[1])
    nn = np.argpartition(Dm, kk - 1, axis=1)[:, :kk]
    nd = np.take_along_axis(Dm, nn, axis=1)
    wts = 1.0 / np.maximum(nd, 1e-12)
    pred = (wts[:, :, None] * Yl[nn]).sum(1) / wts.sum(1)[:, None]
    mu = Yl.mean(0)
    mse_const = ((Yt - mu) ** 2).sum(1).mean()
    nse_knn = 1 - ((Yt - pred) ** 2).sum(1).mean() / mse_const
    nse_persist = 1 - ((Yt - Xt) ** 2).sum(1).mean() / mse_const
    # drift indicator: would the TEST mean beat the train mean?
    nse_testmean = 1 - mse_const / ((Yt - Yt.mean(0)) ** 2).sum(1).mean()
    return float(nse_knn), float(nse_persist), float(nse_testmean)


def ridge_nse(Y, h, split=SPLIT, alphas=(0.1, 1.0, 10.0, 100.0), n_val=100):
    T = Y.shape[0]
    lib = np.arange(0, split - h)
    te = np.arange(split, T - h)
    Xl, Yl = Y[lib], Y[lib + h]
    Xt = Y[te]
    mu_x, sd_x = Xl.mean(0), Xl.std(0) + 1e-9
    mu_y = Yl.mean(0)
    Zl = (Xl - mu_x) / sd_x
    Zt = (Xt - mu_x) / sd_x
    nfit = max(len(lib) - n_val, 50)
    Af, Yfc = Zl[:nfit], Yl[:nfit] - mu_y
    Av, Yvc = Zl[nfit:], Yl[nfit:] - mu_y
    Gf = Af.T @ Af
    best = None
    for al in alphas:
        W = np.linalg.solve(Gf + al * np.eye(Gf.shape[0]), Af.T @ Yfc)
        mse = ((Av @ W - Yvc) ** 2).sum(1).mean()
        if best is None or mse < best[0]:
            best = (mse, al)
    al = best[1]
    A = Zl
    W = np.linalg.solve(A.T @ A + al * np.eye(A.shape[1]), A.T @ (Yl - mu_y))
    pred = Zt @ W + mu_y
    Yt = Y[te + h]
    mse_const = ((Yt - mu_y) ** 2).sum(1).mean()
    return float(1 - ((Yt - pred) ** 2).sum(1).mean() / mse_const)


def kg_stats(Y, h, w=50, qbins=8, small_frac=0.10):
    T = Y.shape[0]
    D = distm(Y)
    valid = T - h
    i, j = np.triu_indices(valid, 1)
    m = (j - i) > w
    r = D[i[m], j[m]]
    rc = D[i[m] + h, j[m] + h]
    ratio = rc / np.maximum(r, 1e-12)
    qs = np.quantile(r, np.linspace(0, 1, qbins + 1))
    bins = []
    for k in range(qbins):
        sel = (r >= qs[k]) & (r <= qs[k + 1])
        if sel.sum() > 20:
            bins.append(float(np.median(ratio[sel])))
        else:
            bins.append(np.nan)
    small = float(np.median(ratio[r <= np.quantile(r, small_frac)]))
    return dict(bins=bins, small=round(small, 4),
                overall=round(float(np.median(ratio)), 4),
                r_med=float(np.median(r)))


def h_half(nse_curve):
    """Horizon where NSE falls to NSE(1)/2 (log2-h interpolation)."""
    a = np.array(nse_curve, dtype=float)
    if not np.isfinite(a[0]) or a[0] <= 0.05:
        return None
    target = a[0] / 2.0
    for k in range(1, len(a)):
        if a[k] <= target:
            x0, x1 = np.log2(HS[k - 1]), np.log2(HS[k])
            return float(2 ** (x0 + (a[k - 1] - target) / max(a[k - 1] - a[k], 1e-12) * (x1 - x0)))
    return None


def main():
    res = {}
    for name in PROMPTS:
        X, _ = load(name)
        Y, sv = pc_scores(X)
        row = dict(hs=HS, nse_knn=[], nse_ridge=[], nse_persist=[],
                   nse_drift=[], knn_iaaft=[], knn_phase=[], knn_shuffle=[],
                   ridge_iaaft_mean=[], ridge_phase_mean=[], kg={})
        # surrogates in PC-50 space
        ia = [miiaft(Y, rng_for(name, 10 + s)) for s in range(N_IA)]
        ph = [phase_surrogate(Y, rng_for(name, 20 + s)) for s in range(N_PH)]
        sh = [Y[rng_for(name, 30 + s).permutation(Y.shape[0])] for s in range(N_SH)]
        C1 = (Y[:-1].T @ Y[1:]) / (Y.shape[0] - 1)
        fid = [float(np.linalg.norm((z[:-1].T @ z[1:]) / (Y.shape[0] - 1) - C1) / np.linalg.norm(C1))
               for z in ia]
        row["miiaft_crosscov_relerr"] = [round(f, 3) for f in fid]
        for h in HS:
            row["nse_knn"].append(knn_nse(Y, h)[0])
            row["nse_persist"].append(knn_nse(Y, h)[1])
            row["nse_drift"].append(knn_nse(Y, h)[2])
            row["nse_ridge"].append(ridge_nse(Y, h))
            row["knn_iaaft"].append([knn_nse(z, h)[0] for z in ia])
            row["knn_phase"].append([knn_nse(z, h)[0] for z in ph])
            row["knn_shuffle"].append([knn_nse(z, h)[0] for z in sh])
            row["ridge_iaaft_mean"].append(round(float(np.mean([ridge_nse(z, h) for z in ia])), 4))
            row["ridge_phase_mean"].append(round(float(np.mean([ridge_nse(z, h) for z in ph])), 4))
            if h in (1, 8, 16):
                row["kg"][h] = dict(
                    data=kg_stats(Y, h),
                    iaaft=[kg_stats(z, h) for z in ia],
                    phase=[kg_stats(z, h) for z in ph],
                    shuffle=[kg_stats(z, h) for z in sh],
                )
        # gaps + half-skill horizon
        i8 = HS.index(8)
        ia8 = np.mean(row["knn_iaaft"][i8]); ph8 = np.mean(row["knn_phase"][i8]); sh8 = np.mean(row["knn_shuffle"][i8])
        row["gap_iaaft_h8"] = round(row["nse_knn"][i8] - ia8, 4)
        row["gap_best_surr_h8"] = round(row["nse_knn"][i8] - max(ia8, ph8, sh8), 4)
        row["gap_ridge_h8"] = round(row["nse_knn"][i8] - row["nse_ridge"][i8], 4)
        sd8 = float(np.std(row["knn_iaaft"][i8], ddof=1))
        row["z_iaaft_h8"] = round((row["nse_knn"][i8] - ia8) / sd8, 2) if sd8 > 0 else None
        row["h_half_knn"] = h_half(row["nse_knn"])
        row["h_half_ridge"] = h_half(row["nse_ridge"])
        row["h_half_iaaft_med"] = h_half([float(np.mean(v)) for v in row["knn_iaaft"]])
        res[name] = row
        print(f"{name[:34]:36s} knn h1={row['nse_knn'][0]:+.3f} h8={row['nse_knn'][i8]:+.3f} "
              f"ridge h8={row['nse_ridge'][i8]:+.3f} gapIA={row['gap_iaaft_h8']:+.3f} "
              f"gapR={row['gap_ridge_h8']:+.3f} kg_h8={row['kg'][8]['data']['small']:.3f}", flush=True)
        save_json(res, "pred_results.json")
    save_json(res, "pred_results.json")
    make_figs(res)


def make_figs(res):
    # ---- summary numbers ----
    for h_i, h in enumerate(HS):
        d = np.array([res[n]["nse_knn"][h_i] for n in PROMPTS])
        ia = np.array([np.mean(res[n]["knn_iaaft"][h_i]) for n in PROMPTS])
        ph = np.array([np.mean(res[n]["knn_phase"][h_i]) for n in PROMPTS])
        sh = np.array([np.mean(res[n]["knn_shuffle"][h_i]) for n in PROMPTS])
        try:
            p = stats.wilcoxon(d, ia).pvalue
        except ValueError:
            p = np.nan
        print(f"h={h:2d}  knn med {np.median(d):+.3f}  iaft med {np.median(ia):+.3f} "
              f"phase {np.median(ph):+.3f} shuffle {np.median(sh):+.3f}  "
              f"gap med {np.median(d - ia):+.3f}  wilcoxon p={p:.2g}", flush=True)
    make_fig3(res)
    make_fig4(res)


def make_fig3(res):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    gap_traj = max(PROMPTS, key=lambda n: res[n]["gap_iaaft_h8"])
    for ax, name in zip(axes.ravel()[:2], ["interstellar_travel", gap_traj]):
        r = res[name]
        ia = np.array(r["knn_iaaft"]); ph = np.array(r["knn_phase"])
        ax.errorbar(HS, r["nse_knn"], yerr=None, fmt="o-", color="tab:blue", label="data kNN")
        ax.plot(HS, r["nse_ridge"], "s-", color="tab:orange", label="data ridge (linear ceiling)")
        ax.plot(HS, r["nse_persist"], "d-", color="tab:brown", label="persistence $x_{t+h}=x_t$")
        ax.errorbar(HS, ia.mean(1), yerr=ia.std(1), fmt="^-", color="tab:green",
                    capsize=2, label="kNN on MIAAFT (n=5)")
        ax.errorbar(HS, ph.mean(1), yerr=ph.std(1), fmt="v-", color="tab:purple",
                    capsize=2, label="kNN on phase (n=5)")
        sh = np.array(r["knn_shuffle"])
        ax.plot(HS, sh.mean(1), "x-", color="tab:gray", label="kNN on time-shuffle (n=3)")
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(HS); ax.set_xticklabels(HS)
        ax.set_xlabel("horizon h (tokens)")
        ax.set_ylabel("NSE (PC-50 vector, held-out last 30%)")
        ttl = name.replace("_", " ")
        extra = " (most repetitive)" if name == REP else ""
        concl = ("no lead over ridge/surrogates" if r["gap_iaaft_h8"] < 0.10
                 else f"MIAAFT gap +{r['gap_iaaft_h8']:.2f} at h=8")
        ax.set_title(f"{ttl}{extra}: {concl}", fontsize=8)
        ax.legend(fontsize=6.2, loc="upper right")
    ax = axes.ravel()[2]
    LOOPY = {"cybersecurity", "transformer_architecture", "history_of_computing",
             "renewable_energy_economics", "history_of_music", "ocean_currents",
             "sleep_and_memory", "urban_transportation"}
    gaps = np.array([res[n]["gap_iaaft_h8"] for n in PROMPTS])
    order = np.argsort(gaps)
    cols = ["tab:red" if PROMPTS[o] in LOOPY else "tab:blue" for o in order]
    ax.bar(range(16), gaps[order], color=cols)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.15, color="tab:green", ls="--", lw=0.8)
    ax.text(0.2, 0.155, "old 'launch' gap was +0.33 above surrogates", fontsize=6, color="tab:green")
    clean = [res[n]["gap_iaaft_h8"] for n in PROMPTS if n not in LOOPY]
    ax.set_ylabel("NSE kNN(data) - mean kNN(MIAAFT), h=8")
    ax.set_xlabel("trajectory rank (red = repetition loop, blue = non-loopy)")
    ax.set_title(f"(c) MIAAFT gap at h=8: entirely the 8 repetition loops (red, max {gaps.max():+.2f});\n"
                 f"non-loopy 8 (blue) max {max(clean):+.3f} -> launch-style gap NOT replicated", fontsize=7.5)
    ax = axes.ravel()[3]
    gaps = np.array([res[n]["gap_ridge_h8"] for n in PROMPTS])
    order = np.argsort(gaps)
    cols = ["tab:red" if PROMPTS[o] in LOOPY else "tab:orange" for o in order]
    ax.bar(range(16), gaps[order], color=cols)
    ax.axhline(0, color="k", lw=0.8)
    cleanr = [res[n]["gap_ridge_h8"] for n in PROMPTS if n not in LOOPY]
    ax.set_ylabel("NSE kNN(data) - ridge(data), h=8")
    ax.set_xlabel("trajectory rank (red = repetition loop, blue/orange = non-loopy)")
    ax.set_title(f"(d) Nonlinear margin over the data's own linear ceiling at h=8:\n"
                 f"loops up to +{gaps.max():.2f}; non-loopy max {max(cleanr):+.3f} (launch was +0.15)", fontsize=7.5)
    fig.suptitle("Fig 3 - kNN predictability vs linear/surrogate/constant baselines, 16 x 1000-token trajectories", fontsize=9.5)
    fig.savefig(OUT / "fig3_predictability.png")
    plt.close(fig)


def make_fig4(res):
    for n in PROMPTS:
        res[n]["kg"] = {int(k): v for k, v in res[n]["kg"].items()}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    gap_traj = max(PROMPTS, key=lambda n: res[n]["gap_iaaft_h8"])
    # (a) KG ratio curves at h=8, example
    ax = axes[0, 0]
    r = res[gap_traj]["kg"][8]
    qs = np.linspace(0, 100, 9)
    x = np.arange(8) + 0.5
    ax.plot(x, np.array(r["data"]["bins"]), "o-", color="tab:blue", label="data")
    ia = np.array([q["bins"] for q in r["iaaft"]])
    ax.plot(x, np.nanmedian(ia, 0), "^-", color="tab:green", label="MIAAFT med (n=5)")
    ax.fill_between(x, np.nanmin(ia, 0), np.nanmax(ia, 0), color="tab:green", alpha=0.25)
    ax.axhline(1.0, color="k", ls=":", lw=0.8, label="rc = r (no continuation structure)")
    ax.set_xlabel("r bin (deciles of pair distance)")
    ax.set_ylabel("median $r_c / r$ at h = 8")
    ax.set_title(f"(a) Kaplan-Glass continuation, '{gap_traj[:24]}': data vs MIAAFT\n"
                 f"small-r ratio data {r['data']['small']:.3f} vs MIAAFT "
                 f"{np.median([q['small'] for q in r['iaaft']]):.3f} -> no sharpness", fontsize=7.5)
    ax.legend(fontsize=6.5)
    # (b) paired small-r ratio at h=8 across 16
    ax = axes[0, 1]
    d = np.array([res[n]["kg"][8]["data"]["small"] for n in PROMPTS])
    ia = np.array([np.median([q["small"] for q in res[n]["kg"][8]["iaaft"]]) for n in PROMPTS])
    ax.plot([0.8, 1.3], [0.8, 1.3], "k--", lw=0.8)
    ax.plot(ia, d, "o", ms=4)
    ri = PROMPTS.index(REP)
    ax.plot([ia[ri]], [d[ri]], "o", ms=5, color="tab:red")
    ax.annotate("renewable", (ia[ri], d[ri]), fontsize=6, color="tab:red",
                xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("MIAAFT median $r_c/r$ (small-r), h=8")
    ax.set_ylabel("data $r_c/r$ (small-r), h=8")
    dd = d - ia
    ax.set_title(f"(b) Continuation across 16: data rc/r HIGHER than MIAAFT in "
                 f"{(dd > 0).sum()}/16 (median {np.median(dd):+.2f})\n"
                 f"-> no continuation sharpness anywhere; the one exception is renewable's loop", fontsize=7.5)
    # (c) NSE gap vs horizon, across 16
    ax = axes[1, 0]
    gaps = np.array([[res[n]["nse_knn"][i] - np.mean(res[n]["knn_iaaft"][i]) for n in PROMPTS]
                     for i in range(len(HS))])
    gp = np.array([[res[n]["nse_knn"][i] - np.mean(res[n]["knn_phase"][i]) for n in PROMPTS]
                   for i in range(len(HS))])
    ax.errorbar(HS, gaps.mean(1), yerr=gaps.std(1) / np.sqrt(16), fmt="o-", color="tab:green",
                capsize=2, label="kNN data - MIAAFT")
    ax.errorbar(HS, gp.mean(1), yerr=gp.std(1) / np.sqrt(16), fmt="v-", color="tab:purple",
                capsize=2, label="kNN data - phase")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(HS); ax.set_xticklabels(HS)
    ax.set_xlabel("horizon h (tokens)"); ax.set_ylabel("mean NSE gap (16 trajs, +/- s.e.m.)")
    ax.set_title("(c) The gap is positive at every horizon but carried by the 8 repetition\n"
                 "loops (bimodal, s.d. >> s.e.m.) - not a gradual decay to zero", fontsize=7.5)
    ax.legend(fontsize=6.5)
    # (d) half-skill horizon
    ax = axes[1, 1]
    hk = [res[n]["h_half_knn"] for n in PROMPTS]
    hr = [res[n]["h_half_ridge"] for n in PROMPTS]
    hi = [res[n]["h_half_iaaft_med"] for n in PROMPTS]
    lab = ["kNN", "ridge", "kNN on MIAAFT"]
    vals = [hk, hr, hi]
    for k, (v, c) in enumerate(zip(vals, ["tab:blue", "tab:orange", "tab:green"])):
        v = [x if x is not None else 64 for x in v]
        ax.scatter(np.full(16, k) + rng_for(PROMPTS[0], k).uniform(-0.13, 0.13, 16), v,
                   s=10, color=c, label=lab[k])
    ax.axhline(64, color="tab:gray", ls=":", lw=0.8)
    ax.text(2.1, 64, " >32 (not reached)", fontsize=6, color="tab:gray", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels(lab)
    ax.set_ylabel("h at which NSE falls to NSE(h=1)/2 (tokens)")
    ax.set_title("(d) Skill half-life ~1.5-2.3 tokens for the non-loopy 8, equal to ridge/MIAAFT\n"
                 "timescales; loops (dots at 64) never lose half their skill by h=32", fontsize=7.5)
    fig.suptitle("Fig 4 - Continuation sharpness and skill horizons: no Kaplan-Glass advantage, no sharp prediction horizon", fontsize=9.5)
    fig.savefig(OUT / "fig4_sharpness.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
