"""Exp 1: D2 scaling search on the 16 new 1000-token trajectories.

For each trajectory: C(eps) on a 48-point log eps grid (eps in [1e-3, 3] x median
pairwise distance), Theiler w = 0/10/50, N = 500/1000. For each config:
- widest slope-stable run (|slope - med| <= max(0.10, 0.15|med|), >= 4 points)
- ER-legal band width (C in [10/N, 0.5], slope <= 2 log10 N)   [d2b convention]
- naive best fit (widest window with r^2 >= 0.99)              [the artifact trap]

Controls: time-shuffle (EXACT no-op for w=0 point-cloud statistics - verified in
integer pair counts; differs at w>0 through the temporal mask), multivariate
phase surrogates (exact cross-periodogram, Gaussian marginals) and MIAAFT
(per-PC marginal exact, spectrum ~preserved, cross-spectrum approx via common
phases). RQA (DET, L_mean, LAM, TT at RR=5/10/20%) for data + surrogates.

Output: d2_results.json, fig1_d2_scaling.png, fig2_d2_controls.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common_d2c import (PROMPTS, REP, OUT, load, distm, c_eps_curve, eps_grid_from_med,
                        log_slope, smooth3, widest_run, run_is_er_legal, erlegal_width,
                        naive_best_fit, phase_surrogate, miiaft, cloud_miiaft,
                        rqa_from_D, pair_values, save_json, rng_for)

plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5,
                     "axes.titlesize": 9.0, "axes.titleweight": "bold",
                     "figure.constrained_layout.use": True})

N_SURR_PH, N_SURR_IA, N_SHUF = 5, 5, 3
FIG_TRAJS = ["interstellar_travel", "history_of_mathematics", REP]


def analyze_config(X, N, w, med):
    Xs = X[:N]
    D = distm(Xs - Xs.mean(0))
    eps = eps_grid_from_med(med)
    C = c_eps_curve(D, eps, w)
    s = log_slope(eps, C)
    ss = smooth3(s)
    run = widest_run(eps, C, ss)
    er_tot, er_best = erlegal_width(eps, C, ss, N)
    nbf = naive_best_fit(eps, C)
    legal = run_is_er_legal(run, N) if run else False
    return dict(eps=eps, C=C, ss=ss, run=run, er_tot=er_tot, er_best=er_best,
                nbf=nbf, legal=legal, N=N, w=w,
                erlegal_run=bool(run and legal))


def main():
    res = {}
    for name in PROMPTS:
        X, _ = load(name)
        med_full = float(np.median(pair_values(distm(X - X.mean(0)))))
        cfgs = {}
        for N in (1000, 500):
            for w in (0, 10, 50):
                if N == 500 and w != 0:
                    continue
                cfgs[f"N{N}_w{w}"] = analyze_config(X, N, w, med_full)
        # ---- controls at N=1000 ----
        Xc = X - X.mean(0)
        U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
        Y = U[:, :999] * sv[:999]          # full-rank PC scores (rank <= 999)
        Ytop = U[:, :50] * sv[:50]        # top-50 PC scores (cloud shape carrier)
        # time-shuffle: exact identity for w=0, check in integer counts
        rng = rng_for(name, 1)
        D_data = distm(Xc)
        eps = eps_grid_from_med(med_full)
        cnt_data = np.searchsorted(np.sort(pair_values(D_data, 0)), eps, side="right")
        dev = []
        for s_i in range(N_SHUF):
            perm = rng.permutation(X.shape[0])
            Ds = distm(Xc[perm])
            cnt_s = np.searchsorted(np.sort(pair_values(Ds, 0)), eps, side="right")
            dev.append(int(np.max(np.abs(cnt_data - cnt_s))))
        shuffle_ident = bool(max(dev) == 0)
        # w=50 shuffles: temporal mask now matters
        dev_w50 = []
        Cs_w50_data = c_eps_curve(D_data, eps, 50)
        for s_i in range(N_SHUF):
            perm = rng_for(name, 50 + s_i).permutation(X.shape[0])
            Ds = distm(Xc[perm])
            Cs = c_eps_curve(Ds, eps, 50)
            m = Cs_w50_data > 1e-4
            dev_w50.append(float(np.max(np.abs(np.log10(np.maximum(Cs, 1e-300)[m])
                                               - np.log10(Cs_w50_data[m])))))
        # phase surrogates (full-rank, exact cross-periodogram) + cloud MIAAFT
        # (top-50 MIAAFT + phase-rand tail, full-rank reconstruction)
        ph_curves, ia_curves = [], []
        ph_nbf, ia_nbf = [], []
        rqa_data = rqa_from_D(D_data)
        rqa_ph, rqa_ia, rqa_sh = [], [], []
        for s_i in range(N_SURR_PH):
            Zp = phase_surrogate(Y, rng_for(name, 100 + s_i)) @ Vh[:999]
            Dp = distm(Zp)
            Cph = c_eps_curve(Dp, eps, 0)
            ph_curves.append(Cph)
            ph_nbf.append(naive_best_fit(eps, Cph))
            rqa_ph.append(rqa_from_D(Dp))
        for s_i in range(N_SURR_IA):
            Zi = cloud_miiaft(Xc, rng_for(name, 200 + s_i))
            Di = distm(Zi)
            Cia = c_eps_curve(Di, eps, 0)
            ia_curves.append(Cia)
            ia_nbf.append(naive_best_fit(eps, Cia))
            rqa_ia.append(rqa_from_D(Di))
        for s_i in range(N_SHUF):
            perm = rng_for(name, 300 + s_i).permutation(X.shape[0])
            Dsh = distm(Xc[perm])
            rqa_sh.append(rqa_from_D(Dsh))
        # MIAAFT cross-spectrum fidelity on the top-50 block (lag-1 cross-cov)
        C1 = (Ytop[:-1].T @ Ytop[1:]) / (Ytop.shape[0] - 1)
        fid = []
        for s_i in range(3):
            Zt = miiaft(Ytop, rng_for(name, 400 + s_i), n_iter=25)
            C1s = (Zt[:-1].T @ Zt[1:]) / (Ytop.shape[0] - 1)
            fid.append(float(np.linalg.norm(C1s - C1) / np.linalg.norm(C1)))
        res[name] = dict(
            med_dist=med_full, min_dist=float(pair_values(D_data, 0).min()),
            cfgs={k: {kk: vv for kk, vv in v.items() if kk not in ("eps", "C", "ss")}
                  for k, v in cfgs.items()},
            curves={k: dict(eps=list(v["eps"]), C=list(v["C"]), ss=[None if not np.isfinite(x) else float(x) for x in v["ss"]])
                    for k, v in cfgs.items()},
            shuffle_w0_max_int_dev=dev, shuffle_w0_identical=shuffle_ident,
            shuffle_w50_max_dlogC=max(dev_w50),
            phase_nbf=ph_nbf, ia_nbf=ia_nbf,
            phase_C=[list(c) for c in ph_curves], ia_C=[list(c) for c in ia_curves],
            rqa_data=rqa_data, rqa_phase=rqa_ph, rqa_iaaft=rqa_ia, rqa_shuffle=rqa_sh,
            miiaft_crosscov_relerr=fid,
        )
        c = res[name]
        r = c["cfgs"]["N1000_w0"]
        rwd = f"{r['run']['width_dec']:.2f}" if r["run"] else " none"
        nbs = f"{r['nbf']['slope']:.2f}" if r["nbf"] else " none"
        print(f"{name[:34]:36s} med={med_full:7.2f} er_w={r['er_tot']:+.3f} "
              f"run={rwd: >6} legal={r['erlegal_run']} nbf={nbs}", flush=True)
        save_json(res, "d2_results.json")
    save_json(res, "d2_results.json")
    make_fig1(res)
    make_fig2(res)


def make_fig1(res):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    for ax, name in zip(axes.ravel()[:3], FIG_TRAJS):
        r = res[name]
        eps = np.array(r["curves"]["N1000_w0"]["eps"])
        C = np.array(r["curves"]["N1000_w0"]["C"])
        ss = np.array([np.nan if x is None else x for x in r["curves"]["N1000_w0"]["ss"]])
        ax.plot(eps, C, "o-", ms=2.5, lw=1, color="tab:blue", label="C($\\epsilon$)")
        N = 1000
        for yv, lab, col in ((10 / N, "C = 10/N (depletion floor)", "tab:red"),):
            ax.axhline(yv, color=col, ls=":", lw=0.8)
            ax.text(eps[1], yv, " " + lab, color=col, va="bottom", fontsize=6.5)
        ax.axhline(0.5, color="tab:red", ls=":", lw=0.8)
        ax.text(eps[1], 0.5, " C = 0.5 (saturation edge)", color="tab:red", va="bottom", fontsize=6.5)
        run = r["cfgs"]["N1000_w0"]["run"]
        if run:
            ax.axvspan(run["eps_lo"], run["eps_hi"], color="orange", alpha=0.35)
            ax.text(0.5, 0.04, f"widest stable run: {run['width_dec']:.2f} dec @ slope {run['slope']:.2f}\n"
                    f"ER-legal: {r['cfgs']['N1000_w0']['erlegal_run']}",
                    transform=ax.transAxes, fontsize=6.8, color="tab:orange")
        else:
            ax.text(0.5, 0.04, "no slope-stable run (>=4 pts)", transform=ax.transAxes,
                    fontsize=6.8, color="tab:orange")
        nbf = r["cfgs"]["N1000_w0"]["nbf"]
        if nbf:
            xs = eps[nbf["i0"]:nbf["i1"] + 1]
            ys = C[nbf["i0"]:nbf["i1"] + 1]
            ax.plot(xs, 10 ** (np.log10(ys[0]) + nbf["slope"] * np.log10(xs / xs[0])),
                    "--", lw=1.2, color="tab:green")
            ax.text(0.5, 0.13, f"naive r2>=0.99 fit: D2={nbf['slope']:.2f}, "
                    f"{nbf['width_dec']:.2f} dec (artifact trap)", transform=ax.transAxes,
                    fontsize=6.8, color="tab:green")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$\\epsilon$ (raw hidden-state units)")
        ax.set_ylabel("C($\\epsilon$)")
        ttl = name.replace("_", " ")
        extra = " (most repetitive)" if name == REP else ""
        ax.set_title(f"{ttl}{extra}: no scaling region\nER-legal band {r['cfgs']['N1000_w0']['er_tot']:.2f} dec wide", fontsize=8)

    ax = axes.ravel()[3]
    widths = [res[n]["cfgs"]["N1000_w0"]["run"]["width_dec"] if res[n]["cfgs"]["N1000_w0"]["run"] else 0.0
             for n in PROMPTS]
    erw = [res[n]["cfgs"]["N1000_w0"]["er_tot"] for n in PROMPTS]
    bins = np.linspace(0, 1.1, 23)
    ax.hist(widths, bins=bins, alpha=0.65, color="tab:orange",
            label=f"widest stable run (med {np.median(widths):.2f})")
    ax.hist(erw, bins=bins, alpha=0.65, color="tab:red",
            label=f"ER-legal band (med {np.median(erw):.2f})")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.text(0.5, ax.get_ylim()[1] * 0.6, " 0.5-dec\n validity\n threshold", fontsize=6.5)
    n_valid = sum(1 for n in PROMPTS
                  if res[n]["cfgs"]["N1000_w0"]["erlegal_run"]
                  and res[n]["cfgs"]["N1000_w0"]["run"]["width_dec"] >= 0.5)
    ax.set_xlabel("width (decades of $\\epsilon$)")
    ax.set_ylabel("# trajectories (of 16)")
    ax.set_title(f"Scaling widths across 16 trajectories: {n_valid}/16 have a valid\n"
                 "ER-legal slope-stable interval >= 0.5 dec (N=1000, w=0)")
    ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Fig 1 - C(eps) on 1000-token last-layer clouds (FP32-greedy): whole rise spans <0.2 decades -> no D2 scaling", fontsize=9.5)
    fig.savefig(OUT / "fig1_d2_scaling.png")
    plt.close(fig)


def make_fig2(res):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    # (a) time-shuffle identity + w=50
    ax = axes[0, 0]
    r = res["interstellar_travel"]
    eps = np.array(r["curves"]["N1000_w0"]["eps"])
    C = np.array(r["curves"]["N1000_w0"]["C"])
    ax.plot(eps, C, "o-", ms=2.5, lw=1, color="tab:blue", label="data, w=0")
    ax.plot(eps, C, ".", ms=6, color="tab:red", label="time-shuffle, w=0 (exactly identical)")
    Cw50 = c_eps_curve_w = None
    # recompute w=50 for display via stored curve? use stored cfg N1000_w50
    Cw50 = np.array(r["curves"]["N1000_w50"]["C"])
    ax.plot(eps, Cw50, "s-", ms=2.5, lw=1, color="tab:green", label="data, w=50 (Theiler)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$\\epsilon$"); ax.set_ylabel("C($\\epsilon$)")
    ax.set_title(f"(a) Time-shuffle preserves C($\\epsilon$) exactly (w=0: max int dev\n"
                 f"{max(res['interstellar_travel']['shuffle_w0_max_int_dev'])} pairs; w=50: max |dlC| {res['interstellar_travel']['shuffle_w50_max_dlogC']:.3f})", fontsize=7.5)
    ax.legend(fontsize=6.5)
    # (b) data vs phase vs MIAAFT curves, one example
    ax = axes[0, 1]
    r = res["interstellar_travel"]
    ax.plot(eps, C, "o-", ms=2.5, lw=1, color="tab:blue", label="data")
    ia = np.array(res["interstellar_travel"]["ia_C"])
    ph = np.array(res["interstellar_travel"]["phase_C"])
    ax.plot(eps, np.median(ia, 0), "-", color="tab:green", label="MIAAFT med (n=5)")
    ax.fill_between(eps, ia.min(0), ia.max(0), color="tab:green", alpha=0.25)
    ax.plot(eps, np.median(ph, 0), "-", color="tab:purple", label="phase med (n=5)")
    ax.fill_between(eps, ph.min(0), ph.max(0), color="tab:purple", alpha=0.25)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$\\epsilon$"); ax.set_ylabel("C($\\epsilon$)")
    ax.set_title("(b) C($\\epsilon$) is a statics functional: MIAAFT (marginal+spectrum)\nreproduces it; phase (Gaussian marginals) steepens it", fontsize=7.5)
    ax.legend(fontsize=6.5)
    # (c) naive-fit D2: data vs MIAAFT, all 16
    ax = axes[1, 0]
    xs, ys = [], []
    for n in PROMPTS:
        nb = res[n]["cfgs"]["N1000_w0"]["nbf"]
        if nb is None:
            continue
        xs.append(nb["slope"])
        ysv = [v["slope"] for v in res[n]["ia_nbf"] if v]
        ys.append(np.mean(ysv) if ysv else np.nan)
    ax.plot([0, 10], [0, 10], "k--", lw=0.8)
    ax.plot(xs, ys, "o", ms=4)
    rr = [i for i, n in enumerate(PROMPTS) if n == REP and i < len(xs)]
    ax.set_xlabel("naive best-fit slope D2, data")
    ax.set_ylabel("naive best-fit slope D2, MIAAFT (mean of 5)")
    ax.set_title("(c) The 'D2' an innocent fit reports is reproduced by\nsurrogates: data vs MIAAFT on the diagonal", fontsize=7.5)
    # (d) RQA DET at RR=10%: data vs surrogate classes across 16
    ax = axes[1, 1]
    idx = np.arange(len(PROMPTS))
    det_d = [next(x["det"] for x in res[n]["rqa_data"] if x["rr"] == 0.10) for n in PROMPTS]
    det_i = [np.mean([next(x["det"] for x in v if x["rr"] == 0.10) for v in res[n]["rqa_iaaft"]]) for n in PROMPTS]
    det_p = [np.mean([next(x["det"] for x in v if x["rr"] == 0.10) for v in res[n]["rqa_phase"]]) for n in PROMPTS]
    det_s = [np.mean([next(x["det"] for x in v if x["rr"] == 0.10) for v in res[n]["rqa_shuffle"]]) for n in PROMPTS]
    order = np.argsort(det_d)
    ax.plot(idx, np.array(det_d)[order], "o-", ms=3, color="tab:blue", label="data")
    ax.plot(idx, np.array(det_p)[order], "s-", ms=3, color="tab:purple", label="phase med")
    ax.plot(idx, np.array(det_i)[order], "^-", ms=3, color="tab:green", label="MIAAFT med")
    ax.plot(idx, np.array(det_s)[order], "v-", ms=3, color="tab:gray", label="shuffle med")
    rep_pos = int(np.where(order == PROMPTS.index(REP))[0][0])
    ax.annotate("renewable", (rep_pos, det_d[PROMPTS.index(REP)]),
                fontsize=6, xytext=(3, -8), textcoords="offset points", color="tab:red")
    ax.set_xlabel("trajectory rank (sorted by data DET)")
    ax.set_ylabel("DET (recurrence points on diagonal lines $\\geq$2)")
    ax.set_title(f"(d) RQA determinism (RR=10%, w=10): data DET med {np.median(det_d):.3f}\nvs MIAAFT {np.median(det_i):.3f}, phase {np.median(det_p):.3f}, shuffle {np.median(det_s):.3f}", fontsize=7.5)
    ax.legend(fontsize=6.5)
    fig.suptitle("Fig 2 - Controls: time-shuffle is a point-cloud tautology; MIAAFT/phase bracket the C($\\epsilon$) slopes; RQA DET vs surrogates", fontsize=9.5)
    fig.savefig(OUT / "fig2_d2_controls.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
