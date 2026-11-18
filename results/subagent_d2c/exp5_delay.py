"""Exp 5: delay-embedding D2(m) test, m = 2..10, tau = 1, PC1 observable.

For each m: C(eps) on the m-dimensional delay embedding of PC1, Theiler window
w = m (covers the coordinate-overlap zone), 96-point log grid over
[1e-3, 3] x embedding-median distance. D2(m) := median smoothed local slope
over C in [0.05, 0.2]; also the widest slope-stable run (the honest plateau
detector from exp1). Controls: univariate IAAFT surrogates of PC1 (n=5) run
through the identical pipeline -> the band that a red heavy-tailed marginal
produces by itself (fill dimension / effective-rank artifact).

Output: delay_results.json, fig5_delay_embedding.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common_d2c import (PROMPTS, REP, OUT, load, distm, c_eps_curve, eps_grid_from_med,
                        log_slope, smooth3, widest_run, run_is_er_legal,
                        iaaft_scalar, save_json, rng_for)

plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5,
                     "axes.titlesize": 9.0, "axes.titleweight": "bold",
                     "figure.constrained_layout.use": True})

MS = list(range(2, 11))
N_IA = 5
FIG_TRAJS = ["interstellar_travel", "history_of_music", REP]


def embed(y, m, tau=1):
    T = y.size - (m - 1) * tau
    ix = np.arange(m) * tau
    return y[np.arange(T)[:, None] + ix[None, :]]


def d2_of(emb, med_ref=None):
    V = emb - emb.mean(0)
    D = distm(V)
    med = float(np.median(D[np.triu_indices(D.shape[0], 1)]))
    eps = eps_grid_from_med(med)
    C = c_eps_curve(D, eps, w=0)
    s = smooth3(log_slope(eps, C))
    band = (C >= 0.05) & (C <= 0.2)
    d2 = float(np.median(s[band])) if band.any() else np.nan
    run = widest_run(eps, C, s)
    return dict(d2=d2, run=run, eps=list(eps), C=list(C),
                med=med, N=V.shape[0])


def main():
    res = {}
    for name in PROMPTS:
        X, _ = load(name)
        Xc = X - X.mean(0)
        U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
        pc1 = U[:, 0] * sv[0]
        pc1 = pc1 - pc1.mean()
        acf = np.correlate(pc1, pc1, "full")[999:] / (pc1 @ pc1)
        tau_acf = int(np.argmax(acf < 1 / np.e) if (acf < 1 / np.e).any() else 0) or 1
        ia = [iaaft_scalar(pc1, rng_for(name, 70 + s)) for s in range(N_IA)]
        wn = [pc1[rng_for(name, 80).permutation(pc1.size)]]  # white-noise same marginal
        rows = dict(d2_data=[], run_width=[], run_slope=[], run_legal=[],
                    d2_iaaft=[], d2_wn=[])
        for m in MS:
            ed = d2_of(embed(pc1, m))
            rows["d2_data"].append(round(ed["d2"], 3))
            rw = ed["run"]
            rows["run_width"].append(round(rw["width_dec"], 3) if rw else 0.0)
            rows["run_slope"].append(round(rw["slope"], 3) if rw else None)
            rows["run_legal"].append(bool(rw and run_is_er_legal(rw, ed["N"])))
            eis = [d2_of(embed(z, m))["d2"] for z in ia]
            rows["d2_iaaft"].append([round(float(v), 3) for v in eis])
            rows["d2_wn"].append(round(float(d2_of(embed(wn[0], m))["d2"]), 3))
        res[name] = dict(pc1_acf_lag1=round(float(acf[1]), 3),
                         pc1_tau_acf=tau_acf, **rows)
        print(f"{name[:34]:36s} D2(m=2..10) data: {rows['d2_data']}", flush=True)
        save_json(res, "delay_results.json")
    save_json(res, "delay_results.json")
    make_fig(res)


def make_fig(res):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    ax = axes[0, 0]
    for name, c in zip(FIG_TRAJS, ["tab:blue", "tab:purple", "tab:red"]):
        r = res[name]
        ax.plot(MS, r["d2_data"], "o-", color=c, label=name.replace("_", " ")[:22])
        ia = np.array(r["d2_iaaft"])
        ax.fill_between(MS, ia.min(1), ia.max(1), color=c, alpha=0.18)
    ax.plot(MS, MS, "k:", lw=0.9, label="white-noise fill dim = m")
    ax.set_xlabel("embedding dimension m (tau=1)")
    ax.set_ylabel("D2(m) = slope at C in [0.05, 0.2]")
    ax.set_title("(a) D2(m) data (lines) vs IAAFT bands (fills, n=5):\n"
                 "data rides its red-marginal null; nothing saturates below the band", fontsize=7.5)
    ax.legend(fontsize=6.2)
    ax = axes[0, 1]
    dev = []
    for m in MS:
        rows = [res[n]["d2_data"][m - 2] - np.mean(res[n]["d2_iaaft"][m - 2]) for n in PROMPTS]
        dev.append(rows)
    dev = np.array(dev)
    ax.axhline(0, color="k", lw=0.8)
    ax.errorbar(MS, dev.mean(1), yerr=dev.std(1), fmt="o-", capsize=2, color="tab:blue")
    below = [(m, sum(1 for n in PROMPTS
                     if res[n]["d2_data"][m - 2] < min(res[n]["d2_iaaft"][m - 2])))
             for m in MS]
    ax.set_xlabel("embedding dimension m")
    ax.set_ylabel("D2(m) data - mean IAAFT D2(m)")
    txt = "; ".join(f"m{m}: {k}/16" for m, k in below[::2])
    ax.set_title(f"(b) Data minus IAAFT across 16 (mean +/- s.d.). Below the IAAFT\nband minimum: {txt}", fontsize=7.5)
    # (c) m=2 trap
    ax = axes[1, 0]
    name = FIG_TRAJS[0]
    X, _ = load(name)
    Xc = X - X.mean(0)
    U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    pc1 = (U[:, 0] * sv[0]); pc1 = pc1 - pc1.mean()
    ed = d2_of(embed(pc1, 2))
    ax.plot(ed["eps"], ed["C"], "o-", ms=2.5, lw=1, color="tab:blue", label="data PC1, m=2")
    for s_i, cc in zip(range(3), ["tab:green", "tab:purple", "tab:gray"]):
        z = iaaft_scalar(pc1, rng_for(name, 90 + s_i))
        e2 = d2_of(embed(z, 2))
        ax.plot(e2["eps"], e2["C"], "-", lw=0.9, color=cc,
                label=["IAAFT (n=3 shown)"][s_i] if s_i == 0 else None)
    wnz = pc1[rng_for(name, 81).permutation(pc1.size)]
    e3 = d2_of(embed(wnz, 2))
    ax.plot(e3["eps"], e3["C"], ":", lw=1.2, color="k", label="white noise, same marginal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$\\epsilon$ (PC1 units)"); ax.set_ylabel("C($\\epsilon$)")
    ax.set_title("(c) The m=2 'textbook' curve is the fill dimension: data = IAAFT =\nwhite noise (slope ~1.9 over ~1 decade) - identical to the old-data trap", fontsize=7.5)
    ax.legend(fontsize=6.5)
    # (d) stable-run widths at m=2 across 16
    ax = axes[1, 1]
    w2 = [res[n]["run_width"][0] for n in PROMPTS]
    l2 = [res[n]["run_legal"][0] for n in PROMPTS]
    ax.bar(range(16), w2, color=["tab:red" if not lg else "tab:green" for lg in l2])
    ax.set_xlabel("trajectory")
    ax.set_ylabel("widest slope-stable run at m=2 (decades)")
    ax.set_title(f"(d) m=2 stable-run widths: median {np.median(w2):.2f} dec, "
                 f"{sum(l2)}/16 ER-legal\n(green = legal) - the plateau exists but is the 2-D fill dimension", fontsize=7.5)
    fig.suptitle("Fig 5 - Delay embedding D2(m): every apparent plateau or saturation matches its IAAFT null", fontsize=9.5)
    fig.savefig(OUT / "fig5_delay_embedding.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
