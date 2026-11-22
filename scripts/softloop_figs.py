"""Analysis + figures for the soft-generation loop campaign.  v2

Objects measured per temperature T (see run_softloop.py for the run protocol):
  lam_sep(T)  OLS slope of log||dh_t|| (last layer) over t in [10, t_sat);
              t_sat = first t with ||dh|| > 0.25*median||h0|| (saturation =
              the two rows sit on different orbit branches).  This is the
              finite-time separation exponent, realised as a bursty staircase.
  lam_pre(T)  the same restricted to t < first argmax flip (the strictly
              pre-flip smooth part, incl. readout bursts but excl. post-flip
              text divergence).  Defined only if the segment has >= 8 steps.
  lam_net(T)  the same on the replay run (identical soft tokens for every
              row: pure network propagation, no feedback).
  gate curve  per-step readout factor sigma_max(CrossCov_p(e,w));
              theory (2-token split): sigma ~ p(1-p) * ||dE||*||dW|| with
              p = sigmoid(-m/T), so log(sigma) should collapse vs m/T.

Figures:
  sl_response_<g>.png   per-T response curves with flip ticks
  sl_decomp_<g>.png     lambda curves + gate statistics
  sl_gate_collapse.png  log(sigma_read) vs m/T, all T pooled  [the mechanism]
  sl_models.png         lambda_loop(T) across model groups
  sl_numbers.json       every quoted number
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L_LAST = -1
T_SKIP = 10
MIN_SEG = 25


def load_group(group_dir):
    out = {}
    for d in sorted(glob.glob(os.path.join(group_dir, "T*"))):
        if not os.path.isdir(d):
            continue
        try:
            T = float(os.path.basename(d)[1:])
        except ValueError:
            continue
        entry = {}
        f = os.path.join(d, "softloop.npz")
        if os.path.exists(f):
            entry["loop"] = np.load(f)
        f = os.path.join(d, "replay", "softloop.npz")
        if os.path.exists(f):
            entry["replay"] = np.load(f)
        if entry:
            out[T] = entry
    return out


def dir_rate(dh_t, h0_med, t_end=None):
    y = dh_t.astype(np.float64)
    sat = y > 0.25 * h0_med
    ts = int(np.argmax(sat)) if sat.any() else len(y)
    b = ts if t_end is None else min(ts, t_end, len(y))
    if b < 13:            # saturation arrived before a fit is meaningful
        return np.nan, ts
    b = max(b, 13)
    seg = np.log(np.clip(y[3:b], 1e-12, None))
    if np.ptp(seg) < 1e-9:
        return 0.0, ts
    return float(np.polyfit(np.arange(3, b), seg, 1)[0]), ts


def group_stats(z):
    dh = z["dh"]
    h0n = z["h0n"]
    K, NL, S = dh.shape
    h0_med = float(np.median(h0n[L_LAST]))
    rates, tsats, fflip = [], [], []
    for k in range(K):
        f = int(np.argmax(z["flips"][k] > 0)) if (z["flips"][k] > 0).any() else -1
        fflip.append(f)
        r, ts = dir_rate(dh[k, L_LAST], h0_med)
        rates.append(r)
        tsats.append(ts)
    pre = []
    for k in range(K):
        tend = fflip[k] if fflip[k] > 0 else S
        r, _ = dir_rate(dh[k, L_LAST], h0_med, t_end=tend)
        pre.append(r)
    return dict(rates=np.array(rates), pre=np.array(pre), tsats=tsats,
                fflip=fflip, h0_med=h0_med, dh=dh, m=z["m_ref"].astype(np.float64),
                ent1=z["ent_ref"].astype(np.float64),
                sig=z["sig_read"].astype(np.float64))


def analyse(group_dir):
    data = load_group(group_dir)
    rows, curves, gates, jumps = [], {}, [], []
    for T in sorted(data.keys()):
        entry = data[T]
        st = group_stats(entry["loop"])
        sig = st["sig"]
        active = sig > 0  # which steps enter the gate-collapse scatter
        row = dict(
            T=T,
            lam_sep=float(np.nanmedian(st["rates"])),
            lam_sep_iqr=[float(np.nanpercentile(st["rates"], 25)),
                         float(np.nanpercentile(st["rates"], 75))],
            lam_pre=float(np.nanmedian(st["pre"])) if np.isfinite(st["pre"]).any() else None,
            n_sat=int(sum(1 for x in st["tsats"] if x < st["dh"].shape[2])),
            tsat_med=float(np.median([x for x in st["tsats"]])) if st["tsats"] else None,
            flip_frac=float(np.mean([f >= 0 for f in st["fflip"]])),
            fflip_med=float(np.median([f for f in st["fflip"] if f >= 0]))
                      if any(f >= 0 for f in st["fflip"]) else None,
            m_med=float(np.median(st["m"])),
            m_min=float(st["m"].min()),
            ent1_med=float(np.median(st["ent1"])),
            active_frac=float((sig[T_SKIP:] > T).mean()),
            gate_budget=float(np.log(np.maximum(sig[T_SKIP:300] / T, 1.0)).mean()),
            readout_active=float(np.log(sig[T_SKIP:300][(sig[T_SKIP:300] > T)] / T).mean())
                           if (sig[T_SKIP:300] > T).any() else None,
        )
        if "replay" in entry:
            stn = group_stats(entry["replay"])
            row["lam_net"] = float(np.nanmedian(stn["rates"]))
            row["lam_net_iqr"] = [float(np.nanpercentile(stn["rates"], 25)),
                                  float(np.nanpercentile(stn["rates"], 75))]
        rows.append(row)
        curves[T] = st
        # gate-jump samples: at steps where the response jumps by >x5 in one
        # step, record (log sigma_read[t-1]/T, log jump factor)
        dhm = np.median(st["dh"][:, L_LAST], axis=0)   # median response curve
        ratio = dhm[1:] / np.clip(dhm[:-1], 1e-9, None)
        jump = ratio > 5.0
        if jump.any():
            sigm = st["sig"]
            lg = np.log(np.clip(sigm[:-1][jump] / T, 1e-12, None))
            jr = np.log(ratio[jump])
            jumps.append(np.stack([lg, jr, np.full(lg.shape, T)]))
        # gate-curve samples (only active steps, cap for plot size)
        mm = st["m"][T_SKIP:300]
        ss = sig[T_SKIP:300]
        act = active[T_SKIP:300]
        if act.any():
            gates.append(np.stack([mm[act], np.log(np.clip(ss[act], 1e-12, None)),
                                   np.full(act.sum(), T)]))
    gate_xyzt = np.concatenate(gates, axis=1) if gates else np.zeros((3, 0))
    jump_xyzt = np.concatenate(jumps, axis=1) if jumps else np.zeros((3, 0))
    return rows, curves, gate_xyzt, jump_xyzt


def fig_response_grid(rows, curves, out, title):
    Ts = [r["T"] for r in rows]
    n = len(Ts)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.4 * nrow), squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i // ncol][i % ncol]
        st = curves[r["T"]]
        dh = st["dh"][:, L_LAST]
        ax.plot(np.median(dh, axis=0), lw=1.4, color="#1a6faa")
        ax.fill_between(np.arange(dh.shape[1]), np.percentile(dh, 25, axis=0),
                        np.percentile(dh, 75, axis=0), alpha=0.25, color="#1a6faa")
        ax.axhline(0.25 * st["h0_med"], ls="--", lw=0.8, color="gray")
        for f in st["fflip"]:
            if f >= 0:
                ax.axvline(f, color="red", lw=0.6, alpha=0.5)
        ax.set_yscale("log")
        tag = f"T={r['T']:g}  sep={r['lam_sep']:+.3f}"
        if r.get("lam_pre") is not None:
            tag += f"  pre={r['lam_pre']:+.3f}"
        ax.set_title(tag, fontsize=8)
        ax.tick_params(labelsize=7)
        if i // ncol == nrow - 1:
            ax.set_xlabel("step t", fontsize=8)
        if i % ncol == 0:
            ax.set_ylabel(r"$\|\delta h_{t,L}\|$", fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title + "  (dashed: saturation level; red: first flips)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_decomp(rows, out, title):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    Ts = np.array([r["T"] for r in rows])
    sep = np.array([r["lam_sep"] for r in rows])
    seplo = np.array([r["lam_sep_iqr"][0] for r in rows])
    sephi = np.array([r["lam_sep_iqr"][1] for r in rows])
    pre = np.array([r["lam_pre"] if r["lam_pre"] is not None else np.nan for r in rows])
    net = np.array([r.get("lam_net", np.nan) for r in rows])
    netlo = np.array([r.get("lam_net_iqr", [np.nan, np.nan])[0] for r in rows])
    nethi = np.array([r.get("lam_net_iqr", [np.nan, np.nan])[1] for r in rows])
    act = np.array([r["active_frac"] for r in rows])
    ro = np.array([r["readout_active"] if r["readout_active"] is not None
                   else np.nan for r in rows])

    ax = axes[0]
    ax.errorbar(Ts, sep, yerr=[sep - seplo, sephi - sep], fmt="o-", color="#1a6faa",
                label=r"$\lambda_{sep}$ (loop, to saturation)")
    ax.plot(Ts, pre, "d-", color="#7a4faa", label=r"$\lambda_{pre}$ (pre-first-flip)")
    ax.plot(Ts, net, "s-", color="#c8102e", label=r"$\lambda_{net}$ (replay: no feedback)")
    ax.fill_between(Ts, netlo, nethi, alpha=0.2, color="#c8102e")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("nats / step")
    ax.legend(fontsize=7)
    ax.set_title("(a) separation rates")

    ax = axes[1]
    ax.plot(Ts, act, "o-", color="#2e8b57")
    ax.set_xscale("log")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("fraction of steps with $\\sigma_{\\max}(\\Sigma_p)>0$")
    ax.set_title("(b) active readout fraction")

    ax = axes[2]
    ax.plot(Ts, ro, "o-", color="#b8860b")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("temperature T")
    ax.set_ylabel(r"$\langle\log(\sigma_{\max}(\Sigma_p)/T)\rangle$ (active steps)")
    ax.set_title("(c) readout gain on active steps")

    ax = axes[3]
    ax.plot(Ts, [r["m_med"] for r in rows], "o-", label="median margin $m_t$")
    ax.plot(Ts, [r["m_min"] for r in rows], "v-", label="min margin")
    ax.plot(Ts, [r["ent1_med"] for r in rows], "s-", label="median entropy at T=1 scale")
    ax.set_xscale("log")
    ax.set_xlabel("temperature T")
    ax.legend(fontsize=7)
    ax.set_title("(d) margin landscape of the loop orbit")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_gate(gate_xyzt, jump_xyzt, out):
    m, logsig, T = gate_xyzt
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    x = m / T
    sel = (x > 0) & np.isfinite(logsig)
    ax.scatter(x[sel], logsig[sel], s=2, c=np.log10(T[sel]), cmap="viridis", alpha=0.4)
    xs = np.logspace(np.log10(max(x[sel].min(), 1e-2)), np.log10(x[sel].max()), 50)
    for q, ls in [(50, "-"), (90, "--"), (10, "--")]:
        p = 1.0 / (1.0 + np.exp(xs))
        base = np.log(p * (1 - p)) - np.log(p * (1 - p)).max()
        ax.plot(xs, base + np.percentile(logsig[sel], q), ls, color="crimson",
                lw=1.2 if ls == "-" else 0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"margin / temperature  $m_t/T$")
    ax.set_ylabel(r"$\log \sigma_{\max}(\Sigma_p)$")
    ax.set_title("(a) readout amplifier vs margin/temperature (all T)")

    # (b) the law, low-T range, LINEAR x: log sigma = -(m/T) + log G_eff
    ax = axes[1]
    lowT = (T > 1e-6) & (T <= 0.3) & (m / T > 0) & np.isfinite(logsig)
    xx = (m / T)[lowT]
    yy = logsig[lowT]
    sl, ic = np.polyfit(xx, yy, 1)
    r2 = 1 - np.sum((yy - (sl * xx + ic)) ** 2) / np.sum((yy - yy.mean()) ** 2)
    ax.scatter(xx, yy, s=2, c=np.log10(T[lowT]), cmap="viridis", alpha=0.45)
    xr = np.linspace(0, min(xx.max(), 30), 50)
    ax.plot(xr, sl * xr + ic, color="crimson", lw=1.5,
            label=rf"fit: $\log\sigma = {sl:+.2f}\,m/T {ic:+.2f}$   $R^2={r2:.3f}$")
    ax.plot(xr, -xr + np.log(2.35), "k--", lw=1.0,
            label=r"theory: $-m/T + \log G$ (2-token, $G=2.35$)")
    ax.set_xlabel(r"$m_t/T$ (linear)")
    ax.set_ylabel(r"$\log \sigma_{\max}(\Sigma_p)$")
    ax.set_ylim(-25, 1)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(b) the gate law, live gates at T = 0.05...0.3 (n=%d)" % lowT.sum())

    ax = axes[2]
    if jump_xyzt.size:
        lg, jr, Tj = jump_xyzt
        ax.scatter(lg, jr, s=14, c=np.log10(Tj), cmap="viridis")
        lim = [min(lg.min(), jr.min()) - 1, max(lg.max(), jr.max()) + 1]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel(r"$\log(\sigma_{\max}(\Sigma_p)/T)$ at the gate step")
        ax.set_ylabel(r"$\log(\|\delta h_{t}\|/\|\delta h_{t-1}\|)$")
        r = np.corrcoef(lg, jr)[0, 1] if len(lg) > 2 else np.nan
        ax.set_title(f"(c) observed jump vs available gain (r={r:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_jumps(group_dir, out):
    """Jump-step vs non-jump-step distribution of m/T (pre-first-flip)."""
    xs_j, xs_n = [], []
    for d in sorted(glob.glob(os.path.join(group_dir, "T*"))):
        if not os.path.isdir(d):
            continue
        try:
            T = float(os.path.basename(d)[1:])
        except ValueError:
            continue
        if T < 1e-6:
            continue
        z = np.load(os.path.join(d, "softloop.npz"))
        m = z["m_ref"].astype(np.float64)
        dh = z["dh"][:, L_LAST].astype(np.float64)
        flips = z["flips"]
        for k in range(dh.shape[0]):
            ff = int(np.argmax(flips[k] > 0)) if (flips[k] > 0).any() else dh.shape[1]
            seg = dh[k, :min(ff, dh.shape[1])]
            if len(seg) < 20:
                continue
            ratio = seg[1:] / np.clip(seg[:-1], 1e-9, None)
            for t in range(1, len(seg)):
                if ratio[t - 1] > 3.0:
                    xs_j.append(m[t] / T)
                else:
                    xs_n.append(m[t] / T)
    if not xs_j:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.logspace(-2, 2.3, 31)
    ax.hist(np.clip(xs_n, 1e-2, 200), bins=bins, density=True, alpha=0.55,
            color="#7f7f7f", label=f"non-jump steps (n={len(xs_n)})")
    ax.hist(np.clip(xs_j, 1e-2, 200), bins=bins, density=True, alpha=0.65,
            color="#c8102e", label=f"gate steps: $\\|\\delta h_t\\|/\\|\\delta h_{{t-1}}\\|>3$ (n={len(xs_j)})")
    ax.set_xscale("log")
    ax.set_xlabel(r"margin / temperature  $m_t/T$")
    ax.set_ylabel("step density (norm.)")
    ax.set_title(f"response bursts sit at $m_t \\lesssim T$: "
                 f"median $m/T$ = {np.median(xs_j):.2f} at gates vs {np.median(xs_n):.1f} elsewhere")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_survival(rows, curves, out, title):
    """Fraction of directions whose response has not saturated, vs step,
    one curve per temperature (the predictability-horizon picture)."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    cmap = plt.get_cmap("viridis")
    Ts = sorted([r["T"] for r in rows])
    for i, r in enumerate(rows):
        st = curves[r["T"]]
        dh = st["dh"][:, L_LAST]
        S = dh.shape[1]
        tsats = np.array([ts if ts < S else S for ts in st["tsats"]], dtype=np.int64)
        surv = [float(np.mean(tsats > t)) for t in range(0, S, 5)]
        color = cmap(i / max(len(Ts) - 1, 1))
        ax.plot(np.arange(0, S, 5), surv, lw=1.5, color=color,
                label=f"T={r['T']:g}")
    ax.set_xlabel("step t")
    ax.set_ylabel("fraction of directions not yet separated")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(title + ": predictability horizon of the soft loop")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_models(allrows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in allrows.items():
        Ts = [r["T"] for r in rows]
        axes[0].plot(Ts, [r["lam_sep"] for r in rows], "o-", label=name)
        axes[1].plot(Ts, [r["active_frac"] for r in rows], "s-", label=name)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("temperature T")
        ax.legend(fontsize=8)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel(r"$\lambda_{sep}$ (nats/step)")
    axes[0].set_title("separation rate vs temperature")
    axes[1].set_ylabel("active readout fraction")
    axes[1].set_title("fraction of live-gate steps")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="chaos_llm_2026/results/softloop")
    ap.add_argument("--out", default="chaos_llm_2026/results/figures/softloop")
    ap.add_argument("--groups", default="ds15,ds15_w100,mamba130,qwen25_3b")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    allrows, gates_all, jumps_all = {}, [], []
    for g in args.groups.split(","):
        gd = os.path.join(args.src, g)
        if not os.path.isdir(gd):
            print("skip (missing):", gd)
            continue
        rows, curves, gate, jump = analyse(gd)
        if not rows:
            continue
        allrows[g] = rows
        gates_all.append(gate)
        jumps_all.append(jump)
        gtag = g.replace("/", "_")
        fig_response_grid(rows, curves, os.path.join(args.out, f"sl_response_{gtag}.png"), g)
        fig_decomp(rows, os.path.join(args.out, f"sl_decomp_{gtag}.png"), g)
        fig_jumps(gd, os.path.join(args.out, f"sl_jumps_{gtag}.png"))
        fig_survival(rows, curves, os.path.join(args.out, f"sl_survival_{gtag}.png"), g)
        print(g, "->", len(rows), "temperatures")
    if allrows:
        fig_models(allrows, os.path.join(args.out, "sl_models.png"))
        if gates_all:
            fig_gate(np.concatenate(gates_all, axis=1),
                     np.concatenate(jumps_all, axis=1) if jumps_all else np.zeros((3, 0)),
                     os.path.join(args.out, "sl_gate_collapse.png"))
    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(os.path.join(args.out, "sl_numbers.json"), "w") as f:
        json.dump(allrows, f, indent=1, default=_default)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
