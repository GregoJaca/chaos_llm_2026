"""Soft-loop figure suite v2 (redesigned after user feedback).

Every panel: labeled axes, legends for all elements, colourbars where data
is coloured.  Quantities are defined in new.md; short reminders in titles.

Figures produced (names are v2; v1 files are kept):
  sl2_gate_law.png           the gate law (3 panels: overview / law fit /
                             jump-factor distributions by m/T bin)
  sl2_decomp.png             decomposition with lambda_net on its own panel
                             and the definitional-vs-physical split of the
                             readout trend
  sl2_phase.png              horizon vs T pooled over prompts + horizon vs
                             live-gate density (the collapsing variable)
  sl2_survival_pooled.png     pooled survival curves, all models and T
  sl2_sampling.png           soft vs sampled-CRN vs sampled-independent
  sl2_gate_law_per_model.png measured gate law per model (needs loopval)
  sl2_twonn_vs_T.png         TwoNN dimension vs T per model (needs states)
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'softloop')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'figures', 'softloop')
L_LAST = -1
T_SKIP = 10

CMAP = plt.get_cmap('viridis')
MCOL = {  # consistent model colours across every figure
    'ds15': '#1a6faa', 'ds15v2': '#1a6faa', 'mamba130': '#c8102e',
    'qwen25_3b': '#2e8b57', 'qwen3_4b': '#b8860b', 'phi4mini': '#7a4faa',
    'randqwen': '#888888', 'pythia410': '#444444', 'qwen35_9b': '#d55e00',
}
MLABEL = {
    'ds15v2': 'DeepSeek-1.5B', 'ds15': 'DeepSeek-1.5B', 'mamba130': 'Mamba-130m',
    'qwen25_3b': 'Qwen2.5-3B', 'qwen3_4b': 'Qwen3-4B', 'phi4mini': 'Phi-4-mini',
    'randqwen': 'randQwen g=1', 'pythia410': 'Pythia-410m',
    'qwen35_9b': 'Qwen3.5-9B', 'ds14': 'DeepSeek-14B',
}


def load_run(group, T, mode='loop'):
    d = os.path.join(SRC, group, f'T{T:g}')
    f = os.path.join(d, 'softloop.npz')
    if not os.path.exists(f):
        return None
    return np.load(f)


def all_runs(group):
    out = {}
    for d in sorted(glob.glob(os.path.join(SRC, group, 'T*'))):
        if not os.path.isdir(d):
            continue
        try:
            T = float(os.path.basename(d)[1:])
        except ValueError:
            continue
        f = os.path.join(d, 'softloop.npz')
        if os.path.exists(f):
            out[T] = np.load(f)
    return out


def rate(dh_t, h0_med, t_end=None):
    y = dh_t.astype(np.float64)
    sat = y > 0.25 * h0_med
    ts = int(np.argmax(sat)) if sat.any() else len(y)
    b = ts if t_end is None else min(ts, t_end, len(y))
    if b < 13:
        return np.nan, ts
    b = max(b, 13)
    seg = np.log(np.clip(y[3:b], 1e-12, None))
    if np.ptp(seg) < 1e-9:
        return 0.0, ts
    return float(np.polyfit(np.arange(3, b), seg, 1)[0]), ts


def tsat_all(z):
    dh = z['dh'][:, L_LAST]
    h0 = float(np.median(z['h0n'][L_LAST]))
    S = dh.shape[1]
    return np.array([int(np.argmax(dh[k] > 0.25 * h0)) if (dh[k] > 0.25 * h0).any()
                     else S for k in range(dh.shape[0])]), h0


# ---------------------------------------------------------------- F1: gate law

def fig_gate_law(groups=('ds15v2',), out='sl2_gate_law.png'):
    xs, ys, ts_ = [], [], []
    jumps = []  # (m/T, log jump) pooled, pre-flip
    for g in groups:
        for T, z in sorted(all_runs(g).items()):
            if T < 1e-6:
                continue
            m = z['m_ref'].astype(np.float64)
            s = z['sig_read'].astype(np.float64)
            act = s > 0
            xs.append(m[act] / T)
            ys.append(np.log(s[act]))
            ts_.append(np.full(act.sum(), T))
            dh = z['dh'][:, L_LAST].astype(np.float64)
            fl = z['flips']
            for k in range(dh.shape[0]):
                ff = int(np.argmax(fl[k] > 0)) if (fl[k] > 0).any() else dh.shape[1]
                seg = dh[k, :min(ff, dh.shape[1])]
                if len(seg) < 20:
                    continue
                ratio = seg[1:] / np.clip(seg[:-1], 1e-9, None)
                jmask = ratio > 3.0
                for t in np.where(jmask)[0] + 1:
                    jumps.append((m[t] / T, np.log(ratio[t - 1])))
    x = np.concatenate(xs); y = np.concatenate(ys); Tc = np.concatenate(ts_)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) overview, log x, ONE theory curve, colourbar
    ax = axes[0]
    sc = ax.scatter(x, y, s=2, c=np.log10(Tc), cmap='viridis', alpha=0.35)
    xr = np.logspace(np.log10(max(1e-2, x.min())), np.log10(x.max()), 200)
    p = 1.0 / (1.0 + np.exp(xr))
    theo = np.log(p * (1 - p) * 2.35)
    ax.plot(xr, theo, color='crimson', lw=1.6,
            label=r'two-token softmax identity: $\sigma=p(1{-}p)\,G$, $G=2.35$ (median, no fit)')
    ax.axhline(np.log(2.35 / 4), color='crimson', ls=':', lw=1.0,
               label=r'plateau $G/4$ (max possible $\sigma$ at a 50/50 tie)')
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r'$\log_{10} T$ (temperature of the orbit)', fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel(r'margin / temperature   $m_t/T$   (log scale)')
    ax.set_ylabel(r'$\log \sigma_{\max}(\Sigma_p)$')
    ax.set_title('(a) readout amplifier on every live-gate step,\n'
                 'all temperatures pooled', fontsize=9)
    ax.legend(fontsize=7, loc='lower left')

    # (b) the law, low T, linear x
    ax = axes[1]
    low = (Tc > 1e-6) & (Tc <= 0.3) & (x > 0)
    xx, yy = x[low], y[low]
    sl, ic = np.polyfit(xx, yy, 1)
    r2 = 1 - np.sum((yy - (sl * xx + ic)) ** 2) / np.sum((yy - yy.mean()) ** 2)
    ax.scatter(xx, yy, s=2, c=np.log10(Tc[low]), cmap='viridis', alpha=0.45)
    xr = np.linspace(0, 30, 50)
    ax.plot(xr, sl * xr + ic, color='crimson', lw=1.8,
            label=rf'fit: $\log\sigma={sl:+.2f}\,m/T{ic:+.2f}$   ($R^2={r2:.3f}$, $n={low.sum()}$)')
    ax.plot(xr, -xr + np.log(2.35), 'k--', lw=1.1,
            label=r'theory: $-m/T+\log G$ (slope $-1$)')
    ax.set_xlabel(r'$m_t/T$ (linear scale)')
    ax.set_ylabel(r'$\log \sigma_{\max}(\Sigma_p)$')
    ax.set_ylim(-25, 1)
    ax.set_title('(b) the gate law at $T=0.05...0.3$:\n'
                 r'$\sigma \approx G_{\rm eff}\,e^{-m/T}$', fontsize=9)
    ax.legend(fontsize=8, loc='lower left')

    # (c) jump-factor distributions by m/T bin (redesigned)
    ax = axes[2]
    if jumps:
        jm = np.array(jumps)
        bins = [(0, 0.5), (0.5, 2), (2, 10), (10, np.inf)]
        data, labels = [], []
        for lo, hi in bins:
            sel = (jm[:, 0] >= lo) & (jm[:, 0] < hi)
            if sel.sum():
                data.append(jm[sel, 1])
                labels.append(f'$m/T\\in[{lo},{hi if hi < 90 else "\\infty"})$\n$n={sel.sum()}$')
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#f2b8b8')
        for med in bp['medians']:
            med.set_color('#c8102e')
            med.set_linewidth(2)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_ylabel(r'$\log\left(\|\delta h_t\|/\|\delta h_{t-1}\|\right)$  at burst steps')
        ax.set_title('(c) response-burst amplitude by gate tightness\n'
                     'bursts ($\\times3^+$ jumps, pre-flip, per direction)', fontsize=9)
        for i, dd in enumerate(data):
            ax.text(i + 1, np.median(dd) + 0.15, f'med {np.median(dd):+.1f}',
                    ha='center', fontsize=7, color='#c8102e')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=150)
    plt.close(fig)
    return dict(law_slope=sl, law_intercept=ic, law_R2=r2, n_law=int(low.sum()))


# ------------------------------------------------------------- F2: decomp v2

def fig_decomp(group='ds15v2', out='sl2_decomp.png'):
    rows = json.load(open(os.path.join(OUT, 'sl_numbers.json')))[group]
    Ts = np.array([r['T'] for r in rows])
    sep = np.array([r['lam_sep'] for r in rows])
    seplo = np.array([r['lam_sep_iqr'][0] for r in rows])
    sephi = np.array([r['lam_sep_iqr'][1] for r in rows])
    pre = np.array([r['lam_pre'] if r['lam_pre'] is not None else np.nan for r in rows])
    net = np.array([r.get('lam_net', np.nan) for r in rows])
    netlo = np.array([r.get('lam_net_iqr', [np.nan, np.nan])[0] for r in rows])
    nethi = np.array([r.get('lam_net_iqr', [np.nan, np.nan])[1] for r in rows])
    act = np.array([r['active_frac'] for r in rows])
    ro = np.array([r['readout_active'] if r['readout_active'] is not None
                   else np.nan for r in rows])
    hz = np.array([min(r['tsat_med'], 620) for r in rows])

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))

    ax = axes[0]
    ax.errorbar(Ts, sep, yerr=[sep - seplo, sephi - sep], fmt='o-', color='#1a6faa',
                label=r'$\lambda_{sep}$ (loop, to saturation, $r=0.03$)')
    ax.plot(Ts, pre, 'd-', color='#7a4faa', ms=4,
            label=r'$\lambda_{pre}$ (before the first argmax flip)')
    ax.scatter([1.0], [0.056], marker='*', s=170, color='#2e8b57', zorder=5,
               label=r'$\lambda_{sep}$ at $r=0.003$ (linear regime, $r$-independent)')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('temperature $T$')
    ax.set_ylabel('nats / step')
    ax.set_title('(a) separation rates of the loop\n'
                 'sep and pre coincide when flips are late; both are positive in\n'
                 'the band because they mix smooth growth and gate straddles —\n'
                 'the green star is the $r\\to0$ smooth part alone', fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.errorbar(Ts, net, yerr=[net - netlo, nethi - net], fmt='s-', color='#c8102e')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('temperature $T$')
    ax.set_ylabel('nats / step')
    ax.set_ylim(-0.03, 0.03)
    ax.set_title('(b) network-only rate $\\lambda_{net}$ (replay)\n'
                 'own scale: never leaves $[-0.002, +0.001]$ at any $T$\n'
                 'the network alone cannot destabilise the loop', fontsize=8)

    ax = axes[2]
    ax.plot(Ts, ro, 'o-', color='#b8860b',
            label=r'measured: $\langle\log(\sigma_{\max}/T)\rangle$ at live gates')
    tref = np.log(2.35 / 4) - np.log(Ts)
    ax.plot(Ts, tref, '--', color='gray',
            label=r'definitional ceiling: $\log(G/4)-\log T$'
                  '\n($\sigma$ is plateau-bounded; only $T$ grows)')
    ax.set_xscale('log')
    ax.set_xlabel('temperature $T$')
    ax.set_ylabel('nats')
    ax.set_title('(c) readout gain at live gates\n'
                 'the fall with $T$ is mostly the $-\\log T$ definition;\n'
                 'the non-trivial part is the gap below the ceiling\n'
                 '(weaker gates, opening gradually)', fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(Ts, act, 'o-', color='#2e8b57', label='live-gate fraction ($\\sigma_{\\max}>T$)')
    ax.plot(Ts, 620 - hz, 's-', color='#1a6faa',
            label='620 $-$ median separation step (inverted horizon)')
    ax.set_xscale('log')
    ax.set_xlabel('temperature $T$')
    ax.legend(fontsize=7)
    ax.set_title('(d) what actually moves: live-gate density\n'
                 'and the horizon (inverted so up = faster separation)', fontsize=8)
    fig.suptitle(f'{MLABEL.get(group, group)} soft-generation loop, decomposition v2', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- F3: phase v2

def fig_phase(group='ds15v2', out='sl2_phase.png'):
    runs = all_runs(group)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    allpts = []
    for T, z in sorted(runs.items()):
        ts, h0 = tsat_all(z)
        S = z['dh'].shape[1]
        ok = ts < S
        pts = ts[ok] if ok.any() else np.array([S])
        allpts.append((T, pts))
        ax.scatter(np.full(len(pts), T), np.clip(pts, 1, 650), s=13,
                   color=CMAP(list(runs).index(T) / max(len(runs) - 1, 1)),
                   alpha=0.75, zorder=3)
        if not ok.any():
            ax.scatter([T], [640], marker='^', s=60, color='k', zorder=4)
    med = [np.median(np.clip(p, 1, 650)) for _, p in allpts]
    ax.plot([T for T, _ in allpts], med, 'k-', lw=1.2, zorder=2,
            label='median separation step')
    ax.axhline(600, color='gray', ls=':', lw=0.9)
    ax.text(1.2e-9, 615, 'censored: no direction separated in 600 steps', fontsize=7)
    ax.text(1.2e-9, 470, '$\\blacktriangle$ = all directions survived', fontsize=7)
    ax.set_xscale('log')
    ax.set_ylim(0, 660)
    ax.set_xlabel('temperature $T$')
    ax.set_ylabel('separation step $t_{sat}$ per direction')
    ax.set_title('(a) predictability horizon vs $T$\n'
                 'scatter = individual directions; black = median;\n'
                 'colour runs cold (blue) $\\to$ hot (yellow)', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')

    # (b) horizon vs live-gate density — the collapsing variable
    ax = axes[1]
    X, Y = [], []
    for T, z in sorted(runs.items()):
        m = z['m_ref'].astype(np.float64)
        if T < 1e-6:
            continue
        ts, _ = tsat_all(z)
        S = z['dh'].shape[1]
        med = np.median(np.clip(ts, 1, 650))
        X.append(float((m < 3 * T).mean()))
        Y.append(med)
        ax.scatter(float((m < 3 * T).mean()), med, s=45,
                   color=CMAP(list(runs).index(T) / max(len(runs) - 1, 1)), zorder=3)
    X = np.array(X); Y = np.array(Y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('fraction of orbit steps with $m_t < 3T$  (live-gate density)')
    ax.set_ylabel('median separation step')
    r = np.corrcoef(np.log(np.clip(X, 1e-4, 1)), np.log(Y))[0, 1] if len(X) > 3 else np.nan
    ax.set_title(f'(b) same data against the mediator:\nlive-gate density '
                 f'(log-log corr = {r:.2f})', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=150)
    plt.close(fig)


# ------------------------------------------------------ F4: pooled survival

def fig_survival_pooled(groups, out='sl2_survival_pooled.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for g in groups:
        for T, z in sorted(all_runs(g).items()):
            if T < 1e-6:
                continue
            ts, _ = tsat_all(z)
            S = z['dh'].shape[1]
            surv = [float(np.mean(ts > t)) for t in range(0, S, 10)]
            ax.plot(np.arange(0, S, 10), surv, lw=1.4,
                    color=MCOL.get(g, '#333'), alpha=0.35 + 0.6 * min(T, 1.5) / 1.5)
    ax.set_xlabel('step $t$')
    ax.set_ylabel('fraction of directions not yet separated')
    handles = [Line2D([0], [0], color=MCOL.get(g, '#333'), lw=2,
                      label=MLABEL.get(g, g)) for g in groups]
    handles += [Line2D([0], [0], color='k', alpha=0.3, lw=2, label='light = colder, dark = hotter')]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_title('(a) pooled survival curves, every model and $T>0$', fontsize=9)

    ax = axes[1]
    for g in groups:
        X, Y, C = [], [], []
        for T, z in sorted(all_runs(g).items()):
            if T < 1e-6:
                continue
            m = z['m_ref'].astype(np.float64)
            ts, _ = tsat_all(z)
            S = z['dh'].shape[1]
            X.append(float((m < 3 * T).mean()))
            Y.append(np.median(np.clip(ts, 1, 650)))
            C.append(T)
        if X:
            mk = {'ds15v2': 'o', 'mamba130': 's', 'qwen25_3b': '^',
                  'qwen3_4b': 'D', 'phi4mini': 'v', 'randqwen': 'P',
                  'pythia410': 'X', 'ds15': 'o'}.get(g, 'o')
            sc = ax.scatter(X, Y, s=42, c=np.log10(C), cmap='viridis',
                            marker=mk, label=MLABEL.get(g, g))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('live-gate density: fraction of steps with $m_t < 3T$')
    ax.set_ylabel('median separation step')
    ax.legend(fontsize=7)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r'$\log_{10} T$', fontsize=8)
    ax.set_title('(b) all models: horizon vs live-gate density', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------- F5: sampling modes

def fig_sampling(out='sl2_sampling.png'):
    base = os.path.join(os.path.dirname(SRC), 'softloop_sample')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    specs = [
        ('ds15/soft_T0.5', 'soft, T=0.5', '#1a6faa', '-'),
        ('ds15/soft_T1.0', 'soft, T=1.0', '#1a6faa', '--'),
        ('ds15/crn_T0.3', 'sampled-CRN, T=0.3', '#2e8b57', '-'),
        ('ds15/crn_T0.5', 'sampled-CRN, T=0.5', '#2e8b57', '--'),
        ('ds15/crn_T1.0', 'sampled-CRN, T=1.0', '#2e8b57', ':'),
        ('ds15/ind_T0.5', 'sampled-independent, T=0.5', '#c8102e', '-'),
        ('ds15/ind_T1.0', 'sampled-independent, T=1.0', '#c8102e', '--'),
    ]
    for tag, lab, col, ls in specs:
        f = os.path.join(base, 'ds15', tag.replace('soft_', '') + '.npz') \
            if False else os.path.join(base, tag)
        # soft runs live in results/softloop/ds15v2; sampling runs in softloop_sample
        if tag.startswith('soft_'):
            f = os.path.join(SRC, 'ds15v2', tag.split('_')[1], 'softloop.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        dh = np.median(z['dh'][:, L_LAST], axis=0)
        ax.plot(np.arange(len(dh)), np.clip(dh, 1e-4, None), color=col, ls=ls,
                lw=1.6, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('step $t$')
    ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
    ax.set_title('(a) three feedback rules, same model and prompt\n'
                 'soft = expected embedding; CRN = sampled tokens shared by all\n'
                 'rows; independent = each row samples its own token', fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    labels, med, mx = [], [], []
    for d in sorted(glob.glob(os.path.join(base, 'ds15', '*'))):
        if not os.path.isdir(d):
            continue
        f = os.path.join(d, 'softloop.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ts, h0 = tsat_all(z)
        S = z['dh'].shape[1]
        ok = ts < S
        name = os.path.basename(d)
        labels.append(name)
        med.append(np.median(ts[ok]) if ok.any() else S)
        mx.append(int(ok.sum()))
    if labels:
        ax.bar(np.arange(len(labels)), med, color=['#2e8b57' if 'crn' in l else '#c8102e'
                                                   for l in labels])
        ax.axhline(600, color='gray', ls=':')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('median separation step')
        ax.set_title('(b) sampled runs: horizons (600 = censored)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=150)
    plt.close(fig)


# ------------------------------------------------- F6: per-model measured law

def fig_gate_law_per_model(out='sl2_gate_law_per_model.png'):
    valdir = os.path.join(os.path.dirname(SRC), 'softloop_val')
    groups = sorted(glob.glob(os.path.join(valdir, '*')))
    groups = [g for g in groups if os.path.isdir(g)]
    if not groups:
        print('no loopval data yet; skip', out)
        return
    n = len(groups)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow),
                             squeeze=False)
    for i, g in enumerate(groups):
        ax = axes[i // ncol][i % ncol]
        name = os.path.basename(g)
        xs, ys, ts_ = [], [], []
        for d in sorted(glob.glob(os.path.join(g, 'T*'))):
            if not os.path.isdir(d):
                continue
            f = os.path.join(d, 'softloop.npz')
            if not os.path.exists(f):
                continue
            T = float(os.path.basename(d)[1:])
            z = np.load(f)
            m = z['m_ref'].astype(np.float64)
            s = z['sig_read'].astype(np.float64)
            act = s > 0
            xs.append(m[act] / T)
            ys.append(np.log(s[act]))
            ts_.append(np.full(act.sum(), T))
        if not xs:
            ax.axis('off')
            continue
        x = np.concatenate(xs); y = np.concatenate(ys)
        sel = (x > 0) & (x < 30)
        ax.scatter(x[sel], y[sel], s=3, c=np.log10(np.concatenate(ts_))[sel],
                   cmap='viridis', alpha=0.5)
        if sel.sum() > 20:
            sl, ic = np.polyfit(x[sel], y[sel], 1)
            r2 = 1 - np.sum((y[sel] - (sl * x[sel] + ic)) ** 2) / \
                np.sum((y[sel] - y[sel].mean()) ** 2)
            xr = np.linspace(0, 30, 20)
            ax.plot(xr, sl * xr + ic, 'crimson', lw=1.6,
                    label=rf'slope ${sl:+.2f}$, $G_{{eff}}={np.exp(ic):.2f}$, $R^2={r2:.3f}$')
            ax.plot(xr, -xr + np.log(2.35), 'k--', lw=1.0, label='theory (slope $-1$)')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(r'$m_t/T$')
        ax.set_ylabel(r'$\log\sigma_{\max}(\Sigma_p)$')
        ax.legend(fontsize=6, loc='lower left')
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('measured gate law per model (loop validation runs, $T=0.05...1.0$)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=150)
    plt.close(fig)


# --------------------------------------------------------- F7: TwoNN vs T

def fig_twonn(out='sl2_twonn_vs_T.png'):
    st = os.path.join(os.path.dirname(SRC), 'softloop_states')
    jsonp = os.path.join(OUT, 'd2pipe_softloop.json')
    if not os.path.exists(jsonp):
        print('no d2pipe json; skip', out)
        return
    d = json.load(open(jsonp))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for tr in d['trajectories']:
        name = tr['name'].replace('loop_', '')
        model = 'ds15'
        if 'mamba' in name:
            model = 'mamba130'
        elif 'qwen25' in name:
            model = 'qwen25_3b'
        T = name.split('T')[-1]
        try:
            Tv = float(T)
        except ValueError:
            continue
        tw = tr['twonn']['mle']
        ax.scatter(Tv, tw, s=55, color=MCOL.get(model, '#333'),
                   marker={'ds15': 'o', 'mamba130': 's', 'qwen25_3b': '^'}.get(model, 'o'))
        ax.annotate(f'{model},T={T}', (Tv, tw), fontsize=6,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('temperature $T$')
    ax.set_ylabel('TwoNN dimension of the loop attractor (MLE)')
    handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor=MCOL.get(g, '#333'),
                      markersize=9, label=MLABEL.get(g, g))
               for g, m in [('ds15', 'o'), ('mamba130', 's'), ('qwen25_3b', '^')]]
    ax.legend(handles=handles, fontsize=7)
    ax.set_title('intrinsic dimension (TwoNN) of the loop state cloud vs temperature\n'
                 'n=3000 states per run, last layer', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=150)
    plt.close(fig)


# ------------------------------------------- F8: horizon vs live-gate density

def fig_horizon_density(groups, out='sl2_horizon_vs_density.png'):
    """Every model x temperature: median separation step against the
    live-gate density of ITS OWN orbit margins (measured) and against the
    A1-corpus predicted density (no loop run)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    mks = {'ds15v2': 'o', 'mamba130': 's', 'qwen25_3b': '^', 'randqwen': 'P',
           'pythia410': 'X', 'qwen3_4b': 'D', 'phi4mini': 'v', 'qwen3_5': 'p',
           'qwen35_9b': '*'}
    # A1-corpus predicted density per model (from sl2_gates_table.json + T)
    tabp = os.path.join(OUT, 'sl2_gates_table.json')
    pred = {}
    if os.path.exists(tabp):
        tab = json.load(open(tabp))
        # we need the full margin ECDF; approximate from stored percentiles is
        # not enough -> recompute from the gates files
        for k in tab:
            path = os.path.join(os.path.dirname(SRC), 'softloop_gates', k)
            if not os.path.isdir(path):
                path = os.path.join(os.path.dirname(SRC), '..', 'results_big',
                                    k, 'softloop_gates')
            fs = glob.glob(os.path.join(path, '*', 'gates.npz'))
            if fs:
                ms = np.concatenate([np.load(f)['m'] for f in fs])
                pred[k] = ms
    used = []
    for g in groups:
        for T, z in sorted(all_runs(g).items()):
            if T < 1e-6:
                continue
            m = z['m_ref'].astype(np.float64)
            dens = float((m < 3 * T).mean())
            ts, _ = tsat_all(z)
            S = z['dh'].shape[1]
            ok = ts < S
            med = np.median(ts[ok]) if ok.any() else S
            col = MCOL.get(g, '#333')
            axes[0].scatter(dens, med, s=60, marker=mks.get(g, 'o'), color=col,
                            zorder=3, label=MLABEL.get(g, g) if g not in used else None)
            # predicted density from the A1 corpus (if available for this model)
            key = g.replace('v2', '')
            if key in pred and T in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
                dpred = float((pred[key] < 3 * T).mean())
                axes[1].scatter(dpred, med, s=60, marker=mks.get(g, 'o'),
                                color=col, zorder=3)
        used.append(g)
    for ax, ttl in [(axes[0], '(a) horizon vs live-gate density of the '
                                 'loop\'s own orbit'),
                    (axes[1], '(b) horizon vs the density predicted from the '
                              'greedy A1 corpora (one pass, no loop)')]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('fraction of steps with $m_t < 3T$')
        ax.set_ylabel('median separation step')
        ax.set_title(ttl, fontsize=9)
        ax.axhline(600, color='gray', ls=':', lw=0.9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default='ds15v2,mamba130,qwen25_3b')
    args = ap.parse_args()
    groups = [g for g in args.groups.split(',') if os.path.isdir(os.path.join(SRC, g))]
    stats = fig_gate_law(groups=('ds15v2',))
    print('gate law:', stats)
    fig_decomp('ds15v2')
    fig_phase('ds15v2')
    if groups:
        fig_survival_pooled(groups)
        fig_horizon_density(groups)
    fig_sampling()
    fig_gate_law_per_model()
    fig_twonn()
    print('v2 figures saved to', OUT)


if __name__ == '__main__':
    main()
