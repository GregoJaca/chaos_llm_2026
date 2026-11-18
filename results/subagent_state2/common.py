"""Shared loading/style helpers for the state2 subagent analysis."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data'
OUT = r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_state2'

LAYCOL = {7: '#9ecae1', 14: '#4292c4', 21: '#2171b5', 27: '#08306b'}
SCOL = {8: '#8c6d31', 16: '#d62728', 32: '#e7ba52', 50: '#b5cf6b',
        75: '#637939', 100: '#31a354', 150: '#17becf', 200: '#7b4173'}


def load_emb():
    d = np.load(BASE + r'\state_deep_emb\state2.npz')
    dh = d['dh'].astype(np.float64)          # [r, s, dir, layer, t]
    cosv = d['cos'].astype(np.float64)
    dm = d['dm'].astype(np.float64)          # [r, s, dir, t]
    steps, radii, layers = d['steps'], d['radii'], d['layers']
    margins = d['margins'].astype(np.float64)
    # causal mask: hide t < s (also removes small pre-injection artifacts)
    T = dh.shape[-1]
    for i_s, s in enumerate(steps):
        dh[:, i_s, :, :, :s] = np.nan
        cosv[:, i_s, :, :, :s] = np.nan
        dm[:, i_s, :, :s] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # empty slices in causal region
        M = np.nanmean(dh, axis=2)            # [r, s, layer, t] mean over 24 dirs
        MC = np.nanmean(cosv, axis=2)
        MD = np.nanmean(np.abs(dm), axis=2)  # mean |dm| over dirs
    return dict(dh=dh, cos=cosv, dm=dm, M=M, MC=MC, MD=MD,
                steps=steps, radii=radii, layers=layers, margins=margins, T=T)


def load_freerun():
    d = np.load(BASE + r'\state_freerun\sf.npz')
    return dict(pred=d['pred'], ref=d['ref'], fsteps=d['steps'], r=float(d['r']))


def load_l14():
    d = np.load(BASE + r'\state_deep_L14\state2.npz')
    dh = d['dh'].astype(np.float64)           # [r, s, dir, layer, t]
    dm = d['dm'].astype(np.float64)
    steps, radii, layers = d['steps'], d['radii'], d['layers']
    for i_s, s in enumerate(steps):
        dh[:, i_s, :, :, :s] = np.nan
        dm[:, i_s, :, :s] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # empty slices in causal region
        M = np.nanmean(dh, axis=2)
    return dict(dh=dh, dm=dm, M=M, steps=steps, radii=radii, layers=layers,
                margins=d['margins'].astype(np.float64))


def base_state_norms(D):
    """||h_base(t=s, L)|| estimated as dh / sqrt(2(1-cos)) at tau=0, r=0.1."""
    norms = {}
    for i_l, L in enumerate(D['layers']):
        vals = []
        for i_s, s in enumerate(D['steps']):
            c = 1.0 - D['cos'][3, i_s, :, i_l, s]
            ok = c > 8e-7                      # float32 reliability floor for cos (~7 ulps)
            h = D['dh'][3, i_s, ok, i_l, s] / np.sqrt(2 * c[ok])
            vals.append(np.median(h) if ok.sum() > 5 else np.nan)
        norms[int(L)] = np.array(vals)
    return norms


def style():
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 150,
        'font.size': 8.5, 'axes.titlesize': 9.2, 'axes.labelsize': 8.5,
        'legend.fontsize': 7.2, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.w_pad': 0.06, 'figure.constrained_layout.h_pad': 0.06,
        'lines.linewidth': 1.1,
    })


def flips(D, ir=3):
    """dirs (per s) where the top1-top2 gap flips sign: dm[t] < -margins[t]."""
    out = {}
    for i_s, s in enumerate(D['steps']):
        dmm = D['dm'][ir, i_s]
        fl = dmm[:, s + 1:] < -D['margins'][None, s + 1:]
        dirs = np.where(fl.any(axis=1))[0]
        out[int(s)] = [(int(i), int(s + 1 + np.argmax(fl[i]))) for i in dirs]
    return out
