"""Exp2: synthetic-loop control for the three repetition-loop trajectories.

S1 (cycle-faithful): x_syn(t) = m_{p(t)} + c(t)*g_{p(t)} + e_perm(t)
    -- period-P cycle of phase-means, per-phase linear drift, residuals
       randomly re-assigned (marginal preserved, residual dynamics destroyed).
S2 (phase-shuffled): same phase means/drift/residual pool, but the sequence of
    phases is a random permutation -> identical statics, NO temporal flow.

Compare C(eps) curves (S1 vs data) and kNN prediction skill (S1/S2 vs data).
If S1 reproduces the data's C-curve and prediction skill, the 'D2' and the
'determinism' of the loopy set are fully explained by trivial repetition.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from common import _block_dist_sq

OUT = os.path.dirname(os.path.abspath(__file__))
LOOPY = {'cyber': 53, 'future': 74, 'great': 69}
H = [1, 2, 4, 8, 16, 32]

def build_loop_model(X, P):
    n = len(X)
    nc = n // P
    use = nc * P
    Xv = X[:use].astype(np.float64)
    cyc = np.arange(use) // P
    ph = np.arange(use) % P
    # per-phase linear-in-cycle fit: x = a_p + c*b_p + e (OLS -> e mean-zero per phase)
    m = np.zeros((P, X.shape[1]))
    g = np.zeros((P, X.shape[1]))
    resid = np.empty((use, X.shape[1]))
    for p in range(P):
        sel = ph == p
        A = np.c_[cyc[sel], np.ones(int(sel.sum()))]
        coef, *_ = np.linalg.lstsq(A, Xv[sel], rcond=None)
        g[p] = coef[0]
        m[p] = coef[1]
        resid[sel] = Xv[sel] - (m[p][None, :] + cyc[sel][:, None] * g[p][None, :])
    return m, g, resid, cyc, ph

def synth_S1(X, P, m, g, resid, rng):
    n0 = len(X)
    use = len(resid)
    nc = use // P
    e = resid[rng.permutation(use)]
    t = np.arange(use)
    return (m[t % P] + (t // P)[:, None] * g[t % P] + e).astype(np.float32)


def resid_cycle_autocorr(resid, P):
    """Median lag-1 autocorrelation (across cycles, per phase) of residuals."""
    use = len(resid)
    nc = use // P
    R = resid[:nc * P].reshape(nc, P, -1)
    A = R[:-1] - R[:-1].mean(0, keepdims=True)  # [nc-1, P, d] centered per phase
    B = R[1:] - R[1:].mean(0, keepdims=True)
    num = (A * B).sum(0)  # [P, d]
    den = A.std(0) * B.std(0) * (nc - 1) + 1e-12
    rho_d = num / den
    return float(np.clip(np.median(rho_d), -0.99, 0.999))


def synth_S1b(X, P, m, g, resid, rho, rng):
    """Cycle + drift + AR(1)-in-cycle-index residuals (persistent deviations).

    Reproduces the tight repeat fidelity of the data (cross-cycle correlated
    residuals) with a trivial linear stochastic mechanism.
    """
    use = len(resid)
    nc = use // P
    R = resid[:nc * P].reshape(nc, P, -1)
    sd = R.std(0)  # per-phase residual sd [P, d]
    E = np.empty_like(R)
    E[0] = rng.standard_normal((P, R.shape[2])) * sd
    for c in range(1, nc):
        E[c] = rho * E[c - 1] + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.standard_normal((P, R.shape[2])) * sd
    e = E.reshape(use, -1)
    t = np.arange(use)
    return (m[t % P] + (t // P)[:, None] * g[t % P] + e).astype(np.float32)

def synth_S2(X, P, m, g, resid, rng):
    use = len(resid)
    nc = use // P
    e = resid[rng.permutation(use)]
    # random order of phases: each cycle visits the P phases in a random order
    # (keeps the statics: all phase means x nc cycles, drift per cycle index)
    ph_seq = np.array([rng.permutation(P) for _ in range(nc)])
    ph = ph_seq.ravel()
    t = np.arange(use)
    cyc = t // P
    return (m[ph] + cyc[:, None] * g[ph] + e).astype(np.float32)

res = {}
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
figP, axesP = plt.subplots(1, 3, figsize=(17, 4.2))

for name, P in LOOPY.items():
    t0 = time.time()
    X = load_traj(name)
    n = len(X)
    scale = median_pairwise_dist(X)
    eps = np.geomspace(0.002 * scale, 2.2 * scale, 80)
    m, g, resid, cyc, ph = build_loop_model(X, P)
    # residual scale vs cloud
    res_rms = float(np.sqrt((resid ** 2).sum(1).mean()) / scale)
    rho_res = resid_cycle_autocorr(resid, P)
    # variance explained by cycle model
    xvar = float(((X - X.mean(0)) ** 2).sum(1).mean())
    rvar = float((resid ** 2).sum(1).mean())
    rng = np.random.default_rng(123 + hash(name) % 100)
    S1 = synth_S1(X, P, m, g, resid, rng)
    S1b = synth_S1b(X, P, m, g, resid, rho_res, rng)
    S2 = synth_S2(X, P, m, g, resid, rng)

    C_data, _ = c_curve(X, eps, theiler=0)
    C_s1, _ = c_curve(S1, eps, theiler=0)
    C_s1b, _ = c_curve(S1b, eps, theiler=0)
    C_s2, _ = c_curve(S2, eps, theiler=0)
    sd = slope_at_C(eps, C_data)
    s1 = slope_at_C(eps, C_s1)
    s1b = slope_at_C(eps, C_s1b)
    s2 = slope_at_C(eps, C_s2)

    # repeat fidelity of the synthetics (should match data's ~2%)
    def fidelity(Z, L):
        d = np.linalg.norm(Z[L:].astype(np.float64) - Z[:-L].astype(np.float64), axis=1)
        return float(np.median(d) / scale)

    # prediction skill: data vs S1 vs S1b vs S2 (test half, W=200)
    W = 200
    r = {'P': P, 'resid_rms_over_scale': res_rms,
         'resid_cycle_autocorr': rho_res,
         'frac_var_explained_by_cycle': float(1 - rvar / xvar),
         'fidelity': {'data': fidelity(X, P), 'S1': fidelity(S1, P),
                      'S1b': fidelity(S1b, P), 'S2': fidelity(S2, P)},
         'slopes_at_C': {'data': sd, 'S1': s1, 'S1b': s1b, 'S2': s2}, 'pred': {}}
    for tag, Z in [('data', X), ('S1', S1), ('S1b', S1b), ('S2', S2)]:
        nb_i, nb_d = knn_neighbors(Z, k=20, W=W, test_slice=(len(Z) // 2, len(Z) - max(H) - 1))
        sk = predict_from_neighbors(Z, nb_i, nb_d, H, k=12, test_offset=len(Z) // 2)
        ds = {h: determinism_score(Z, nb_i, nb_d, h, test_offset=len(Z) // 2)[0] for h in (1, 8, 16)}
        r['pred'][tag] = {'skill': {str(h): sk[h] for h in H}, 'det_score': ds}
    res[name] = r

    print(f'{name} P={P}: resid_rms={res_rms:.4f} scale, rho_res={rho_res:.3f}, cycle explains '
          f'{r["frac_var_explained_by_cycle"]*100:.2f}% of var ({time.time()-t0:.0f}s)')
    print('  fidelity data/S1/S1b/S2:', {k: round(v, 4) for k, v in r['fidelity'].items()})
    print('  slopes data:', {k: round(float(v), 2) for k, v in sd.items()})
    print('  slopes S1b :', {k: round(float(v), 2) for k, v in s1b.items()})
    print('  slopes S2  :', {k: round(float(v), 2) for k, v in s2.items()})
    print('  NSE data:', {h: round(r['pred']['data']['skill'][str(h)]['nse_knn'], 3) for h in H})
    print('  NSE S1  :', {h: round(r['pred']['S1']['skill'][str(h)]['nse_knn'], 3) for h in H})
    print('  NSE S1b :', {h: round(r['pred']['S1b']['skill'][str(h)]['nse_knn'], 3) for h in H})
    print('  NSE S2  :', {h: round(r['pred']['S2']['skill'][str(h)]['nse_knn'], 3) for h in H})
    print('  det data/S1/S1b/S2:',
          {tag: {h: round(v, 3) for h, v in r['pred'][tag]['det_score'].items()} for tag in ('data', 'S1', 'S1b', 'S2')})

    ax = axes[list(LOOPY).index(name)]
    ax.plot(eps / scale, C_data, 'k-', lw=1.8, label='data')
    ax.plot(eps / scale, C_s1, color='tab:red', lw=1.1, label='S1: cycle+drift+iid resid')
    ax.plot(eps / scale, C_s1b, color='tab:orange', lw=1.1, ls='-.', label='S1b: cycle+AR resid')
    ax.plot(eps / scale, C_s2, 'b--', lw=1.1, label='S2: phase-shuffled')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$/scale'); ax.set_ylabel(r'$C(\epsilon)$')
    ax.set_title(f'{name} (P={P})'); ax.legend(fontsize=7); ax.grid(alpha=0.25)

    axp = axesP[list(LOOPY).index(name)]
    hs = H
    for tag, col in [('data', 'k'), ('S1', 'tab:red'), ('S1b', 'tab:orange'), ('S2', 'b')]:
        nse = [r['pred'][tag]['skill'][str(h)]['nse_knn'] for h in hs]
        axp.plot(hs, nse, 'o-', color=col, label=tag)
    axp.set_xscale('log'); axp.axhline(0, color='gray', lw=0.8)
    axp.set_ylim(-1.2, 1.05)
    axp.set_xlabel('horizon h (tokens)'); axp.set_ylabel('kNN prediction NSE')
    axp.set_title(f'{name}'); axp.legend(fontsize=8); axp.grid(alpha=0.25)

fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_synthloop_C.png'), dpi=130); plt.close(fig)
figP.tight_layout(); figP.savefig(os.path.join(OUT, 'fig2_synthloop_pred.png'), dpi=130); plt.close(figP)
with open(os.path.join(OUT, 'exp2_synthloop.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved exp2_synthloop.json')
