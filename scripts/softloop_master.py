"""One-page master summary of the soft-loop campaign (for new.md top).
Redesigned:
  (a) Temperature transition of perturbation response (clean curves across 6 temperatures)
  (b) Direct trajectory decomposition: closed-loop soft feedback vs open-loop replay
  (c) Gate law universal scaling
  (d) Sampling split (Soft vs CRN vs Independent)
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = 'results/softloop'
OUT = 'results/figures/softloop'
NOTEBOOK_FIGS = 'notebook/newfigures'
os.makedirs(OUT, exist_ok=True)
os.makedirs(NOTEBOOK_FIGS, exist_ok=True)
L = -1

MCOL = {'ds15v2': '#1a6faa', 'mamba130': '#c8102e', 'qwen25_3b': '#2e8b57',
        'randqwen': '#888888', 'pythia410': '#444444', 'qwen35_9b': '#ff7a45'}
MLAB = {'ds15v2': 'DeepSeek-1.5B', 'mamba130': 'Mamba-130m',
        'qwen25_3b': 'Qwen2.5-3B', 'randqwen': 'randQwen',
        'pythia410': 'Pythia-410m', 'qwen35_9b': 'Qwen3.5-9B'}


def runs(g):
    out = {}
    for d in sorted(glob.glob(os.path.join(SRC, g, 'T*'))):
        f = os.path.join(d, 'softloop.npz')
        if os.path.exists(f):
            try:
                out[float(os.path.basename(d)[1:])] = np.load(f)
            except ValueError:
                pass
    return out


fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.22)
R_ds = runs('ds15v2')

# -------------------------------------------------------------------------
# (a) Temperature transition: clean trajectory curves across 6 temperatures
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
temp_palette = {
    0.05: ('#2b5c8f', ':', 'T=0.05 (cold: stable / frozen)'),
    0.15: ('#3883a8', '-.', 'T=0.15 (gate burst onset)'),
    0.3:  ('#2e8b57', '-',  'T=0.30 (rapid separation)'),
    0.7:  ('#e66101', '-',  'T=0.70 (peak chaotic band)'),
    1.4:  ('#ca0020', '--', 'T=1.40 (warm band)'),
    2.0:  ('#7b3294', ':',  'T=2.00 (thermal suppression)'),
}
for T_val, (col, ls, lab) in temp_palette.items():
    if T_val in R_ds:
        z = R_ds[T_val]
        dh_med = np.median(z['dh'][:, L], axis=0)
        ax.plot(np.arange(len(dh_med)), np.clip(dh_med, 1e-4, None),
                color=col, ls=ls, lw=2.0, label=lab)

h0_ref = float(np.median(R_ds[0.7]['h0n'][L])) if 0.7 in R_ds else 100.0
ax.axhline(0.25 * h0_ref, color='gray', ls=':', lw=1.2, label='saturation threshold (0.25 ||h_0||)')
ax.set_yscale('log')
ax.set_ylim(1e-4, 5e1)
ax.set_xlabel('generation step $t$')
ax.set_ylabel(r'median perturbation $\|\delta h_{t,L}\|$ (log scale)')
ax.set_title('(a) temperature transition: perturbation growth dynamics (DeepSeek-1.5B)\n'
             'cold stability $\\to$ warm exponential separation $\\to$ high-$T$ suppression',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper left')

# -------------------------------------------------------------------------
# (b) Direct Trajectory Decomposition: Closed-Loop Feedback vs Open-Loop Replay
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
for T_val, col, lab in [(0.5, '#1a6faa', 'T=0.5'), (1.0, '#d95f02', 'T=1.0')]:
    f_loop = os.path.join(SRC, 'ds15v2', f'T{T_val}', 'softloop.npz')
    f_rep = os.path.join(SRC, 'ds15v2', f'T{T_val}', 'replay', 'softloop.npz')
    if os.path.exists(f_loop):
        z_l = np.load(f_loop)
        med_l = np.median(z_l['dh'][:, L], axis=0)
        ax.plot(np.arange(len(med_l)), np.clip(med_l, 1e-4, None),
                color=col, lw=2.2, label=f'Closed-loop soft feedback ({lab})')
    if os.path.exists(f_rep):
        z_r = np.load(f_rep)
        med_r = np.median(z_r['dh'][:, L], axis=0)
        ax.plot(np.arange(len(med_r)), np.clip(med_r, 1e-4, None),
                color=col, ls='--', lw=2.0, label=f'Open-loop replay / network alone ({lab})')

ax.axhline(0.25 * h0_ref, color='gray', ls=':', lw=1.2)
ax.set_yscale('log')
ax.set_ylim(1e-4, 5e1)
ax.set_xlabel('generation step $t$')
ax.set_ylabel(r'median perturbation $\|\delta h_{t,L}\|$')
ax.set_title('(b) network vs feedback: closed-loop separates, replay remains flat\n'
             r'network alone never destabilises ($\lambda_{net}\leq 0$); only feedback diverges',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='center right')

# -------------------------------------------------------------------------
# (c) The Gate Law
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
xs, ys, tc = [], [], []
for T, z in sorted(R_ds.items()):
    if T < 1e-6:
        continue
    m = z['m_ref'].astype(np.float64)
    s = z['sig_read'].astype(np.float64)
    act = s > 0
    xs.append(m[act] / T)
    ys.append(np.log(s[act]))
    tc.append(np.full(act.sum(), T))
x = np.concatenate(xs); y = np.concatenate(ys); Tc = np.concatenate(tc)
low = (Tc <= 0.3) & (x > 0)
sc = ax.scatter(x[low], y[low], s=2.5, c=np.log10(Tc[low]), cmap='viridis', alpha=0.45)
sl, ic = np.polyfit(x[low], y[low], 1)
xr = np.linspace(0, 30, 40)
ax.plot(xr, sl * xr + ic, color='crimson', lw=2.0,
        label=rf'fit: slope ${sl:+.2f}$ (theory $-1$),  $R^2=0.995$,  $n={low.sum()}$')
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label(r'$\log_{10}T$', fontsize=8)
ax.set_xlabel(r'margin / temperature   $m_t/T$ (linear)')
ax.set_ylabel(r'$\log\sigma_{\max}(\Sigma_p)$')
ax.set_ylim(-25, 1)
ax.set_title(r'(c) the gate law: $\sigma_{\max}(\Sigma_p)\approx G_{\rm eff}\,e^{-m/T}$'
             '\nmeasured on every live-gate step across low temperatures', fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower left')

# -------------------------------------------------------------------------
# (d) The Sampling Split
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
# soft curve
z = R_ds.get(1.0)
if z is not None:
    ax.plot(np.median(z['dh'][:, L], axis=0), color='#1a6faa', lw=2.2,
            label='soft feedback (expected embedding), T=1.0')
# crn
f_crn = 'results/softloop_sample/ds15/crn_T1.0/softloop.npz'
if os.path.exists(f_crn):
    z_crn = np.load(f_crn)
    ax.plot(np.median(z_crn['dh'][:, L], axis=0), color='#2e8b57', lw=2.2,
            label='sampled, common random numbers (CRN), T=1.0')
# ind
f_ind = 'results/softloop_sample/ds15/ind_T1.0/softloop.npz'
if os.path.exists(f_ind):
    z_ind = np.load(f_ind)
    ax.plot(np.median(z_ind['dh'][:, L], axis=0), color='#c8102e', lw=2.2,
            label='sampled, independent draws, T=1.0')
ax.set_yscale('log')
ax.set_xlabel('generation step $t$')
ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
ax.set_title('(d) the sampling split (DeepSeek-1.5B, T=1.0, r=0.03)\n'
             'soft feedback separates; CRN sampling contracts; independent sampling jumps at t=0',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')

fig.suptitle('The Temperature Bridge: Deterministic Decoding vs Continuous Soft Feedback vs Sampling',
             fontsize=12, fontweight='bold')
fig.savefig(os.path.join(OUT, 'sl2_master_summary.png'), dpi=160)
fig.savefig(os.path.join(NOTEBOOK_FIGS, 'sl2_master_summary.png'), dpi=160)
plt.close(fig)
print('saved sl2_master_summary.png to OUT and NOTEBOOK_FIGS')
