"""Generate comprehensive figures showing trajectory response curves like Figure 0.1 (d)
across models, feedback modes, radii, and temperatures.

Produces:
1. sl2_sampling_cross_models.png: Soft vs CRN vs Independent across models & parameter sweeps
2. sl2_response_all_architectures.png: Response curves across all 8 model families & temperatures
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_LOOP = 'results/softloop'
SRC_SAMPLE = 'results/softloop_sample'
SRC_BIG = 'results_big/softloop'
OUT = 'results/figures/softloop'
NOTEBOOK_FIGS = 'notebook/newfigures'
os.makedirs(OUT, exist_ok=True)
os.makedirs(NOTEBOOK_FIGS, exist_ok=True)
L = -1

# -----------------------------------------------------------------------------
# Figure 1: sl2_sampling_cross_models.png
# -----------------------------------------------------------------------------
def make_sampling_cross_models():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # (1) DeepSeek-1.5B, T=1.0
    ax = axes[0]
    f_soft = os.path.join(SRC_LOOP, 'ds15v2', 'T1', 'softloop.npz')
    f_crn = os.path.join(SRC_SAMPLE, 'ds15', 'crn_T1.0', 'softloop.npz')
    f_ind = os.path.join(SRC_SAMPLE, 'ds15', 'ind_T1.0', 'softloop.npz')
    if os.path.exists(f_soft):
        ax.plot(np.median(np.load(f_soft)['dh'][:, L], axis=0), color='#1a6faa', lw=2.2, label='Soft feedback (T=1.0)')
    if os.path.exists(f_crn):
        ax.plot(np.median(np.load(f_crn)['dh'][:, L], axis=0), color='#2e8b57', lw=2.2, label='CRN sampling (shared noise)')
    if os.path.exists(f_ind):
        ax.plot(np.median(np.load(f_ind)['dh'][:, L], axis=0), color='#c8102e', lw=2.2, label='Independent sampling')
    ax.set_yscale('log')
    ax.set_xlabel('generation step $t$')
    ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
    ax.set_title('(a) DeepSeek-1.5B (T=1.0, r=0.03)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (2) DeepSeek-1.5B, T=0.5
    ax = axes[1]
    f_soft = os.path.join(SRC_LOOP, 'ds15v2', 'T0.5', 'softloop.npz')
    f_crn = os.path.join(SRC_SAMPLE, 'ds15', 'crn_T0.5', 'softloop.npz')
    f_ind = os.path.join(SRC_SAMPLE, 'ds15', 'ind_T0.5', 'softloop.npz')
    if os.path.exists(f_soft):
        ax.plot(np.median(np.load(f_soft)['dh'][:, L], axis=0), color='#1a6faa', lw=2.2, label='Soft feedback (T=0.5)')
    if os.path.exists(f_crn):
        ax.plot(np.median(np.load(f_crn)['dh'][:, L], axis=0), color='#2e8b57', lw=2.2, label='CRN sampling (shared noise)')
    if os.path.exists(f_ind):
        ax.plot(np.median(np.load(f_ind)['dh'][:, L], axis=0), color='#c8102e', lw=2.2, label='Independent sampling')
    ax.set_yscale('log')
    ax.set_xlabel('generation step $t$')
    ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
    ax.set_title('(b) DeepSeek-1.5B (T=0.5, r=0.03)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (3) Qwen2.5-3B, T=1.0
    ax = axes[2]
    f_soft = os.path.join(SRC_LOOP, 'qwen25_3b', 'T1', 'softloop.npz')
    f_crn = os.path.join(SRC_SAMPLE, 'qwen25_3b', 'crn_T1.0', 'softloop.npz')
    f_ind = os.path.join(SRC_SAMPLE, 'qwen25_3b', 'ind_T1.0', 'softloop.npz')
    if os.path.exists(f_soft):
        ax.plot(np.median(np.load(f_soft)['dh'][:, L], axis=0), color='#1a6faa', lw=2.2, label='Soft feedback (T=1.0)')
    if os.path.exists(f_crn):
        ax.plot(np.median(np.load(f_crn)['dh'][:, L], axis=0), color='#2e8b57', lw=2.2, label='CRN sampling')
    if os.path.exists(f_ind):
        ax.plot(np.median(np.load(f_ind)['dh'][:, L], axis=0), color='#c8102e', lw=2.2, label='Independent sampling')
    ax.set_yscale('log')
    ax.set_xlabel('generation step $t$')
    ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
    ax.set_title('(c) Qwen2.5-3B (T=1.0, r=0.3)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (4) Mamba-130m, T=1.0
    ax = axes[3]
    f_soft = os.path.join(SRC_LOOP, 'mamba130', 'T1', 'softloop.npz')
    f_crn = os.path.join(SRC_SAMPLE, 'mamba130', 'crn_T1.0', 'softloop.npz')
    f_ind = os.path.join(SRC_SAMPLE, 'mamba130', 'ind_T1.0', 'softloop.npz')
    if os.path.exists(f_soft):
        ax.plot(np.median(np.load(f_soft)['dh'][:, L], axis=0), color='#1a6faa', lw=2.2, label='Soft feedback (T=1.0)')
    if os.path.exists(f_crn):
        ax.plot(np.median(np.load(f_crn)['dh'][:, L], axis=0), color='#2e8b57', lw=2.2, label='CRN sampling')
    if os.path.exists(f_ind):
        ax.plot(np.median(np.load(f_ind)['dh'][:, L], axis=0), color='#c8102e', lw=2.2, label='Independent sampling')
    ax.set_yscale('log')
    ax.set_xlabel('generation step $t$')
    ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
    ax.set_title('(d) Mamba-130m (SSM, T=1.0, r=0.1)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (5) CRN Radius Sweep (DeepSeek-1.5B, T=1.0)
    ax = axes[4]
    radii = [0.003, 0.01, 0.1, 0.3]
    cmap = plt.get_cmap('viridis')
    for idx, r in enumerate(radii):
        f = os.path.join(SRC_SAMPLE, 'ds15', f'crn_T1.0_r{r}', 'softloop.npz')
        if os.path.exists(f):
            dh = np.median(np.load(f)['dh'][:, L], axis=0)
            ax.plot(np.arange(len(dh)), np.clip(dh, 1e-5, None), lw=2.0,
                    color=cmap(idx / len(radii)), label=f'r = {r}')
    ax.set_yscale('log')
    ax.set_xlabel('generation step $t$')
    ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
    ax.set_title('(e) CRN Perturbation Radius Sweep (T=1.0)\ncontracts across 2 orders of magnitude of radius', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # (6) CRN Temperature Sweep (DeepSeek-1.5B)
    ax = axes[5]
    for T, col in [(0.3, '#2b5c8f'), (0.5, '#2e8b57'), (1.0, '#e66101')]:
        f = os.path.join(SRC_SAMPLE, 'ds15', f'crn_T{T}', 'softloop.npz')
        if os.path.exists(f):
            dh = np.median(np.load(f)['dh'][:, L], axis=0)
            ax.plot(np.arange(len(dh)), np.clip(dh, 1e-5, None), lw=2.0,
                    color=col, label=f'CRN T={T}')
    ax.set_yscale('log')
    ax.set_xlabel('generation step $t$')
    ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
    ax.set_title('(f) CRN Temperature Sweep (r=0.03)\nstable and contractive at all temperatures', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Decoding Rule Split Across Architectures and Parameters: Soft vs Common Random Numbers (CRN) vs Independent',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sl2_sampling_cross_models.png'), dpi=160)
    fig.savefig(os.path.join(NOTEBOOK_FIGS, 'sl2_sampling_cross_models.png'), dpi=160)
    plt.close(fig)
    print('saved sl2_sampling_cross_models.png')


# -----------------------------------------------------------------------------
# Figure 2: sl2_response_all_architectures.png
# -----------------------------------------------------------------------------
def make_response_all_architectures():
    models = [
        ('ds15v2', 'DeepSeek-1.5B (fp32)', SRC_LOOP),
        ('qwen25_3b', 'Qwen2.5-3B', SRC_LOOP),
        ('qwen3_4b', 'Qwen3-4B', SRC_LOOP),
        ('phi4mini', 'Phi-4-mini', SRC_LOOP),
        ('mamba130', 'Mamba-130m', SRC_LOOP),
        ('pythia410', 'Pythia-410m', SRC_LOOP),
        ('randqwen', 'randQwen (random control)', SRC_LOOP),
        ('qwen35_9b', 'Qwen3.5-9B (hybrid)', SRC_BIG if os.path.exists(SRC_BIG) else SRC_LOOP),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    temps_to_plot = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.4, 2.0]
    cmap = plt.get_cmap('coolwarm')

    for i, (mkey, mname, src_dir) in enumerate(models):
        ax = axes[i]
        path_m = os.path.join(src_dir, mkey)
        if not os.path.isdir(path_m) and src_dir == SRC_BIG:
            path_m = os.path.join('results/softloop', mkey)
        
        runs_found = 0
        for idx, T in enumerate(temps_to_plot):
            d = os.path.join(path_m, f'T{T:g}')
            f = os.path.join(d, 'softloop.npz')
            if not os.path.exists(f):
                continue
            try:
                z = np.load(f)
                dh = z['dh'][:, L]
                med = np.median(dh, axis=0)
                col = cmap(idx / (len(temps_to_plot) - 1))
                ax.plot(np.arange(len(med)), np.clip(med, 1e-4, None),
                        color=col, lw=1.8, label=f'T={T:g}')
                runs_found += 1
            except Exception:
                pass
        
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1e2)
        ax.set_xlabel('step $t$')
        ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
        ax.set_title(f'({chr(97+i)}) {mname}', fontsize=10, fontweight='bold')
        if runs_found > 0:
            ax.legend(fontsize=6.5, loc='lower right', ncol=2)
        else:
            ax.text(0.5, 0.5, 'Run queued on GPU', ha='center', va='center', transform=ax.transAxes, color='gray')

    fig.suptitle(r'Closed-Loop Soft Perturbation Trajectories $\|\delta h_{t,L}\|$ Across Architectures and Temperatures',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sl2_response_all_architectures.png'), dpi=160)
    fig.savefig(os.path.join(NOTEBOOK_FIGS, 'sl2_response_all_architectures.png'), dpi=160)
    plt.close(fig)
    print('saved sl2_response_all_architectures.png')


if __name__ == '__main__':
    make_sampling_cross_models()
    make_response_all_architectures()
