import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/LLM/chaos_llm_2026/results')
OUT = os.path.join(ROOT, 'figures'); os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(ROOT, 'depth_w100/depth.npz'))
lam = d['lam']; local = d['local_gain']; hd = d['h_depth']; tokens = d['tokens']; layers = d['layers']
nt, nL, k = lam.shape
sel = [0, 2, 4]  # token indices for individual plots

# 1. top exponent vs layer, individual
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, ti in zip(axs, sel):
    for i in range(min(3, k)):
        ax.plot(layers, lam[ti, :, i], marker='.', ms=3, lw=1, label=f'λ{i+1}')
    ax.set_title(f'token t={tokens[ti]}'); ax.set_xlabel('layer'); ax.grid(alpha=.25)
axs[0].set_ylabel('finite-time depth exponent'); axs[0].legend(fontsize=8)
fig.suptitle('Depth map: top exponents vs layer (individual tokens)'); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'depth_individual.png'), dpi=200); plt.close(fig)

# 2. aggregate top exponent
fig, ax = plt.subplots(figsize=(8, 4.5))
x = lam[:, :, 0]
med = np.median(x, axis=0); q1 = np.quantile(x, .25, axis=0); q3 = np.quantile(x, .75, axis=0)
ax.plot(layers, med, lw=1.5, label='median')
ax.fill_between(layers, q1, q3, alpha=.2, label='IQR')
ax.set_xlabel('layer'); ax.set_ylabel('top depth exponent λ1'); ax.set_title('Depth map: aggregate top exponent')
ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'depth_aggregate.png'), dpi=200); plt.close(fig)

# 3. spectrum at final layer, several tokens
fig, ax = plt.subplots(figsize=(7, 4.5))
for ti in range(nt):
    ax.plot(np.arange(1, k+1), lam[ti, -1], marker='o', ms=3, lw=1, label=f't={tokens[ti]}')
ax.set_xlabel('exponent index'); ax.set_ylabel('λ_i at final layer'); ax.set_title('Depth spectrum (top 8)')
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'depth_spectrum.png'), dpi=200); plt.close(fig)

# 4. local gain vs layer
fig, ax = plt.subplots(figsize=(8, 4.5))
for ti in sel:
    ax.plot(layers, local[ti], marker='.', ms=3, lw=1, label=f't={tokens[ti]}')
ax.set_xlabel('layer'); ax.set_ylabel(r'mean $||J_\ell q||$'); ax.set_title('Local layer gain')
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'depth_local_gain.png'), dpi=200); plt.close(fig)

# 5. h_depth distribution
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(np.arange(nt), hd, color='steelblue')
ax.set_xticks(np.arange(nt)); ax.set_xticklabels([f'{int(t)}' for t in tokens])
ax.set_xlabel('token'); ax.set_ylabel(r'$h_{depth}=\sum_{\lambda_i>0}\lambda_i$')
ax.set_title('Expansion entropy of the depth map'); ax.grid(alpha=.25, axis='y')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'h_depth.png'), dpi=200); plt.close(fig)

print('saved depth figures')
