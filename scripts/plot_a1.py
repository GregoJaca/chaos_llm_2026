import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/LLM/chaos_llm_2026/results')
OUT = os.path.join(ROOT, 'figures'); os.makedirs(OUT, exist_ok=True)
small = np.load(os.path.join(ROOT, 'a1_ds_w100/a1.npz'))
large = np.load(os.path.join(ROOT, 'a1_ds_w100_large/a1.npz'))

def merge(key):
    a, b = small[key], large[key]
    return np.concatenate([a, b], axis=0)

radii = np.concatenate([small['radii'], large['radii']])
dh = merge('dh_norm'); dm = merge('dm'); rc = merge('r_c'); pt = merge('pred_T')
m = small['margins']; layers = small['layers']; K = dh.shape[1]; T = dh.shape[-1]
# dedupe radii 1e-2 (appears in both)
_, idx = np.unique(np.round(radii, 10), return_index=True)
radii = radii[idx]; dh = dh[idx]; dm = dm[idx]; rc = rc[idx]; pt = pt[idx]
print('radii', radii)

# 1. individual response, last layer
sel_r = [1e-3, 1e-2, 0.1, 0.3]
sel_r = [r for r in sel_r if np.any(np.isclose(radii, r))]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, di in zip(axs, [0, 1, 2]):
    for r in sel_r:
        i = int(np.argmin(np.abs(radii - r)))
        ax.plot(np.arange(T), np.maximum(dh[i, di, -1], 1e-12), lw=1, label=f'r={r:g}')
    for t in [16, 123, 143]:
        ax.axvline(t, color='k', ls=':', alpha=.4)
    ax.set_xlabel('generated token index'); ax.set_yscale('log'); ax.grid(alpha=.25)
    ax.set_title(f'direction {di}')
axs[0].set_ylabel(r'$||\delta h_t||$ at last layer')
axs[0].legend(fontsize=8)
fig.suptitle('A1 prompt response: 3 individual directions'); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'a1_response_individual.png'), dpi=200); plt.close(fig)

# 2. aggregate median/IQR
fig, ax = plt.subplots(figsize=(8, 5))
for r in sel_r:
    i = int(np.argmin(np.abs(radii - r)))
    x = dh[i, :, -1, :]  # [K,T]
    med = np.median(x, axis=0); q1 = np.quantile(x, .25, axis=0); q3 = np.quantile(x, .75, axis=0)
    line, = ax.plot(np.arange(T), med, lw=1.4, label=f'r={r:g}')
    ax.fill_between(np.arange(T), q1, q3, color=line.get_color(), alpha=.18)
for t in [16, 123, 143]:
    ax.axvline(t, color='k', ls=':', alpha=.4)
ax.set_yscale('log'); ax.set_xlabel('generated token index'); ax.set_ylabel(r'$||\delta h_t||$ at last layer')
ax.set_title('A1 prompt response: median and IQR over 20 directions'); ax.legend(fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'a1_response_aggregate.png'), dpi=200); plt.close(fig)

# 3. margins
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.arange(T), np.maximum(m, 1e-6), lw=.9)
for t in [16, 123, 143]:
    ax.axvline(t, color='r', ls='--', alpha=.5)
    ax.text(t, max(m[t], 1e-5), f' {t}', color='r', fontsize=8)
ax.set_yscale('log'); ax.set_xlabel('generated token index'); ax.set_ylabel('top1 - top2 margin')
ax.set_title('Baseline logit margins (window 100, FP32)'); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'margins.png'), dpi=200); plt.close(fig)

# 4. response vs r at selected steps
fig, ax = plt.subplots(figsize=(7, 5))
for t in [16, 50, 100, 123, 200]:
    y = np.median(dh[:, :, -1, t], axis=1)
    ax.plot(radii, y, marker='o', ms=3, lw=1, label=f't={t}')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('r'); ax.set_ylabel(r'median $||\delta h_t||$')
ax.set_title('Response vs r (last layer)'); ax.legend(fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'response_vs_r.png'), dpi=200); plt.close(fig)

# 5. r_c heatmap at r=0.1 (or nearest)
r0 = 0.1
i0 = int(np.argmin(np.abs(radii - r0)))
rc0 = rc[i0]  # [K,T]
order = np.argsort(rc0.min(axis=1))
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(np.log10(np.maximum(rc0[order], 1e-6)), aspect='auto', origin='lower',
               cmap='viridis', vmin=-1, vmax=1.5)
ax.set_xlabel('generated token index'); ax.set_ylabel('direction (sorted by min r_c)')
ax.set_title(f'log10 local flip threshold r_c(t,u), r={radii[i0]:g}')
fig.colorbar(im, ax=ax, label='log10 r_c')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'rc_heatmap.png'), dpi=200); plt.close(fig)

# 6. r_star distribution
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for r in [0.01, 0.1, 0.3]:
    i = int(np.argmin(np.abs(radii - r)))
    rstar = rc[i].min(axis=1)
    axs[0].hist(np.log10(rstar), bins=15, alpha=.5, label=f'r={r:g}')
    x = np.sort(rstar); axs[1].plot(x, np.arange(1, len(x)+1)/len(x), marker='.', label=f'r={r:g}')
axs[0].set_xlabel('log10 r_* (min over t)'); axs[0].set_ylabel('count'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
axs[1].set_xscale('log'); axs[1].set_xlabel('r_*'); axs[1].set_ylabel('ECDF'); axs[1].legend(fontsize=8); axs[1].grid(alpha=.25)
fig.suptitle('Flux-tube radius distribution over directions'); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'rstar_distribution.png'), dpi=200); plt.close(fig)

# 7. predicted vs observed T at r=0.03/0.1/0.3
fig, ax = plt.subplots(figsize=(6, 4))
for r in [0.03, 0.1, 0.3]:
    i = int(np.argmin(np.abs(radii - r)))
    ax.scatter(np.full(K, r), pt[i], s=18, label=f'r={r:g}')
ax.set_xscale('log'); ax.set_xlabel('r'); ax.set_ylabel('predicted first flip T (-1 = none)')
ax.set_title('Predicted T(r,u) from teacher forcing'); ax.legend(fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'predicted_T.png'), dpi=200); plt.close(fig)

with open(os.path.join(OUT, 'INDEX.md'), 'w') as f:
    f.write('# Figure index\n\n')
    f.write('- `a1_response_individual.png`: response of 3 directions, last layer, several r.\n')
    f.write('- `a1_response_aggregate.png`: median/IQR of the same response over 20 directions.\n')
    f.write('- `margins.png`: baseline top1-top2 margin vs generated step; near-ties marked.\n')
    f.write('- `response_vs_r.png`: median response vs r at selected steps; shows floor and scaling.\n')
    f.write('- `rc_heatmap.png`: local flip threshold r_c(t,u) over steps and directions.\n')
    f.write('- `rstar_distribution.png`: distribution of the flux-tube radius r_* over directions.\n')
    f.write('- `predicted_T.png`: predicted first-flip time from teacher forcing.\n')
print('saved figures to', OUT)
