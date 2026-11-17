import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/LLM/chaos_llm_2026/results')
OUT = os.path.join(ROOT, 'figures'); os.makedirs(OUT, exist_ok=True)
s = np.load(os.path.join(ROOT, 'a1_ds_w100/a1.npz')); l = np.load(os.path.join(ROOT, 'a1_ds_w100_large/a1.npz'))
radii = np.concatenate([s['radii'], l['radii']])
d = np.concatenate([s['dh_norm'], l['dh_norm']], axis=0)[:, :, -1, :]  # [nr,K,T]
_, idx = np.unique(np.round(radii, 10), return_index=True)
radii = radii[idx]; d = d[idx]
K, T = d.shape[1], d.shape[2]
rhos = [1.5, 2.0, 4.0]
res = {}
for rho in rhos:
    lam = np.full((len(radii), K), np.nan); cens = np.zeros((len(radii), K), bool)
    for ri in range(len(radii)):
        for di in range(K):
            x = d[ri, di]; d0 = x[0]
            if d0 <= 0: cens[ri, di] = True; continue
            hit = np.where(x >= rho * d0)[0]
            if len(hit) == 0:
                cens[ri, di] = True; continue
            tau = max(int(hit[0]), 1)
            lam[ri, di] = np.log(rho) / tau
    res[rho] = dict(median=[float(np.nanmedian(lam[i])) if np.any(~cens[i]) else None for i in range(len(radii))],
                    frac_cens=[float(cens[i].mean()) for i in range(len(radii))])
    print('rho', rho, 'median lambda', res[rho]['median'], 'cens', res[rho]['frac_cens'])

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for rho in rhos:
    y = np.array([np.nan if v is None else v for v in res[rho]['median']], float)
    axs[0].plot(radii, y, marker='o', label=f'rho={rho}')
    axs[1].plot(radii, res[rho]['frac_cens'], marker='o', label=f'rho={rho}')
axs[0].set_xscale('log'); axs[0].set_xlabel('r'); axs[0].set_ylabel('median FSLE lambda')
axs[0].set_title('Time-axis FSLE'); axs[0].legend(); axs[0].grid(alpha=.25)
axs[1].set_xscale('log'); axs[1].set_xlabel('r'); axs[1].set_ylabel('fraction censored')
axs[1].set_title('No growth by rho within horizon'); axs[1].legend(); axs[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fsle_time.png'), dpi=200); plt.close(fig)

# 3 individual separation curves for selected r
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, di in zip(axs, [0, 1, 2]):
    for r in [1e-3, 1e-2, 0.1, 0.3]:
        ri = int(np.argmin(np.abs(radii - r)))
        ax.plot(np.arange(T), np.maximum(d[ri, di], 1e-12), lw=1, label=f'r={r:g}')
    ax.set_yscale('log'); ax.set_xlabel('generated token index'); ax.set_title(f'direction {di}'); ax.grid(alpha=.25)
axs[0].set_ylabel(r'$||\delta h_t||$'); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fsle_individual.png'), dpi=200); plt.close(fig)

json.dump(dict(radii=radii.tolist(), rhos=rhos, results={str(k): v for k, v in res.items()}),
          open(os.path.join(ROOT, 'fsle_time.json'), 'w'), indent=2)
print('saved fsle figures')
