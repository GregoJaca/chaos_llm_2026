import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
runs=[(1e-3,'state_w100'),(3e-3,'state_w100_r0.003'),(1e-2,'state_w100_r0.01'),(3e-2,'state_w100_r0.03')]
ds=[]; steps=None
for r,run in runs:
    d=np.load(os.path.join(ROOT,run,'state.npz')); ds.append(d['dh'][:,:16]); steps=d['steps']
fig,axs=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
for si,s in enumerate(steps):
    ax=axs[si]
    for di in [0,1,2]:
        y=[d[si,di,s] for d in ds]
        ax.loglog([x[0] for x in runs],y,marker='o',lw=1,label=f'dir{di}')
    med=[np.median(d[si,:,s]) for d in ds]
    ax.loglog([x[0] for x in runs],med,marker='s',lw=2,color='k',label='median')
    ax.set_xlabel('r'); ax.set_title(f's={s}, t=s'); ax.grid(alpha=.25)
axs[0].set_ylabel(r'$||\delta h_s||$'); axs[0].legend(fontsize=7)
fig.suptitle('Current-state response vs r at injection'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'state_r_scaling.png'),dpi=200); plt.close(fig)
# decay at each r for s=50
fig,ax=plt.subplots(figsize=(7,4.5))
for (r,run),d in zip(runs,ds):
    s=50; si=list(steps).index(s); med=np.median(d[si],axis=0)
    ax.plot(np.arange(s,300),med[s:],lw=1,label=f'r={r:g}')
ax.set_yscale('log'); ax.set_xlabel('generated token'); ax.set_ylabel(r'median $||\delta h_t||$'); ax.set_title('Current-state decay vs r (s=50)')
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'state_r_decay.png'),dpi=200); plt.close(fig)
print('saved')
