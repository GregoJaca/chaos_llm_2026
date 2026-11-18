import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
j=json.load(open(os.path.join(ROOT,'fsle_depth.json'))); curves={float(k):np.asarray(v) for k,v in j['curves'].items()}
# separate panels: normalized at layer 1 to remove the initial jump
fig,axs=plt.subplots(1,len(curves),figsize=(4*len(curves),4),sharey=False)
for ax,(r,d) in zip(axs,sorted(curves.items())):
    base=d[1] if len(d)>1 else d[0]
    ax.plot(np.arange(len(d)),d/d[0],marker='.',ms=3,lw=1,color='tab:blue',label='d(l)/d(0)')
    ax.plot(np.arange(len(d)),d/base,marker='.',ms=3,lw=1,color='tab:orange',label='d(l)/d(1)')
    ax.set_yscale('log'); ax.set_title(f'r={r:g}'); ax.set_xlabel('layer'); ax.grid(alpha=.25)
    if ax is axs[0]: ax.set_ylabel('depth separation ratio')
axs[0].legend(fontsize=8)
fig.suptitle('Depth separation: initial jump then flat/decay (no shared y)'); fig.tight_layout()
fig.savefig(os.path.join(OUT,'fsle_depth_detail.png'),dpi=200); plt.close(fig)
# local gain d(l)/d(l-1)
fig,ax=plt.subplots(figsize=(8,4.5))
for r,d in sorted(curves.items()):
    g=np.diff(d)/np.maximum(d[:-1],1e-30)
    ax.plot(np.arange(1,len(d)),g,marker='.',ms=3,lw=1,label=f'r={r:g}')
ax.set_yscale('log'); ax.set_xlabel('layer'); ax.set_ylabel('d(l)/d(l-1)'); ax.set_title('Local depth gain')
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'depth_local_gain_existing.png'),dpi=200); plt.close(fig)
print('saved')
