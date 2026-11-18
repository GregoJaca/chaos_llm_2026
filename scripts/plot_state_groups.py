import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
d=np.load(os.path.join(ROOT,'state_w100_32dirs/state.npz')); dh=d['dh']; steps=d['steps']; T=dh.shape[-1]; K=dh.shape[1]
# pairwise correlations per s
stats={}
for si,s in enumerate(steps):
    X=np.log(np.maximum(dh[si,:,s:],1e-12))
    C=np.corrcoef(X)
    iu=np.triu_indices(K,1); vals=C[iu]
    stats[int(s)]=dict(median=float(np.nanmedian(vals)),p10=float(np.nanquantile(vals,.1)),p90=float(np.nanquantile(vals,.9)),frac_gt_0_9=float(np.mean(vals>0.9)))
    if si==1:
        order=leaves_list(linkage(squareform(1-C,checks=False),method='average'))
        fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(C[np.ix_(order,order)],cmap='viridis',vmin=0,vmax=1)
        ax.set_title(f'Direction correlation of log response (s={s})'); ax.set_xlabel('direction (clustered)'); ax.set_ylabel('direction (clustered)')
        fig.colorbar(im,ax=ax,label='Pearson r'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'state_direction_corr.png'),dpi=200); plt.close(fig)
print('stats',stats)
# 3 individual + median/IQR per s
fig,axs=plt.subplots(1,len(steps),figsize=(4*len(steps),4),sharey=True)
for ax,s in zip(axs,steps):
    si=list(steps).index(s); x=dh[si,:,s:]
    for di in [0,1,2]: ax.plot(np.arange(s,T),np.maximum(x[di],1e-12),lw=1,alpha=.8,label=f'd{di}')
    med=np.median(x,axis=0); q1=np.quantile(x,.25,axis=0); q3=np.quantile(x,.75,axis=0)
    ax.plot(np.arange(s,T),med,lw=2,color='k',label='median'); ax.fill_between(np.arange(s,T),q1,q3,color='k',alpha=.15)
    ax.set_yscale('log'); ax.set_xlabel('generated token'); ax.set_title(f's={s}'); ax.grid(alpha=.25)
axs[0].set_ylabel(r'$||\delta h_t||$'); axs[0].legend(fontsize=6)
fig.suptitle('Current-state response: 32 directions'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'state_directions.png'),dpi=200); plt.close(fig)
# shifted R(t,s) vs lag
fig,ax=plt.subplots(figsize=(8,4.5))
for s in steps:
    si=list(steps).index(s); x=dh[si,:,s:]; base=np.maximum(dh[si,:,s:s+1],1e-30)
    R=x/base; med=np.median(R,axis=0)
    ax.plot(np.arange(len(med)),med,lw=1.5,label=f's={s}')
ax.set_yscale('log'); ax.set_xlabel(r'lag $\tau=t-s$'); ax.set_ylabel('median R(t,s)'); ax.set_title('Two-time response vs lag (median over 32 dirs)')
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'state_shift.png'),dpi=200); plt.close(fig)
json.dump(stats,open(os.path.join(ROOT,'state_direction_stats.json'),'w'),indent=2)
print('saved')
