import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
cfg=[(50,'a1_ds_w50_large'),(100,'a1_ds_w100_layers'),(200,'a1_ds_w200_large'),(500,'a1_ds_w500_large')]
fig,axs=plt.subplots(2,2,figsize=(12,8))
for (w,run),ax in zip(cfg,axs.flat):
    d=np.load(os.path.join(ROOT,run,'a1.npz')); radii=d['radii']; dh=d['dh_norm']; L=d['layers']
    li=list(L).index(27) if 27 in L else -1; ri=int(np.argmin(np.abs(radii-0.1)))
    x=dh[ri,:,li,:]
    for di in [0,1,2]: ax.plot(np.arange(300),np.maximum(x[di],1e-12),lw=1,alpha=.7,label=f'dir{di}')
    med=np.median(x,axis=0); ax.plot(np.arange(300),np.maximum(med,1e-12),lw=2,color='k',label='median')
    tclip=max(0,w-65)
    ax.axvline(tclip,color='r',ls='--',alpha=.7); ax.text(tclip+2,np.nanmax(med)*.7,f'context exceeds\nwindow at t={tclip}',color='r',fontsize=8)
    ax.set_yscale('log'); ax.set_title(f'window={w}, r=0.1, last layer'); ax.set_xlabel('generated token')
    ax.set_ylabel(r'$||\delta h_t||$'); ax.grid(alpha=.25); ax.legend(fontsize=7)
    lo = max(np.nanmin(x[x > 0]), 1e-3) * 0.6
    ax.set_ylim(lo, np.nanmax(x) * 1.6)
fig.suptitle('A1: the rise/decay regimes move with the sliding window (separate y axes)')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_window_regimes.png'),dpi=200); plt.close(fig)
# overlay medians, normalized to t=0, to show shape but avoid shared absolute y
fig,ax=plt.subplots(figsize=(8,4.5))
for w,run in cfg:
    d=np.load(os.path.join(ROOT,run,'a1.npz')); radii=d['radii']; dh=d['dh_norm']; L=d['layers']; li=list(L).index(27) if 27 in L else -1
    ri=int(np.argmin(np.abs(radii-0.1))); med=np.median(dh[ri,:,li,:],axis=0); med=np.maximum(med,1e-12)
    ax.plot(np.arange(300),med/med[0],lw=1.2,label=f'w={w}')
ax.set_yscale('log'); ax.set_xlabel('generated token'); ax.set_ylabel('response normalized to t=0'); ax.set_title('Window changes the shape and clip time')
ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_window_overlay.png'),dpi=200)
print('saved')
