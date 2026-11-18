import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
mid=np.load(os.path.join(ROOT,'state_mid_w100/state_mid.npz')); inp=np.load(os.path.join(ROOT,'state_w100/state.npz'))
dh_m=mid['dh']; dh_i=inp['dh']; steps=mid['steps']; T=dh_m.shape[-1]
fig,axs=plt.subplots(1,len(steps),figsize=(5*len(steps),4),sharey=True)
for ax,s in zip(axs,steps):
    si=list(steps).index(s)
    for arr,label,col in [(dh_i[si],'input embedding','tab:blue'),(dh_m[si],'layer-14 residual','tab:orange')]:
        x=arr[:,s:]; med=np.median(x,axis=0); q1=np.quantile(x,.25,axis=0); q3=np.quantile(x,.75,axis=0)
        ax.plot(np.arange(s,T),med,lw=1.5,label=label,color=col); ax.fill_between(np.arange(s,T),q1,q3,color=col,alpha=.2)
    ax.set_yscale('log'); ax.set_title(f's={s}'); ax.set_xlabel('generated token'); ax.grid(alpha=.25)
axs[0].set_ylabel(r'$||\delta h_t||$'); axs[0].legend(fontsize=8)
fig.suptitle('Input-embedding vs middle-layer residual injection (r=1e-3)'); fig.tight_layout()
fig.savefig(os.path.join(OUT,'state_mid_compare.png'),dpi=200)
print('saved')
