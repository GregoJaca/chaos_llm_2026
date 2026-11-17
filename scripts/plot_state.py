import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
d=np.load(os.path.join(ROOT,'state_w100/state.npz')); dh=d['dh']; steps=d['steps']; T=dh.shape[-1]
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for si,s in enumerate(steps):
    for di in [0,1,2]:
        axs[0].plot(np.arange(s,T),np.maximum(dh[si,di,s:],1e-12),lw=1,alpha=.8,label=f's={s} d={di}')
    med=np.median(dh[si,:,s:],axis=0); axs[1].plot(np.arange(s,T),med/np.maximum(dh[si,:,s][:,None],1e-30).mean(),lw=2,label=f's={s} median R')
axs[0].set_yscale('log'); axs[0].set_xlabel('generated token index'); axs[0].set_ylabel(r'$||\delta h_t||$'); axs[0].set_title('Current-state perturbation response')
axs[0].legend(fontsize=6,ncol=2); axs[0].grid(alpha=.25)
axs[1].set_yscale('log'); axs[1].set_xlabel('generated token index'); axs[1].set_ylabel('R(t,s) (median)'); axs[1].set_title('Two-time response')
axs[1].legend(fontsize=8); axs[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'state_response.png'),dpi=200)
print('saved')
