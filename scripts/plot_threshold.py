import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
s=np.load(os.path.join(ROOT,'a1_ds_w100/a1.npz')); l=np.load(os.path.join(ROOT,'a1_ds_w100_large/a1.npz'))
radii=np.concatenate([s['radii'],l['radii']]); rc=np.concatenate([s['r_c'],l['r_c']],axis=0); pt=np.concatenate([s['pred_T'],l['pred_T']],axis=0)
K=rc.shape[1]; T=rc.shape[2]; _,idx=np.unique(np.round(radii,10),return_index=True); radii=radii[idx]; rc=rc[idx]; pt=pt[idx]
rc=np.clip(rc,1e-3,100.0)
fig,axs=plt.subplots(1,3,figsize=(16,4.6))
ri=int(np.argmin(np.abs(radii-0.1)))
for di in [0,1,2]:
    axs[0].plot(np.arange(T),rc[ri,di],lw=1,label=f'direction {di}')
for t in [16,123,143]: axs[0].axvline(t,color='k',ls=':',alpha=.4)
axs[0].set_yscale('log'); axs[0].set_xlim(0,160); axs[0].set_xlabel('generated token index'); axs[0].set_ylabel('local flip threshold r_c (clipped at 100)')
axs[0].set_title('r_c(t,u), r=0.1, first 160 steps'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
for t,c in [(16,'tab:blue'),(123,'tab:orange'),(143,'tab:green')]:
    axs[1].scatter(np.arange(K)+(t==123)*0.15-(t==143)*0.15,rc[ri,:,t],s=18,color=c,label=f't={t}')
axs[1].axhline(0.1,color='k',ls='--',alpha=.6,label='r=0.1'); axs[1].axhline(0.3,color='k',ls=':',alpha=.6,label='r=0.3')
axs[1].set_yscale('log'); axs[1].set_xlabel('direction index'); axs[1].set_ylabel('r_c at step (clipped)')
axs[1].set_title('Threshold across directions'); axs[1].legend(fontsize=8); axs[1].grid(alpha=.25)
# validation counts
obs={}
for f in ['free_run/summary.json','free_run_large/summary.json']:
    j=json.load(open(os.path.join(ROOT,'a1_ds_w100',f)))
    for r,o in zip(j['radii'],j['obs']): obs[round(float(r),6)]=np.asarray(o)
cats=['none','t=16','t=123','other']
def counts(arr):
    c=np.array([np.sum(arr<0),np.sum(arr==16),np.sum(arr==123),np.sum((arr>=0)&(arr!=16)&(arr!=123))])
    return c
x=np.arange(len(cats)); w=0.2
for k,(r,col) in enumerate([(0.1,'tab:blue'),(0.3,'tab:red')]):
    ri=int(np.argmin(np.abs(radii-r))); o=obs.get(round(r,6));
    if o is None: continue
    axs[2].bar(x+(k-0.5)*w,counts(pt[ri]),width=w,color=col,alpha=.45,label=f'pred r={r:g}')
    axs[2].bar(x+(k-0.5)*w,counts(o),width=w,color=col,alpha=1.0,label=f'obs r={r:g}',fill=False,edgecolor=col,lw=2)
axs[2].set_xticks(x); axs[2].set_xticklabels(cats); axs[2].set_ylabel('number of directions'); axs[2].set_title('Prediction vs free run')
axs[2].legend(fontsize=7); axs[2].grid(alpha=.25,axis='y')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'threshold_validation.png'),dpi=200); plt.close(fig)
print('saved')
