import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
d=np.load(os.path.join(ROOT,'a1_ds_w100_layers/a1.npz')); radii=d['radii']; m=d['margins']; dm=d['dm']; rc=d['r_c'];
i01=int(np.argmin(np.abs(radii-0.01))); g=np.abs(dm[i01])/radii[i01]; rcg=m[None,:]/np.maximum(g,1e-30)
fig,axs=plt.subplots(1,3,figsize=(18,4.8))
# A: margin vs gain
tgrid=np.repeat(np.arange(len(m))[None,:],g.shape[0],axis=0)
sel=np.random.default_rng(0).choice(g.size,size=min(3000,g.size),replace=False)
axs[0].scatter(g.ravel()[sel],m[tgrid.ravel()[sel]],s=8,c=np.log10(np.clip(rcg.ravel()[sel],1e-3,10)),cmap='viridis',alpha=.7)
for r,c in [(0.1,'k'),(0.3,'r')]:
    gg=np.logspace(-3,2,100); axs[0].plot(gg,r*gg,c,ls='--',label=f'r_c={r}')
axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_xlabel(r'$|g_t|=|\delta m_t|/r$'); axs[0].set_ylabel('margin $m_t$')
axs[0].set_title('Decision-boundary geometry'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
# B: r_* vs controlling step
for run,label,col in [('a1_ds_w100_layers','w100','tab:blue'),('a1_ds_w200_large','w200','tab:orange')]:
    dd=np.load(os.path.join(ROOT,run,'a1.npz')); rr=dd['r_c'][int(np.argmin(np.abs(dd['radii']-0.1)))]
    rstar=rr.min(axis=1); step=rr.argmin(axis=1)
    axs[1].scatter(step,rstar,s=18,color=col,label=label,alpha=.8)
axs[1].axhline(0.1,color='k',ls='--',alpha=.5); axs[1].axhline(0.3,color='r',ls=':',alpha=.5)
axs[1].set_yscale('log'); axs[1].set_xlabel('controlling near-tie step (argmin r_c)'); axs[1].set_ylabel(r'$r_*=\min_t r_c$')
axs[1].set_title('Which near-tie sets the threshold'); axs[1].legend(fontsize=8); axs[1].grid(alpha=.25)
# C: confusion matrix predicted vs observed at selected r
classes=['none','16','44','55','123','other']
def cls(a):
    out=[]
    for v in np.asarray(a).ravel():
        if v<0: out.append('none')
        elif int(v) in (16,44,55,123): out.append(str(int(v)))
        else: out.append('other')
    return np.array(out)
M=np.zeros((len(classes),len(classes)),int)
for run,fr,rs in [('a1_ds_w100_layers','a1_ds_w100/free_run_large',[0.3]),('a1_ds_w200_large','a1_ds_w200_large/free_run',[0.1,0.3])]:
    dd=np.load(os.path.join(ROOT,run,'a1.npz')); j=json.load(open(os.path.join(ROOT,fr,'summary.json')))
    for ri,r in enumerate(j['radii']):
        if round(float(r),3) not in rs: continue
        pred=cls(dd['pred_T'][int(np.argmin(np.abs(dd['radii']-r)))])
        obs=cls(j['obs'][ri])
        for p,o in zip(pred,obs): M[classes.index(p),classes.index(o)]+=1
im=axs[2].imshow(M,cmap='Blues')
for i in range(len(classes)):
    for j in range(len(classes)):
        if M[i,j]: axs[2].text(j,i,str(M[i,j]),ha='center',va='center',fontsize=9)
axs[2].set_xticks(range(len(classes))); axs[2].set_xticklabels(classes); axs[2].set_yticks(range(len(classes))); axs[2].set_yticklabels(classes)
axs[2].set_xlabel('observed first flip'); axs[2].set_ylabel('predicted first flip'); axs[2].set_title('Prediction vs free run (counts)')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'threshold_geometry.png'),dpi=200); plt.close(fig)
print('saved')
