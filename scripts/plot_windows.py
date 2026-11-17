import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
def load(a1run, frun):
    d=np.load(os.path.join(ROOT,a1run,'a1.npz')); j=json.load(open(os.path.join(ROOT,frun,'summary.json')))
    return d, j
dat={}
for w,a1run,frun in [(100,'a1_ds_w100_large','a1_ds_w100/free_run_large'),(200,'a1_ds_w200_large','a1_ds_w200_large/free_run')]:
    dat[w]=load(a1run,frun)
fig,axs=plt.subplots(1,2,figsize=(13,4.5))
cols={100:'tab:blue',200:'tab:orange'}
for w,(d,j) in dat.items():
    for ri,r in enumerate(j['radii']):
        if round(float(r),3) not in (0.1,0.3): continue
        o=np.asarray(j['obs'][ri]); y=np.where(o<0,300,o)
        axs[0].scatter(np.arange(len(o))+(0.15 if w==200 else 0),y,color=cols[w],marker='o' if r<0.2 else 'x',label=f'w={w} r={r:g}')
axs[0].axhline(300,color='k',ls=':',alpha=.4); axs[0].set_xlabel('direction index'); axs[0].set_ylabel('observed first flip (none drawn at 300)')
axs[0].set_title('Observed flips: window 100 vs 200'); axs[0].legend(fontsize=7,ncol=2); axs[0].grid(alpha=.25)
# flip fractions predicted vs observed
labels=[]; vals=[]; colors=[]
for w,(d,j) in dat.items():
    for ri,r in enumerate(j['radii']):
        if round(float(r),3) not in (0.1,0.3): continue
        o=np.asarray(j['obs'][ri]); labels.append(f'w{w} r{r:g}'); colors.append(cols[w])
        vals.append((np.sum(o>=0)/len(o), np.sum(d['pred_T'][ri]>=0)/len(d['pred_T'][ri])))
x=np.arange(len(labels)); pred=[v[1] for v in vals]; obs=[v[0] for v in vals]
axs[1].bar(x-0.18,pred,width=.36,color='lightgray',label='predicted'); axs[1].bar(x+0.18,obs,width=.36,color=colors,alpha=.75,label='observed')
axs[1].set_xticks(x); axs[1].set_xticklabels(labels,rotation=20); axs[1].set_ylabel('fraction of directions that flip'); axs[1].legend(); axs[1].grid(alpha=.25,axis='y')
axs[1].set_title('Prediction vs observation across windows')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'window_comparison.png'),dpi=200)
print('saved')
