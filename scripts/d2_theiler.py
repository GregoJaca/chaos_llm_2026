import os, glob, json
import numpy as np, torch
from chaos_llm.analysis.dimension import d2_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_num_threads(8)
ROOT=os.path.expanduser('~/LLM'); OUT=os.path.join(ROOT,'chaos_llm_2026/results/d2'); FIG=os.path.join(ROOT,'chaos_llm_2026/results/figures')
def load(p): return torch.load(p,map_location='cpu',weights_only=True).float().squeeze().numpy()
def prep(X,n=2000,ded=False):
    if ded: X=np.unique(X,axis=0)
    X=X[:n]; return X-X.mean(0,keepdims=True)
prompts=sorted([d for d in glob.glob(ROOT+'/results_pecs/*/') if os.path.exists(d+'hidden_states_last.pt')])[:3]
res={}
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for d in prompts:
    name=os.path.basename(d.rstrip('/'))
    for tag,fn,ded in [('first','hidden_states_first.pt',True),('last','hidden_states_last.pt',False)]:
        X=prep(load(d+fn),ded=ded); res[f'{name}_{tag}']={}
        for W in [0,10,50]:
            c=d2_curve(X,n_eps=30,theiler=W)
            res[f'{name}_{tag}'][f'W{W}']=dict(d2=c['d2'],r2=c['r2'],range=c['fit_range'])
            e=np.asarray(c['eps']); C=np.asarray(c['C']); sl=np.gradient(np.log(C),np.log(e))
            axs[0].plot(np.log10(e/np.median(e)),sl,lw=1,label=f'{name[:8]} {tag} W={W}')
            print(name,tag,'W',W,'D2',round(c['d2'],2),'R2',round(c['r2'],3),flush=True)
json.dump(res,open(os.path.join(OUT,'d2_theiler.json'),'w'),indent=2)
axs[0].set_xlabel('log10(eps / median eps)'); axs[0].set_ylabel('local slope'); axs[0].set_title('D2 local slope vs Theiler'); axs[0].legend(fontsize=5,ncol=2); axs[0].grid(alpha=.25)
axs[1].hist([res[k]['W0']['d2'] for k in res if k.endswith('last')],bins=10,alpha=.6,label='last W=0')
axs[1].hist([res[k]['W50']['d2'] for k in res if k.endswith('last')],bins=10,alpha=.6,label='last W=50')
axs[1].axvline(2*np.log10(2000),color='k',ls='--',label='Eckmann-Ruelle 2log10N')
axs[1].set_xlabel('D2'); axs[1].legend(fontsize=8); axs[1].set_title('Effect of Theiler window')
fig.tight_layout(); fig.savefig(os.path.join(FIG,'d2_theiler.png'),dpi=200)
print('saved')
