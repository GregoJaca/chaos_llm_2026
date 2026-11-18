import os, glob, json
import numpy as np, torch
from chaos_llm.analysis.dimension import d2_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_num_threads(8)
ROOT=os.path.expanduser('~/LLM'); OUT=os.path.join(ROOT,'chaos_llm_2026/results/d2'); FIG=os.path.join(ROOT,'chaos_llm_2026/results/figures')
prompts=sorted([d for d in glob.glob(ROOT+'/results_pecs/*/') if os.path.exists(d+'hidden_states_last.pt')])
res={}
for d in prompts:
    name=os.path.basename(d.rstrip('/')); X=torch.load(d+'hidden_states_last.pt',map_location='cpu',weights_only=True).float().squeeze().numpy()
    X=X[:2000]; X=X-X.mean(0,keepdims=True)
    c=d2_curve(X,n_eps=22,theiler=20)
    Xs=X[np.random.default_rng(0).permutation(len(X))]
    cs=d2_curve(Xs,n_eps=22,theiler=0)
    res[name]=dict(raw=c['d2'],shuffled=cs['d2'],r2=c['r2'],r2s=cs['r2'])
    print(name,res[name],flush=True)
json.dump(res,open(os.path.join(OUT,'d2_raw_vs_shuffled.json'),'w'),indent=2)
fig,ax=plt.subplots(figsize=(5,5))
x=[res[k]['shuffled'] for k in res]; y=[res[k]['raw'] for k in res]
ax.scatter(x,y,s=40); lim=[0,max(x+y)*1.1]; ax.plot(lim,lim,'k--')
for k in res: ax.annotate(k[:12],(res[k]['shuffled'],res[k]['raw']),fontsize=6)
ax.set_xlabel('D2 (time-shuffled)'); ax.set_ylabel('D2 (raw)'); ax.set_title('D2 is static geometry, not temporal dynamics'); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'d2_raw_vs_shuffled.png'),dpi=200)
print('saved')
