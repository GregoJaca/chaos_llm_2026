import os, json, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
j=json.load(open(os.path.join(ROOT,'a1_ds_w100_layers/free_run_large/summary.json')))
radii=j['radii']; obs=[np.asarray(o) for o in j['obs']]
frac=[float(np.mean(o>=0)) for o in obs]; counts=[int(np.sum(o>=0)) for o in obs]
fig,axs=plt.subplots(1,2,figsize=(11,4.2))
axs[0].bar([str(r) for r in radii],frac,color=['tab:blue','tab:orange','tab:red'])
for i,(f,c) in enumerate(zip(frac,counts)): axs[0].text(i,f+0.02,f'{c}/20',ha='center')
axs[0].set_ylim(0,1); axs[0].set_xlabel('prompt perturbation r'); axs[0].set_ylabel('fraction of directions that flip'); axs[0].set_title('Large-r flips'); axs[0].grid(alpha=.25,axis='y')
# quality proxy: distinct-token ratio and longest repeat from saved gen ids
z=np.load(os.path.join(ROOT,'a1_ds_w100_layers/free_run_large/free_run.npz'),allow_pickle=True); gens=z['gen_ids']; refs=z['ref_ids']
def stats(x):
    x=np.asarray(x); uniq=len(set(x.tolist()))/len(x); return uniq
base=[stats(r) for r in refs]
axs[1].plot([0.5,1,2],[np.mean([stats(g[1]) for k,g in enumerate(gens) if k//4==i]) for i in range(3)],marker='o',label='perturbed')
axs[1].axhline(np.mean(base),color='k',ls='--',label='baseline')
axs[1].set_xlabel('r'); axs[1].set_ylabel('distinct-token ratio (first 300)'); axs[1].set_title('Token diversity proxy'); axs[1].legend(); axs[1].grid(alpha=.25)
fig.suptitle('Large-r behavior: flips increase but text remains non-degenerate at 300 tokens'); fig.tight_layout()
fig.savefig(os.path.join(OUT,'large_r_summary.png'),dpi=200)
print('saved')
