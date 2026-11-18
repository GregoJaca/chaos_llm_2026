import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
cfg=[('interstellar','a1_ds_w100/free_run_large',[0.1,0.3]),('childhood','a1_childhood_w100/free_run',[0.01,0.1,0.3]),('cybersecurity','a1_cyber_w100/free_run',[0.01,0.1,0.3])]
labels=[]; fr=[]; steps=[]
for name,run,rs in cfg:
    j=json.load(open(os.path.join(ROOT,run,'summary.json')))
    for r,o in zip(j['radii'],j['obs']):
        if round(float(r),3) not in rs: continue
        o=np.asarray(o); labels.append(f'{name}\nr={r:g}'); fr.append(float(np.mean(o>=0)))
        vals=o[o>=0]; steps.append(sorted(set(vals.tolist()))[:4])
fig,ax=plt.subplots(figsize=(10,4.5))
x=np.arange(len(labels)); ax.bar(x,fr,color='tab:blue',alpha=.8)
for i,(f,s) in enumerate(zip(fr,steps)): ax.text(i,f+0.02,f'{int(f*20)}/20\n{s}',ha='center',fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=7); ax.set_ylabel('fraction of directions that flip'); ax.set_ylim(0,1)
ax.set_title('Threshold generality across prompts (window 100, 300 tokens)'); ax.grid(alpha=.25,axis='y')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'prompt_generality.png'),dpi=200)
print('saved')
