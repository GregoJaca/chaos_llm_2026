import os, json, numpy as np, torch
from transformers import AutoTokenizer
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results')
run=ROOT+'/a1_ds_w100_layers/free_run_large'
z=np.load(run+'/free_run.npz',allow_pickle=True); refs=z['ref_ids']; gens=z['gen_ids']; radii=z['radii']
snap=sorted(__import__('glob').glob(os.path.expanduser('~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/')))[0]
tok=AutoTokenizer.from_pretrained(snap,local_files_only=True)
print('radii',radii,'refs',len(refs),'gens',len(gens))
for r_idx,r in enumerate(radii):
    # chunks: 4 per radius, each chunk 6 dirs
    for c in range(4):
        k=r_idx*4+c
        if k>=len(gens): break
        g=np.asarray(gens[k]); ref=np.asarray(refs[k])
        if c==0:
            text=tok.decode(g[1].tolist(),skip_special_tokens=True)
            bt=tok.decode(ref.tolist(),skip_special_tokens=True)
            def stats(x):
                x=np.asarray(x); uniq=len(set(x.tolist()))/len(x); rep=1+np.argmax(np.diff(np.r_[0,np.where(np.diff(x)!=0)[0]+1,len(x)]))
                return uniq,rep
            print(f'\n=== r={r} chunk {c} ===')
            print('baseline:',bt[:350].replace('\n',' '))
            print('perturbed:',text[:350].replace('\n',' '))
            print('stats baseline/pert uniq,longest_run:',stats(ref),stats(g[1]))
