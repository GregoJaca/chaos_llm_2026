import os, json, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
d=np.load(os.path.join(ROOT,'attention_windows/attention.npz'))
wins=[50,100,200]; layers=[7,14,21,27]; res={}
fig,axs=plt.subplots(1,3,figsize=(16,4.5))
for w in wins:
    m=np.load(os.path.join(ROOT,f'margins_windows/w{w}.npz'))['margins'][:300]
    for l in layers:
        H=d[f'w{w}_l{l}']; H=H/H.max()  # normalize per layer/window
        if l==27:
            axs[0].plot(np.arange(300),H,lw=.9,label=f'w{w}')
        # correlation with margin
        rho,p=spearmanr(H,m)
        res[f'w{w}_l{l}']=dict(median=float(np.median(H)),rho=float(rho),p=float(p),
            at16=float(H[16]) if len(H)>16 else None,at44=float(H[44]) if len(H)>44 else None,
            at123=float(H[123]) if len(H)>123 else None,at143=float(H[143]) if len(H)>143 else None)
    axs[0].set_xlabel('generated token'); axs[0].set_ylabel('normalized attention entropy'); axs[0].set_title('Layer 27 attention entropy'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
# scatter entropy vs margin for w100 L27
H=d['w100_l27']; H=H/H.max(); m=np.load(os.path.join(ROOT,'margins_windows/w100.npz'))['margins'][:300]
axs[1].scatter(np.log10(np.maximum(m,1e-4)),H,s=6,alpha=.6); axs[1].set_xlabel('log10 margin'); axs[1].set_ylabel('normalized entropy'); axs[1].set_title('Entropy vs margin (w100 L27)'); axs[1].grid(alpha=.25)
# bar of median entropy per layer/window
labels=[]; vals=[]
for w in wins:
    for l in layers:
        labels.append(f'w{w}L{l}'); vals.append(res[f'w{w}_l{l}']['median'])
axs[2].bar(range(len(vals)),vals); axs[2].set_xticks(range(len(vals))); axs[2].set_xticklabels(labels,rotation=45,fontsize=7); axs[2].set_ylabel('median normalized entropy'); axs[2].set_title('Entropy by window/layer'); axs[2].grid(alpha=.25,axis='y')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'attention_entropy.png'),dpi=200)
json.dump(res,open(os.path.join(ROOT,'attention_entropy.json'),'w'),indent=2)
print(json.dumps(res,indent=1))
