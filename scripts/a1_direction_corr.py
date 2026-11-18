import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
d=np.load(os.path.join(ROOT,'a1_ds_w100_layers/a1.npz')); radii=d['radii']; dh=d['dh_norm']; rc=d['r_c']; K=dh.shape[1]
i=int(np.argmin(np.abs(radii-0.1))); X=np.log(np.maximum(dh[i,:,-1,:],1e-12)); C=np.corrcoef(X); iu=np.triu_indices(K,1); vals=C[iu]
print('prompt direction corr median/p10/p90/f>0.9',float(np.median(vals)),float(np.quantile(vals,.1)),float(np.quantile(vals,.9)),float(np.mean(vals>.9)))
order=leaves_list(linkage(squareform(1-C,checks=False),method='average'))
fig,axs=plt.subplots(1,2,figsize=(12,5))
im=axs[0].imshow(C[np.ix_(order,order)],cmap='viridis',vmin=0,vmax=1); axs[0].set_title('Prompt-direction correlation (r=0.1, last layer)'); axs[0].set_xlabel('direction (clustered)'); axs[0].set_ylabel('direction (clustered)')
fig.colorbar(im,ax=axs[0],label='Pearson r')
rstar=rc[i].min(axis=1); ctrl=rc[i].argmin(axis=1); axs[1].scatter(ctrl,rstar,s=30)
axs[1].set_yscale('log'); axs[1].set_xlabel('controlling near-tie'); axs[1].set_ylabel('r_*'); axs[1].set_title('Direction threshold vs controlling step'); axs[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_direction_corr.png'),dpi=200)
print('saved')
