import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
d=np.load(os.path.join(ROOT,'a1_ds_w100_layers/a1.npz')); radii=d['radii']; dh=d['dh_norm']; layers=d['layers']; K,T=dh.shape[1],dh.shape[3]
fr=(radii>=1e-3)&(radii<=0.1)
alpha=np.full((len(layers),T),np.nan)
for li,lay in enumerate(layers):
    med=np.median(dh[:,:,li,:],axis=1)
    for t in range(T):
        y=med[fr,t]; x=np.log(radii[fr]); ok=y>0
        if ok.sum()>=4: alpha[li,t]=np.polyfit(x[ok],np.log(y[ok]),1)[0]
# smooth
def smooth(a):
    a=np.asarray(a,float); return np.convolve(np.nan_to_num(a,nan=np.nanmedian(a)),np.ones(9)/9,mode='same')
# collapse for selected layers
sel=[1,2,3,7,14,27]; sel=[l for l in sel if l in layers]
fig,axs=plt.subplots(len(sel),2,figsize=(11,3*len(sel)),sharex=True)
for row,lay in enumerate(sel):
    li=list(layers).index(lay); med=np.median(dh[:,:,li,:],axis=1)
    for r in [1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0]:
        i=int(np.argmin(np.abs(radii-r)))
        axs[row,0].plot(np.arange(T),med[i],lw=.8,label=f'r={r:g}')
        axs[row,1].plot(np.arange(T),med[i]/r,lw=.8)
    axs[row,0].set_yscale('log'); axs[row,1].set_yscale('log')
    axs[row,0].set_ylabel(f'layer {lay}\n'+r'$||\delta h||$'); axs[row,1].set_ylabel(r'$||\delta h||/r$')
    axs[row,0].grid(alpha=.25); axs[row,1].grid(alpha=.25)
axs[0,0].legend(fontsize=6,ncol=2); axs[-1,0].set_xlabel('generated token'); axs[-1,1].set_xlabel('generated token')
fig.suptitle('Raw vs r-normalized response by layer (curves should overlap on the right)'); fig.tight_layout()
fig.savefig(os.path.join(OUT,'a1_collapse_by_layer.png'),dpi=200); plt.close(fig)

# alpha heatmap
fig,ax=plt.subplots(figsize=(9,4))
im=ax.imshow(alpha,aspect='auto',origin='lower',cmap='viridis',vmin=0.5,vmax=1.05)
ax.set_yticks(range(len(layers))); ax.set_yticklabels([f'L{l}' for l in layers])
ax.set_xlabel('generated token'); ax.set_title(r'fitted exponent $\alpha(t,\ell)$'); fig.colorbar(im,ax=ax,label=r'$\alpha$')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_alpha_heatmap.png'),dpi=200); plt.close(fig)

# fixed r layer comparison
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for r,ax in zip([0.1,0.3],axs):
    i=int(np.argmin(np.abs(radii-r)))
    for li,lay in enumerate(layers):
        ax.plot(np.arange(T),np.median(dh[i,:,li,:],axis=0),lw=1,label=f'L{lay}')
    ax.set_yscale('log'); ax.set_xlabel('generated token'); ax.set_title(f'r={r:g}'); ax.grid(alpha=.25)
axs[0].set_ylabel(r'median $||\delta h||$'); axs[0].legend(fontsize=7,ncol=2)
fig.suptitle('Layer dependence at fixed r'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_layers_fixed_r.png'),dpi=200); plt.close(fig)

# alpha vs layer at selected t
fig,ax=plt.subplots(figsize=(7,4.5))
for t in [0,16,50,100,123,200]:
    vals=[np.nanmedian(alpha[list(layers).index(l),max(0,t-2):t+3]) for l in layers if l!=0]
    ax.plot([l for l in layers if l!=0],vals,marker='o',label=f't={t}')
ax.set_xlabel('layer'); ax.set_ylabel(r'median $\alpha$'); ax.set_title('Exponent vs layer'); ax.legend(fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_alpha_vs_layer.png'),dpi=200); plt.close(fig)
json.dump(dict(layers=layers.tolist(),radii=radii.tolist(),alpha=alpha.tolist()),open(os.path.join(ROOT,'a1_scaling_layers.json'),'w'))
print('saved')
