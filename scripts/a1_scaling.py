import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
s=np.load(os.path.join(ROOT,'a1_ds_w100/a1.npz')); l=np.load(os.path.join(ROOT,'a1_ds_w100_large/a1.npz'))
radii=np.concatenate([s['radii'],l['radii']]); dh=np.concatenate([s['dh_norm'],l['dh_norm']],axis=0)
layers=s['layers']; _,idx=np.unique(np.round(radii,10),return_index=True); radii=radii[idx]; dh=dh[idx]
K,T=dh.shape[1],dh.shape[3]
# fitted range: avoid numerical floor and strong saturation
fr=(radii>=1e-3)&(radii<=0.1)
res={}
for li,lay in enumerate(layers):
    med=np.median(dh[:,:,li,:],axis=1)  # [nr,T]
    alpha=np.full(T,np.nan); norm={k:np.zeros_like(med) for k in ['r','sqrt','fit']}
    for t in range(T):
        y=med[fr,t]; x=np.log(radii[fr]); ok=y>0
        if ok.sum()>=4:
            a,b=np.polyfit(x[ok],np.log(y[ok]),1); alpha[t]=a
    norm['r']=med/radii[:,None]
    norm['sqrt']=med/np.sqrt(radii[:,None])
    norm['fit']=med/radii[:,None]**np.where(np.isfinite(alpha),alpha,1.0)[None,:]
    spread={}
    for k in norm:
        z=norm[k][fr]
        z=np.where(z>0,z,np.nan)
        spread[k]=float(np.nanmedian(np.nanstd(np.log(z),axis=0)))
    res[int(lay)]=dict(alpha=alpha.tolist(),spread=spread)
    print('layer',lay,'median alpha',float(np.nanmedian(alpha)),'spread',spread,flush=True)

# raw vs normalized at last layer
li_last=len(layers)-1; med=np.median(dh[:,:,li_last,:],axis=1)
alpha=np.asarray(res[int(layers[li_last])]['alpha']); alpha_s=np.convolve(alpha,np.ones(11)/11,mode='same')
sel_r=[1e-3,3e-3,1e-2,3e-2,0.1,0.3]
fig,axs=plt.subplots(2,2,figsize=(12,8))
for r in sel_r:
    i=int(np.argmin(np.abs(radii-r)))
    axs[0,0].plot(np.arange(T),med[i],lw=1,label=f'r={r:g}')
    axs[0,1].plot(np.arange(T),med[i]/r,lw=1)
    axs[1,0].plot(np.arange(T),med[i]/np.sqrt(r),lw=1)
    axs[1,1].plot(np.arange(T),med[i]/r**np.nan_to_num(alpha_s,nan=1.0),lw=1)
for ax,title in zip(axs.flat,['raw','divide by r','divide by sqrt(r)','divide by r^alpha(t)']):
    ax.set_title(title); ax.set_xlabel('generated token'); ax.set_yscale('log'); ax.grid(alpha=.25)
axs[0,0].set_ylabel(r'$||\delta h_t||$'); axs[0,0].legend(fontsize=7)
fig.suptitle('A1 scaling collapse, last layer (median over 20 directions)'); fig.tight_layout()
fig.savefig(os.path.join(OUT,'a1_scaling_collapse.png'),dpi=200); plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.5))
for li,lay in enumerate(layers):
    a=np.asarray(res[int(lay)]['alpha']); a=np.convolve(a,np.ones(11)/11,mode='same')
    ax.plot(np.arange(T),a,lw=1,label=f'layer {lay}')
ax.set_xlabel('generated token'); ax.set_ylabel(r'fitted exponent $\alpha(t)$'); ax.set_title('Response scaling exponent vs time')
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_exponent_vs_time.png'),dpi=200); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.5))
for t in [0,16,50,100,123,200]:
    vals=[np.nanmedian(np.asarray(res[int(lay)]['alpha'])[max(0,t-2):t+3]) for lay in layers]
    ax.plot(layers,vals,marker='o',label=f't={t}')
ax.set_xlabel('layer'); ax.set_ylabel(r'median $\alpha$ near t'); ax.set_title('Scaling exponent vs layer')
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_exponent_vs_layer.png'),dpi=200); plt.close(fig)

# fixed r, median response across layers
fig,ax=plt.subplots(figsize=(8,4.5))
for li,lay in enumerate(layers):
    medl=np.median(dh[:,:,li,:],axis=1)
    i=int(np.argmin(np.abs(radii-0.1))); ax.plot(np.arange(T),medl[i],lw=1,label=f'layer {lay}')
ax.set_yscale('log'); ax.set_xlabel('generated token'); ax.set_ylabel(r'$||\delta h_t||$'); ax.set_title('Layer comparison at r=0.1')
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'a1_layers_r0.1.png'),dpi=200); plt.close(fig)

json.dump(dict(radii=radii.tolist(),layers=layers.tolist(),res=res),open(os.path.join(ROOT,'a1_scaling.json'),'w'),indent=2)
print('saved')
