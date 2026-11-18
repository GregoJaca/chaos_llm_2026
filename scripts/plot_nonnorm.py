import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
d=np.load(os.path.join(ROOT,'nonnorm_w100_big/nonnorm.npz')); toks=d['tokens']; layers=d['layers']
mlp_s,mlp_r,mlp_k=d['mlp_smax'],d['mlp_rho'],d['mlp_kappa']
fs,fr,fk=d['full_smax'],d['full_rho'],d['full_kappa']; ang=d['angles']; sl=d['sel_layers']
valid=np.where(~np.all(np.isnan(mlp_s),axis=0))[0]
fig,axs=plt.subplots(1,3,figsize=(16,4.5))
for ti,t in enumerate(toks):
    axs[0].plot(valid,mlp_s[ti,valid],marker='o',lw=1,label=f't={t}')
    axs[1].plot(valid,mlp_r[ti,valid],marker='o',lw=1,label=f't={t}')
    axs[2].plot(valid,mlp_k[ti,valid],marker='o',lw=1,label=f't={t}')
for ax,title in zip(axs,[r'MLP $\sigma_{max}$',r'MLP $\rho(J)$',r'MLP $\kappa=\sigma_{max}/\rho$']):
    ax.set_xlabel('layer'); ax.set_title(title); ax.grid(alpha=.25); ax.legend(fontsize=8)
fig.suptitle('MLP non-normality'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'nonnorm_mlp_big.png'),dpi=200); plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4.2))
for ti,t in enumerate(toks):
    axs[0].plot(sl,fs[ti,sl],marker='o',lw=1,label=f't={t}')
    axs[1].plot(sl,fr[ti,sl],marker='o',lw=1,label=f't={t}')
    axs[2].plot(sl,fk[ti,sl],marker='o',lw=1,label=f't={t}')
for ax,title in zip(axs,[r'Full block $\sigma_{max}$',r'Full block $\rho(J)$',r'Full block $\kappa$']):
    ax.set_xlabel('layer'); ax.set_title(title); ax.grid(alpha=.25); ax.legend(fontsize=8)
fig.suptitle('Full-block non-normality'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'nonnorm_full_big.png'),dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(5,5))
x=mlp_s[~np.isnan(mlp_s)]; y=mlp_r[~np.isnan(mlp_r)]
ax.scatter(y,x,s=30,label='MLP'); x2=fs[~np.isnan(fs)]; y2=fr[~np.isnan(fr)]; ax.scatter(y2,x2,s=30,marker='x',label='full')
lim=[0,max(x.max(),x2.max())*1.05]; ax.plot(lim,lim,'k--',label='sigma=rho')
ax.set_xlabel(r'$\rho(J)$'); ax.set_ylabel(r'$\sigma_{max}(J)$'); ax.set_title('Non-normality: sigma_max vs rho'); ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'nonnorm_scatter_big.png'),dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.5))
for ti,t in enumerate(toks): ax.plot(np.arange(len(ang[ti])),ang[ti],marker='.',ms=3,lw=1,label=f't={t}')
ax.set_xlabel('layer (between l and l+1)'); ax.set_ylabel('principal angle (deg)'); ax.set_title('Top singular-subspace rotation')
ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT,'nonnorm_angles_big.png'),dpi=200); plt.close(fig)
print('saved nonnorm figures')
