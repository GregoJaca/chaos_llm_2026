import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results'); OUT=os.path.join(ROOT,'figures')
S=json.load(open(os.path.join(ROOT,'margins_windows/summary.json')))
wins=[50,100,200,500]
fig,axs=plt.subplots(2,2,figsize=(12,7),sharey=True)
for w,ax in zip(wins,axs.flat):
    d=np.load(os.path.join(ROOT,f'margins_windows/w{w}.npz')); m=d['margins']; im=int(np.argmin(m))
    ax.plot(np.arange(len(m)),np.maximum(m,1e-6),lw=.8)
    ax.axvline(im,color='r',ls='--',alpha=.6); ax.text(im,max(m[im],1e-5),f' min={m[im]:.3g} @ {im}',color='r',fontsize=8)
    ax.set_yscale('log'); ax.set_title(f'window {w}: min={m[im]:.3g} at t={im}'); ax.set_xlabel('generated token'); ax.grid(alpha=.25)
axs[0,0].set_ylabel('top1 - top2 margin'); axs[1,0].set_ylabel('top1 - top2 margin')
fig.suptitle('Baseline margin spectrum vs attention window (FP32, 600 tokens)'); fig.tight_layout()
fig.savefig(os.path.join(OUT,'margins_windows.png'),dpi=200); plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(14,4.2))
ws=sorted(int(k) for k in S); mins=[S[str(w)]['min_margin'] for w in ws]; n05=[S[str(w)]['n_below_0_05'] for w in ws]; med=[S[str(w)]['median'] for w in ws]
axs[0].plot(ws,mins,marker='o'); axs[0].set_xscale('log'); axs[0].set_xlabel('window'); axs[0].set_ylabel('minimum margin'); axs[0].set_title('Smallest margin vs window'); axs[0].grid(alpha=.25)
axs[1].plot(ws,n05,marker='o',color='tab:red'); axs[1].set_xscale('log'); axs[1].set_xlabel('window'); axs[1].set_ylabel('count margin < 0.05'); axs[1].set_title('Number of fragile steps'); axs[1].grid(alpha=.25)
axs[2].plot(ws,med,marker='o',color='tab:green'); axs[2].set_xscale('log'); axs[2].set_xlabel('window'); axs[2].set_ylabel('median margin'); axs[2].set_title('Typical margin'); axs[2].grid(alpha=.25)
fig.suptitle('Margin spectrum changes with attention window'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'margins_vs_window.png'),dpi=200); plt.close(fig)
print('saved')
