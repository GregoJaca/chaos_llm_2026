import os, glob, json
import numpy as np, torch
from safetensors import safe_open
from chaos_llm.analysis.dimension import d2_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_num_threads(8)
ROOT=os.path.expanduser('~/LLM'); OUT=os.path.join(ROOT,'chaos_llm_2026/results/d2'); FIG=os.path.join(ROOT,'chaos_llm_2026/results/figures')
def load(p): return torch.load(p,map_location='cpu',weights_only=True).float().squeeze().numpy()
def prep(X,n=2000):
    X=X[:n]; return X-X.mean(0,keepdims=True)
def twonn(X,n=2000):
    X=prep(X,n=n); D=torch.cdist(torch.tensor(X),torch.tensor(X)).numpy(); np.fill_diagonal(D,np.inf)
    D.sort(axis=1); mu=D[:,1]/np.maximum(D[:,0],1e-30); mu=mu[np.isfinite(mu)&(mu>1)]
    mu=np.sort(mu); N=len(mu); F=(np.arange(N)+1)/N; ok=(1-F)>1e-9
    return float(np.polyfit(np.log(mu[ok]),-np.log(1-F[ok]),1)[0]),N
def phase_surrogate(X,seed=0):
    rng=np.random.default_rng(seed); F=np.fft.rfft(X,axis=0); ph=np.exp(2j*np.pi*rng.random(F.shape)); return np.fft.irfft(F*ph,n=X.shape[0],axis=0)
def vocab():
    snap=sorted(glob.glob(os.path.expanduser('~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/')))[0]
    cands=glob.glob(os.path.join(snap,'model*.safetensors'))
    for p in cands:
        try:
            f=safe_open(p,framework='pt'); W=f.get_tensor('model.embed_tokens.weight').float().numpy()
            return W
        except Exception: pass
    return None
X=load(ROOT+'/launch_sep/interstellar_propulsion_review_0_0.00035/hidden_states_cond_0_layer_-1.pt')
X=prep(X,n=2500)
W=vocab(); V=None
if W is not None:
    V=W[np.random.default_rng(0).choice(len(W),size=2500,replace=False)]
res={}
for name,Z in [('raw',X),('shuffled',X[np.random.default_rng(1).permutation(len(X))]),('phase',phase_surrogate(X,2))]:
    c=d2_curve(Z,n_eps=25,theiler=20 if name=='raw' else 0); res[name]=dict(d2=c['d2'],r2=c['r2'],range=c['fit_range'],twonn=twonn(Z,n=2000))
    print(name,res[name],flush=True)
if V is not None:
    c=d2_curve(prep(V,n=2500),n_eps=25,theiler=0); res['vocab']=dict(d2=c['d2'],r2=c['r2'],range=c['fit_range'],twonn=twonn(V,n=2000)); print('vocab',res['vocab'],flush=True)
json.dump(res,open(os.path.join(OUT,'d2_controls.json'),'w'),indent=2)
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for name in res:
    Z={'raw':X,'shuffled':X[np.random.default_rng(1).permutation(len(X))],'phase':phase_surrogate(X,2)}.get(name)
    if name=='vocab': Z=V
    c=d2_curve(Z,n_eps=25,theiler=20 if name=='raw' else 0); e=np.asarray(c['eps']); C=np.asarray(c['C'])
    axs[0].loglog(e/np.median(e),C,lw=1,label=f'{name} D2={c["d2"]:.1f}')
    axs[1].plot(np.log10(e/np.median(e)),np.gradient(np.log(C),np.log(e)),lw=1,label=name)
axs[0].set_xlabel('eps/median eps'); axs[0].set_ylabel('C(eps)'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
axs[1].set_xlabel('log10(eps/median)'); axs[1].set_ylabel('local slope'); axs[1].legend(fontsize=8); axs[1].grid(alpha=.25)
fig.suptitle('D2 controls: raw vs shuffled vs phase surrogate vs vocab'); fig.tight_layout(); fig.savefig(os.path.join(FIG,'d2_controls.png'),dpi=200); plt.close(fig)
print('saved')
