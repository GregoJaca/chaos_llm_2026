import os, glob, json
import numpy as np, torch
from safetensors import safe_open
from chaos_llm.analysis.dimension import d2_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_num_threads(8)
ROOT=os.path.expanduser('~/LLM'); OUT=os.path.join(ROOT,'chaos_llm_2026/results/d2'); FIG=os.path.join(ROOT,'chaos_llm_2026/results/figures')
os.makedirs(OUT,exist_ok=True); os.makedirs(FIG,exist_ok=True)
def load(p): return torch.load(p,map_location='cpu',weights_only=True).float().squeeze().numpy()
def prep(X,n=3000,dedupe=False):
    if dedupe: X=np.unique(X,axis=0)
    X=X[:n]; return X-X.mean(0,keepdims=True)
def delay(X,m,tau):
    N=len(X)-(m-1)*tau; return np.stack([X[i*tau:i*tau+N] for i in range(m)],axis=1).reshape(N,-1)
def twonn(X,n=2000):
    X=prep(X,n=n); D=torch.cdist(torch.tensor(X),torch.tensor(X)).numpy(); np.fill_diagonal(D,np.inf)
    D.sort(axis=1); mu=D[:,1]/D[:,0]; mu=mu[np.isfinite(mu)&(mu>1)]
    mu=np.sort(mu); N=len(mu); F=(np.arange(N)+1)/N
    ok=(1-F)>0; slope=np.polyfit(np.log(mu[ok]),-np.log(1-F[ok]),1)[0]
    return float(slope), N
def pca_whiten(X,q):
    Xt=torch.tensor(X,dtype=torch.float32); U,S,V=torch.pca_lowrank(Xt,q=q,center=True)
    Z=Xt@V; Z=Z/(S/np.sqrt(len(X))+1e-8); return Z.numpy()
# data
last=load(ROOT+'/results_pecs/cybersecurity_threats_defenses/hidden_states_last.pt')
first=load(ROOT+'/results_pecs/cybersecurity_threats_defenses/hidden_states_first.pt')
long=load(ROOT+'/launch_sep/interstellar_propulsion_review_0_0.00035/hidden_states_cond_0_layer_-1.pt')
W=None
try:
    snap=sorted(glob.glob(os.path.expanduser('~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/')))[0]
    wm=json.load(open(os.path.join(snap,'model.safetensors.index.json')))['weight_map']['model.embed_tokens.weight']
    f=safe_open(os.path.join(snap,wm),framework='pt'); W=f.get_tensor('model.embed_tokens.weight').float().numpy()
    W=W[np.random.default_rng(0).choice(len(W),size=3000,replace=False)]
except Exception as e: print('no vocab',e)
res={}
for name,X,ded in [('last',last,False),('first',first,True),('launch_last',long,False)]:
    Xp=prep(X,n=3000,dedupe=ded); c=d2_curve(Xp,n_eps=25,theiler=20)
    res[name]=dict(d2=c['d2'],r2=c['r2'],range=c['fit_range'],n=len(Xp))
    print(name,res[name],flush=True)
# delay embeddings on last
delay_res={}
for tau in [1,5,10]:
    for m in [3,5,10]:
        Xd=delay(prep(last,n=2500),m,tau); c=d2_curve(Xd,n_eps=25,theiler=20)
        delay_res[f'm{m}_tau{tau}']=dict(d2=c['d2'],r2=c['r2'],range=c['fit_range'],n=len(Xd))
        print('delay',m,tau,delay_res[f'm{m}_tau{tau}'],flush=True)
# PCA whitening
pca_res={}
for q in [20,50,100]:
    Xp=pca_whiten(prep(last,n=3000),q); c=d2_curve(Xp,n_eps=25,theiler=20)
    pca_res[f'q{q}']=dict(d2=c['d2'],r2=c['r2'],range=c['fit_range'])
    print('pca',q,pca_res[f'q{q}'],flush=True)
# TwoNN
tw={}
for name,X,ded in [('last',last,False),('first',first,True),('launch_last',long,False),('vocab',W,False)]:
    if X is None: continue
    tw[name]=twonn(X,n=2000); print('twonn',name,tw[name],flush=True)
json.dump(dict(raw=res,delay=delay_res,pca=pca_res,twonn=tw),open(os.path.join(OUT,'d2_explore.json'),'w'),indent=2)
# plots: C(eps) examples and D2 vs delay
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for name,X,ded in [('last',last,False),('first',first,True),('launch_last',long,False)]:
    Xp=prep(X,n=3000,dedupe=ded); c=d2_curve(Xp,n_eps=25,theiler=20); e=np.asarray(c['eps']); C=np.asarray(c['C'])
    axs[0].loglog(e,C,lw=1,label=f'{name} D2={c["d2"]:.1f}'); axs[1].plot(np.log10(e/np.median(e)),np.gradient(np.log(C),np.log(e)),lw=1,label=name)
axs[0].set_xlabel('epsilon'); axs[0].set_ylabel('C(epsilon)'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
axs[1].set_xlabel('log10(eps/median)'); axs[1].set_ylabel('local slope'); axs[1].legend(fontsize=8); axs[1].grid(alpha=.25)
fig.suptitle('D2 scaling examples'); fig.tight_layout(); fig.savefig(os.path.join(FIG,'d2_scaling_examples.png'),dpi=200); plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.2))
labels=list(delay_res); axs[0].bar(range(len(labels)),[delay_res[k]['d2'] for k in labels]); axs[0].set_xticks(range(len(labels))); axs[0].set_xticklabels(labels,rotation=45); axs[0].set_ylabel('D2'); axs[0].set_title('Delay-embedding D2')
axs[1].bar(list(tw),[tw[k][0] for k in tw]); axs[1].set_ylabel('TwoNN dimension'); axs[1].set_title('TwoNN intrinsic dimension'); axs[1].grid(alpha=.25,axis='y')
fig.tight_layout(); fig.savefig(os.path.join(FIG,'d2_delay_twonn.png'),dpi=200); plt.close(fig)
print('saved')
