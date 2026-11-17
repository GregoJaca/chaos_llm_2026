import os, glob, json, collections
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.expanduser('~/LLM'); DS=ROOT+'/TDK_LLM/runs_aug/results_perturbations_OLD'
OUT=ROOT+'/chaos_llm_2026/results/figures'; os.makedirs(OUT,exist_ok=True)
# collect runs
runs=[]
for d in glob.glob(DS+'/*/'):
    cf=d+'config.json'
    if not os.path.exists(cf) or not glob.glob(d+'*layer_27.pt'): continue
    c=json.load(open(cf)); runs.append(dict(dir=d, r=float(c['radius']), seed=int(c['seed']), prompt=c.get('prompt_text','?')))
print('runs',len(runs))
groups=collections.defaultdict(list)
for x in runs: groups[(x['prompt'],x['seed'])].append(x)
key=max(groups,key=lambda k: len({round(x['r'],10) for x in groups[k]}))
items=sorted(groups[key],key=lambda x:x['r'])
print('group',key,'n',len(items),'radii',[x['r'] for x in items])
target=[1e-4,1e-3,1e-2,1e-1]
sel=[]
for t in target:
    x=min(items,key=lambda z: abs(np.log10(max(z['r'],1e-12))-np.log10(t)))
    if x not in sel: sel.append(x)
print('selected',[(x['r'],x['dir']) for x in sel])

def load_layer(d,l):
    p=glob.glob(d+f'*layer_{l}.pt')[0]
    t=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    return t.float().squeeze().numpy()  # [N,D]

curves={}
for x in sel:
    d=x['dir']; r=x['r']; ds=[]
    for l in range(28):
        H=load_layer(d,l)
        H=H[:101]  # condition 0 + 100
        H=H-H.mean(0,keepdims=True)
        D=torch.cdist(torch.tensor(H[:1]),torch.tensor(H))[0].numpy()
        ds.append(np.median(D[1:]))
    ds=np.asarray(ds); curves[r]=(ds,ds[0])
    print('r',r,'d0',ds[0],'max',ds.max(),'final',ds[-1],flush=True)

rhos=[1.5,2.0,4.0]
res={}
for rho in rhos:
    rows=[]
    for r,(ds,d0) in curves.items():
        hit=np.where(ds>=rho*d0)[0]
        tau=int(hit[0]) if len(hit) else None
        rows.append(dict(r=r,tau=tau,lam=(np.log(rho)/tau if tau and tau>0 else None),cens=tau is None))
    res[rho]=rows
    print('rho',rho,rows)

fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for r,(ds,d0) in curves.items():
    axs[0].plot(np.arange(28),ds/d0,marker='.',ms=3,lw=1,label=f'r={r:g}')
axs[0].set_xlabel('layer'); axs[0].set_ylabel('d(layer)/d(0)'); axs[0].set_title('Depth separation growth')
axs[0].set_yscale('log'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
for rho in rhos:
    rs=[x['r'] for x in res[rho] if x['lam'] is not None]; ls=[x['lam'] for x in res[rho] if x['lam'] is not None]
    axs[1].plot(rs,ls,marker='o',label=f'rho={rho}')
axs[1].set_xscale('log'); axs[1].set_xlabel('r'); axs[1].set_ylabel('depth FSLE lambda')
axs[1].set_title('Depth-axis FSLE'); axs[1].legend(fontsize=8); axs[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT+'/fsle_depth.png',dpi=200)
json.dump(dict(group=list(key),selected=[x['r'] for x in sel],curves={str(r):v[0].tolist() for r,v in curves.items()},res={str(k):v for k,v in res.items()}),open(ROOT+'/chaos_llm_2026/results/fsle_depth.json','w'),indent=2)
print('saved')

