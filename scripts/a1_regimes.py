import os, numpy as np, json
ROOT=os.path.expanduser('~/LLM/chaos_llm_2026/results')
d=np.load(os.path.join(ROOT,'a1_ds_w100_layers/a1.npz')); radii=d['radii']; dh=d['dh_norm']; layers=d['layers']
i=int(np.argmin(np.abs(radii-0.1))); out={}
for li,l in enumerate(layers):
    y=np.median(dh[i,:,li,:],axis=0); y=np.maximum(y,1e-12); t=np.arange(len(y)); pk=int(np.argmax(y))
    # fit log y vs t before/after peak, excluding first/last few and floor
    pre=(t<max(pk,5)); post=(t>pk)&(y>y[pk]*0.02)
    a1=np.polyfit(t[pre],np.log(y[pre]),1)[0] if pre.sum()>3 else None
    a2=np.polyfit(t[post],np.log(y[post]),1)[0] if post.sum()>3 else None
    out[int(l)]=dict(peak_t=pk,peak=float(y[pk]),rise_slope=float(a1) if a1 is not None else None,
                     decay_slope=float(a2) if a2 is not None else None)
    print(l,out[int(l)])
json.dump(out,open(os.path.join(ROOT,'a1_regimes.json'),'w'),indent=2)
