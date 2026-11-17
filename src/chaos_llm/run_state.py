"""Current-state (input-embedding) perturbation: two-time response R(t,s)."""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings, save_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--steps',default='16,50,100'); ap.add_argument('--dirs',type=int,default=8)
    ap.add_argument('--r',type=float,default=1e-3); ap.add_argument('--layer',type=int,default=27)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    d=np.load(os.path.join(args.run,'a1.npz')); meta=json.load(open(os.path.join(args.run,'meta.json')))
    pids=torch.tensor(d['prompt_ids'])[None].cuda(); gids=torch.tensor(d['gen_ids'])[None].cuda()
    P=int(meta['prompt_len']); steps=[int(x) for x in args.steps.split(',')]; T=gids.shape[1]; D=1536
    model,tok,_=load_model(args.model,attn='sdpa'); set_window(model,meta['window'])
    E=full_embeddings(model,pids,gids)
    with torch.inference_mode():
        o=model(inputs_embeds=E,attention_mask=torch.ones((1,E.shape[1]),dtype=torch.long,device='cuda'),output_hidden_states=True,use_cache=False)
    h0=o.hidden_states[args.layer][0,P:P+T].detach()
    dh=np.zeros((len(steps),args.dirs,T),np.float32); dm=np.zeros((len(steps),args.dirs,T),np.float32)
    top=torch.topk(o.logits[0,P-1:P+T-1],k=2,dim=-1); m=(top.values[:,0]-top.values[:,1]).detach().cpu().numpy()
    t1=top.indices[:,0]; t2=top.indices[:,1]
    for si,s in enumerate(steps):
        for di in range(args.dirs):
            g=torch.Generator(device='cuda').manual_seed(1000+di)
            v=torch.randn(D,device='cuda',generator=g); v/=v.norm()
            E2=E.clone(); E2[0,P+s]+=args.r*v
            with torch.inference_mode():
                oo=model(inputs_embeds=E2,attention_mask=torch.ones((1,E2.shape[1]),dtype=torch.long,device='cuda'),output_hidden_states=True,use_cache=False)
            dh[si,di]= (oo.hidden_states[args.layer][0,P:P+T]-h0).norm(dim=-1).detach().cpu().numpy()
            z=oo.logits[0,P-1:P+T-1]
            z1=torch.gather(z,1,t1[:,None]).squeeze(1); z2=torch.gather(z,1,t2[:,None]).squeeze(1)
            dm[si,di]=((z1-z2).detach().cpu().numpy()-m)
            del oo,E2
        print('step',s,'done',flush=True)
    np.savez(os.path.join(args.out,'state.npz'),dh=dh,dm=dm,steps=np.asarray(steps),r=args.r,layer=args.layer)
    save_json(os.path.join(args.out,'meta.json'),dict(run=args.run,steps=steps,dirs=args.dirs,r=args.r,layer=args.layer,prompt_len=P,T=T))
    print('saved',args.out)

if __name__=='__main__': main()
