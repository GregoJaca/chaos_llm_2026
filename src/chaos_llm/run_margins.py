"""Baseline margins vs attention window, FP32, fixed length."""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, prompt_ids, set_window, generate, save_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--prompt',default='@prompts.txt')
    ap.add_argument('--windows',default='50,100,200,500,1000,4096')
    ap.add_argument('--max-new-tokens',type=int,default=600)
    ap.add_argument('--layers',default='1,2,3,7,14,21,27')
    ap.add_argument('--out',required=True)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    text=args.prompt
    if text.startswith('@'):
        line=open(text[1:]).read().splitlines()[0]; text=line.split('\t',1)[1] if '\t' in line else line
    model,tok,_=load_model(args.model); pids=prompt_ids(tok,text); P=pids.shape[1]
    layers=[int(x) for x in args.layers.split(',')]
    out={}
    for w in [int(x) for x in args.windows.split(',')]:
        set_window(model,w)
        with torch.inference_mode(): base=model.get_input_embeddings()(pids)[0]
        E=base[None].expand(2,-1,-1)
        gen=generate(model,E,args.max_new_tokens)
        assert torch.equal(gen[0],gen[1]), 'r=0 duplicate differs'
        gids=gen[0:1]; T=gids.shape[1]
        Efull=model.get_input_embeddings()(torch.cat([pids,gids],dim=1))
        with torch.inference_mode():
            o=model(inputs_embeds=Efull,attention_mask=torch.ones((1,Efull.shape[1]),dtype=torch.long,device='cuda'),output_hidden_states=True,use_cache=False)
        pos=torch.arange(P-1,P+T-1,device='cuda'); z=o.logits[0][pos]
        top=torch.topk(z,k=2,dim=-1); m=(top.values[:,0]-top.values[:,1]).cpu().numpy().astype(np.float32)
        h={l:o.hidden_states[l][0,P:P+T].cpu().numpy().astype(np.float32) for l in layers}
        np.savez(os.path.join(args.out,f'w{w}.npz'),gen_ids=gids[0].cpu().numpy(),margins=m,
                 top1=top.indices[:,0].cpu().numpy(),top2=top.indices[:,1].cpu().numpy(),
                 **{f'h{l}':h[l] for l in layers})
        im=int(np.argmin(m)); out[w]=dict(T=int(T),min_margin=float(m[im]),min_step=im,
                                          p01=float(np.quantile(m,.01)),p05=float(np.quantile(m,.05)),
                                          median=float(np.median(m)),n_below_0_01=int((m<0.01).sum()),
                                          n_below_0_05=int((m<0.05).sum()))
        print('window',w,out[w],flush=True)
        del o,Efull,gen
        torch.cuda.empty_cache()
    save_json(os.path.join(args.out,'summary.json'),out)
    print('saved',args.out)

if __name__=='__main__': main()
