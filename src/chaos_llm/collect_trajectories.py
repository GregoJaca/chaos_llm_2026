"""Collect long greedy trajectories with hidden states, FP32, regular settings."""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, prompt_ids, set_window, save_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--prompts',required=True)
    ap.add_argument('--window',type=int,default=4096)
    ap.add_argument('--max-new-tokens',type=int,default=1000)
    ap.add_argument('--layers',default='7,14,21,27')
    ap.add_argument('--out',required=True)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    model,tok,_=load_model(args.model); set_window(model,args.window)
    layers=[int(x) for x in args.layers.split(',')]
    lines=[l for l in open(args.prompts).read().splitlines() if l.strip()]
    for line in lines:
        name,text=line.split('\t',1) if '\t' in line else (f'prompt_{len(os.listdir(args.out))}',line)
        pids=prompt_ids(tok,text); P=pids.shape[1]
        with torch.inference_mode(): emb=model.get_input_embeddings()(pids)
        past=None; next_ids=None; gen=[]; hs={l:[] for l in layers}; full_hs=[]
        mamba_cache=False; pos=P
        for step in range(args.max_new_tokens):
            with torch.inference_mode():
                if past is None:
                    o=model(inputs_embeds=emb,use_cache=True,output_hidden_states=True,return_dict=True)
                    if 'past_key_values' in o:
                        past=o.past_key_values
                    elif getattr(o,'cache_params',None) is not None:
                        past=o.cache_params; mamba_cache=True
                elif mamba_cache:
                    o=model(input_ids=next_ids,cache_params=past,
                            cache_position=torch.tensor([pos],device=next_ids.device),
                            use_cache=True,output_hidden_states=True,return_dict=True)
                    pos+=1
                else:
                    o=model(input_ids=next_ids,past_key_values=past,use_cache=True,output_hidden_states=True,return_dict=True)
            for l in layers: hs[l].append(o.hidden_states[l][0,-1].detach().float().cpu().numpy().astype(np.float32))
            full_hs.append(o.hidden_states[-1][0,-1].detach().float().cpu().numpy().astype(np.float32))
            nxt=torch.argmax(o.logits[0,-1]).view(1,1)
            gen.append(int(nxt.item())); next_ids=nxt
            del o
        gen=np.asarray(gen,dtype=np.int32)
        np.savez(os.path.join(args.out,f'{name}.npz'),prompt_ids=pids[0].cpu().numpy(),gen_ids=gen,
                 h_last=np.stack(full_hs),**{f'h{l}':np.stack(hs[l]) for l in layers})
        save_json(os.path.join(args.out,f'{name}.json'),dict(name=name,text=text,prompt_len=int(P),
                 max_new_tokens=args.max_new_tokens,window=args.window,layers=layers))
        print(name,'done',len(gen),flush=True)

if __name__=='__main__': main()
