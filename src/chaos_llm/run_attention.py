"""Attention entropy vs margins/near-ties, for a few windows."""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, prompt_ids, set_window, full_embeddings, save_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--windows',default='50,100,200'); ap.add_argument('--layers',default='7,14,21,27')
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    d=np.load(os.path.join(args.run,'a1.npz')); meta=json.load(open(os.path.join(args.run,'meta.json')))
    pids=torch.tensor(d['prompt_ids'])[None].cuda(); gids=torch.tensor(d['gen_ids'])[None].cuda()
    P=int(meta['prompt_len']); layers=[int(x) for x in args.layers.split(',')]
    model,tok,_=load_model(args.model); E=full_embeddings(model,pids,gids)
    out={}
    for w in [int(x) for x in args.windows.split(',')]:
        set_window(model,w)
        with torch.inference_mode():
            o=model(inputs_embeds=E,attention_mask=torch.ones((1,E.shape[1]),dtype=torch.long,device='cuda'),
                    output_attentions=True,output_hidden_states=False,use_cache=False)
        ent={}
        for l in layers:
            A=o.attentions[l][0]  # [heads,L,L]
            p=A.clamp_min(1e-12); H=-(p*p.log()).sum(-1)  # [heads,L]
            ent[l]=H.mean(0)[P:P+gids.shape[1]].float().cpu().numpy().astype(np.float32)
        out[w]={str(l):ent[l].tolist() for l in layers}
        print('window',w,'done',flush=True)
        del o
        torch.cuda.empty_cache()
    np.savez(os.path.join(args.out,'attention.npz'),**{f'w{w}_l{l}':np.asarray(out[w][str(l)]) for w in out for l in layers})
    save_json(os.path.join(args.out,'meta.json'),dict(run=args.run,windows=[int(x) for x in args.windows.split(',')],layers=layers))
    print('saved',args.out)

if __name__=='__main__': main()
