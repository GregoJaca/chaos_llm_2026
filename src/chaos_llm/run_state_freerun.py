"""Autoregressive validation of a current-state kick.

Feed the prompt plus the baseline tokens up to step s, add r*v to the embedding
of that position, then continue greedily.  Compare with the unperturbed tokens.
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', default='16,50')
    ap.add_argument('--dirs', type=int, default=24)
    ap.add_argument('--r', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, 'a1.npz'))
    meta = json.load(open(os.path.join(args.run, 'meta.json')))
    pids = torch.as_tensor(d['prompt_ids'], device='cuda')[None]
    gids = torch.as_tensor(d['gen_ids'], device='cuda')[None]
    P = int(meta['prompt_len']); T = int(gids.shape[1])
    steps = [int(x) for x in args.steps.split(',')]
    model, tok, _ = load_model(args.model)
    set_window(model, meta['window'])

    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    D = model.config.hidden_size
    dirs = []
    for _ in range(args.dirs):
        v = torch.randn(D, device='cuda', generator=gen)
        dirs.append(v / v.norm())
    dirs = torch.stack(dirs)

    out = np.zeros((len(steps), args.dirs, T), np.int64) - 1
    for si, s in enumerate(steps):
        pos = P + s
        for di in range(args.dirs):
            with torch.inference_mode():
                emb = model.get_input_embeddings()(torch.cat([pids, gids[:, :s + 1]], dim=1))
                emb[0, pos] = emb[0, pos] + args.r * dirs[di]
                o = model(inputs_embeds=emb, use_cache=True, output_hidden_states=False)
                past = o.past_key_values
                nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
                toks = [int(nxt.item())]
                for t in range(s + 2, T):
                    o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
                    toks.append(int(nxt.item()))
                out[si, di, :] = -1
                out[si, di, s + 1:s + 1 + len(toks)] = np.asarray(toks)
            del o, past, emb
            torch.cuda.empty_cache()
        print('step', s, 'done', flush=True)

    ref = gids[0].float().cpu().numpy()
    np.savez(os.path.join(args.out, 'state_freerun.npz'), pred=out, ref=ref,
             steps=np.asarray(steps), r=args.r, dirs=dirs.float().cpu().numpy())
    save_json(os.path.join(args.out, 'meta.json'), dict(run=args.run, steps=steps, dirs=args.dirs,
               r=args.r, seed=args.seed, prompt_len=P, T=T))
    print('saved', args.out)


if __name__ == '__main__':
    main()
