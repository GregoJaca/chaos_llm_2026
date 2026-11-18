"""Layer-resolved amplification of a perturbation injected into one layer.

For each injection layer L, add r*v to the residual stream at one position and
measure the response at every later layer.  This settles where amplification
happens without the ambiguity of same-position tangent products.
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default=','.join(str(i) for i in range(0, 28)))
    ap.add_argument('--dirs', type=int, default=6)
    ap.add_argument('--r', type=float, default=0.05)
    ap.add_argument('--pos', type=int, default=50, help='generated position (0-based)')
    ap.add_argument('--seed', type=int, default=3000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, 'a1.npz'))
    meta = json.load(open(os.path.join(args.run, 'meta.json')))
    pids = torch.as_tensor(d['prompt_ids'], device='cuda')[None]
    gids = torch.as_tensor(d['gen_ids'], device='cuda')[None]
    P = int(meta['prompt_len']); T = int(gids.shape[1])
    model, tok, _ = load_model(args.model)
    set_window(model, meta['window'])
    D = model.config.hidden_size
    inj_layers = [int(x) for x in args.layers.split(',')]

    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    dirs = torch.stack([torch.randn(D, device='cuda', generator=gen) for _ in range(args.dirs)])
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    E = full_embeddings(model, pids, gids)
    with torch.inference_mode():
        o0 = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
                   output_hidden_states=True, use_cache=False)
    h0 = {l: o0.hidden_states[l][0, P:P + T].detach() for l in range(28)}
    del o0
    torch.cuda.empty_cache()

    pos = P + args.pos
    dh = np.zeros((len(inj_layers), args.dirs, 28, T), np.float32)
    for li, L in enumerate(inj_layers):
        for di in range(args.dirs):
            delta = (args.r * dirs[di])
            def hook(module, inputs, output, L=L, delta=delta):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[:, pos, :] = h[:, pos, :] + delta
                    return (h,) + output[1:]
                h = output.clone()
                h[:, pos, :] = h[:, pos, :] + delta
                return h
            hnd = model.model.layers[L].register_forward_hook(hook)
            with torch.inference_mode():
                o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
                          output_hidden_states=True, use_cache=False)
            hnd.remove()
            for l in range(28):
                dh[li, di, l] = (o.hidden_states[l][0, P:P + T].detach() - h0[l]).norm(dim=-1).cpu().numpy()
            del o
            torch.cuda.empty_cache()
        print('injection layer', L, 'done', flush=True)

    np.savez(os.path.join(args.out, 'layer_probe.npz'), dh=dh, layers=np.asarray(inj_layers),
             pos=args.pos, r=args.r, dirs=dirs.cpu().numpy())
    save_json(os.path.join(args.out, 'meta.json'), dict(run=args.run, layers=inj_layers,
               pos=args.pos, r=args.r, dirs=args.dirs, seed=args.seed, prompt_len=P, T=T))
    print('saved', args.out)


if __name__ == '__main__':
    main()
