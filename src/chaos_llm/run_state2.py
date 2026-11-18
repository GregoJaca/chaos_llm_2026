"""Current-state perturbation, deeper version.

Injects a perturbation into the residual stream at a chosen layer and token
position s, then measures how the response propagates in depth and time.

Saved arrays:
  dh[r, s, dir, layer, t]   ||h_pert(t) - h_base(t)|| at each measured layer
  cos[r, s, dir, layer, t]  cosine(h_pert, h_base)
  dm[r, s, dir, t]          change of the top1-top2 logit margin
  dlogit_top[r, s, dir, t]  change of the baseline top-1 logit
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings, save_json


def make_hook(store, inject_layer, pos, delta):
    def hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h.clone()
            h[:, pos, :] = h[:, pos, :] + delta
            return (h,) + output[1:]
        h = output.clone()
        h[:, pos, :] = h[:, pos, :] + delta
        return h
    return hook

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', default='8,16,32,50,75,100,150,200')
    ap.add_argument('--dirs', type=int, default=24)
    ap.add_argument('--radii', default='3e-3,1e-2,3e-2,1e-1')
    ap.add_argument('--layers', default='7,14,21,27')
    ap.add_argument('--inject-layer', type=int, default=0,
                    help='0 = input embeddings, L>0 = residual stream after block L-1')
    ap.add_argument('--batch-dirs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--window', type=int, default=0, help='override the run window')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, 'a1.npz'))
    meta = json.load(open(os.path.join(args.run, 'meta.json')))
    pids = torch.tensor(d['prompt_ids'])[None].cuda()
    gids = torch.tensor(d['gen_ids'])[None].cuda()
    P = int(meta['prompt_len']); T = int(gids.shape[1]); layers = [int(x) for x in args.layers.split(',')]
    steps = [int(x) for x in args.steps.split(',')]
    radii = [float(x) for x in args.radii.split(',')]

    model, tok, _ = load_model(args.model, attn='sdpa')
    set_window(model, args.window or meta['window'])
    D = model.config.hidden_size
    E = full_embeddings(model, pids, gids)
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
                  output_hidden_states=True, use_cache=False)
    h0 = {l: o.hidden_states[l][0, P:P + T].detach() for l in layers}
    top = torch.topk(o.logits[0, P - 1:P + T - 1], k=2, dim=-1)
    m = (top.values[:, 0] - top.values[:, 1]).detach().cpu().numpy()
    t1 = top.indices[:, 0]
    del o
    torch.cuda.empty_cache()

    nr, ns, K, nl = len(radii), len(steps), args.dirs, len(layers)
    dh = np.zeros((nr, ns, K, nl, T), np.float32)
    cs = np.zeros((nr, ns, K, nl, T), np.float32)
    dm = np.zeros((nr, ns, K, T), np.float32)

    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    dirs = []
    for _ in range(K):
        v = torch.randn(D, device='cuda', generator=gen)
        dirs.append(v / v.norm())
    dirs = torch.stack(dirs)

    for ri, r in enumerate(radii):
        for si, s in enumerate(steps):
            for start in range(0, K, args.batch_dirs):
                idx = list(range(start, min(start + args.batch_dirs, K)))
                B = len(idx)
                Eb = E.expand(B, -1, -1).clone()
                deltas = dirs[idx] * r  # [B,D]
                hnd = None
                if args.inject_layer == 0:
                    Eb[:, P + s, :] += deltas
                else:
                    hnd = model.model.layers[args.inject_layer - 1].register_forward_hook(
                        make_hook(None, args.inject_layer, P + s, deltas))
                with torch.inference_mode():
                    oo = model(inputs_embeds=Eb, attention_mask=torch.ones((B, Eb.shape[1]), dtype=torch.long, device='cuda'),
                               output_hidden_states=True, use_cache=False)
                if hnd is not None:
                    hnd.remove()
                for li, l in enumerate(layers):
                    h = oo.hidden_states[l][:, P:P + T, :]
                    diff = h - h0[l][None]
                    dh[ri, si, idx, li] = diff.norm(dim=-1).detach().cpu().numpy()
                    cs[ri, si, idx, li] = torch.nn.functional.cosine_similarity(
                        h, h0[l][None].expand_as(h), dim=-1).detach().cpu().numpy()
                z = oo.logits[:, P - 1:P + T - 1, :]
                z1 = torch.gather(z, 2, t1.view(1, -1, 1).expand(B, -1, -1)).squeeze(-1)
                z2 = torch.gather(z, 2, torch.as_tensor(d['top2'], device='cuda').view(1, -1, 1).expand(B, -1, -1)).squeeze(-1)
                dm[ri, si, idx] = ((z1 - z2).detach().cpu().numpy() - m)
                del oo, Eb, deltas, z, z1, z2
                torch.cuda.empty_cache()
            print(f'r={r:g} s={s} dirs {start}-{start+len(idx)-1} done', flush=True)

    np.savez(os.path.join(args.out, 'state2.npz'), dh=dh, cos=cs, dm=dm,
             steps=np.asarray(steps), radii=np.asarray(radii), layers=np.asarray(layers),
             inject_layer=args.inject_layer, dirs=dirs.cpu().numpy(), margins=m)
    save_json(os.path.join(args.out, 'meta.json'), dict(run=args.run, steps=steps, radii=radii,
               dirs=args.dirs, layers=layers, inject_layer=args.inject_layer, seed=args.seed,
               prompt_len=P, T=T, window=meta['window']))
    print('saved', args.out)


if __name__ == '__main__':
    main()
