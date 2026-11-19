"""Long-sequence prompt perturbation: avoids materialising [L,V] logits.

Uses the base model for hidden states and applies lm_head only at the generated
positions.  Memory ~= weights + hidden states + T*V floats.
"""
import argparse, json, os, time
import numpy as np, torch
from chaos_llm.protocol import (build_simplex, full_embeddings, load_model,
                                make_directions, prompt_ids, save_json, set_window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--prompt', default='@prompts.txt')
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--max-new-tokens', type=int, default=3000)
    ap.add_argument('--fixed-tokens-from', required=True)
    ap.add_argument('--num-directions', type=int, default=2)
    ap.add_argument('--radii', default='0.1')
    ap.add_argument('--layers', default='7,14,21,27')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    text = args.prompt
    if text.startswith('@'):
        line = open(text[1:]).read().splitlines()[0]
        text = line.split('\t', 1)[1] if '\t' in line else line
    fd = np.load(args.fixed_tokens_from, allow_pickle=True)
    pids = torch.as_tensor(fd['prompt_ids'], device='cuda', dtype=torch.long)[None]
    gen = torch.as_tensor(fd['gen_ids'][:args.max_new_tokens], device='cuda', dtype=torch.long)[None]
    P = int(pids.shape[1]); T = int(gen.shape[1])
    radii = [float(x) for x in args.radii.split(',')]
    layers = [int(x) for x in args.layers.split(',')]

    model, tok, model_path = load_model(args.model)
    set_window(model, args.window)
    D = model.config.hidden_size
    t0 = time.time()
    Efull = full_embeddings(model, pids, gen)
    mask = torch.ones((1, Efull.shape[1]), dtype=torch.long, device='cuda')
    pos = torch.arange(P - 1, P + T - 1, device='cuda')

    def forward(emb, B=1):
        hs = model.model(inputs_embeds=emb, attention_mask=torch.ones((B, emb.shape[1]), dtype=torch.long, device='cuda'),
                         output_hidden_states=True, use_cache=False)
        z = model.lm_head(hs.last_hidden_state[:, pos, :])          # [B,T,V]
        return hs, z

    with torch.inference_mode():
        hs0, z0 = forward(Efull)
        top2 = torch.topk(z0[0], k=2, dim=-1)
        margins = (top2.values[:, 0] - top2.values[:, 1]).float().cpu().numpy().astype(np.float32)
        top1 = top2.indices[:, 0]; top2i = top2.indices[:, 1]
        h0 = {l: hs0.hidden_states[l][0, P:P + T] for l in layers}
        norm_h0 = np.stack([h0[l].norm(dim=-1).float().cpu().numpy() for l in layers])
        del hs0, z0, top2
        torch.cuda.empty_cache()

    simplex, sub = make_directions(args.num_directions, P, D, 69)
    sub_t = torch.as_tensor(sub, device='cuda', dtype=torch.long)
    K = args.num_directions; nl = len(layers); nr = len(radii)
    dh_norm = np.zeros((nr, K, nl, T), np.float32)
    dm = np.zeros((nr, K, T), np.float32)
    top1v = top1.view(1, T); top2v = top2i.view(1, T)

    for ri, r in enumerate(radii):
        for di in range(K):
            E = Efull.clone()
            delta = torch.zeros(P * D, device='cuda', dtype=torch.float32)
            delta.index_copy_(0, sub_t, torch.as_tensor(simplex[di], device='cuda') * r)
            E[0, :P] += delta.view(P, D)
            with torch.inference_mode():
                hs, z = forward(E)
                z1 = torch.gather(z[0], 1, top1v.T).squeeze(-1)
                z2 = torch.gather(z[0], 1, top2v.T).squeeze(-1)
                dm[ri, di] = ((z1 - z2).float().cpu().numpy() - margins)
                for li, l in enumerate(layers):
                    dh_norm[ri, di, li] = (hs.hidden_states[l][0, P:P + T] - h0[l]).norm(dim=-1).float().cpu().numpy()
                del hs, z, z1, z2
            del E, delta
            torch.cuda.empty_cache()
            print(f'r={r} dir={di} done {time.time()-t0:.0f}s', flush=True)

    eps = 1e-30
    r_c = margins[None, None, :] / np.maximum(np.abs(dm) / np.asarray(radii)[:, None, None], eps)
    pred_T = np.full((nr, K), -1, np.int32)
    for ri, r in enumerate(radii):
        hit = r_c[ri] <= r
        anyhit = hit.any(axis=1)
        pred_T[ri, anyhit] = hit.argmax(axis=1)[anyhit]

    np.savez(os.path.join(args.out, 'a1.npz'), prompt_ids=pids[0].cpu().numpy(),
             gen_ids=gen[0].cpu().numpy(), margins=margins, top1=top1.cpu().numpy(),
             top2=top2i.cpu().numpy(), norm_h0=norm_h0, dh_norm=dh_norm, dm=dm,
             r_c=r_c, pred_T=pred_T, radii=np.asarray(radii), layers=np.asarray(layers),
             simplex=simplex, subspace=sub)
    save_json(os.path.join(args.out, 'meta.json'), dict(model=args.model, window=args.window,
               prompt=text, prompt_len=P, max_new_tokens=T, num_directions=K, radii=radii,
               layers=layers, fixed_tokens_from=args.fixed_tokens_from,
               elapsed_sec=round(time.time() - t0, 1)))
    print('saved', args.out)


if __name__ == '__main__':
    main()
