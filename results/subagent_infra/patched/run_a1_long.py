"""Long-sequence prompt perturbation: avoids materialising [L,V] logits.

Patched for multi-model use (protocol_general). The old version called
``model.model`` + ``model.lm_head`` directly, which only works for Qwen2-style
classes; this version calls the wrapper with ``logits_to_keep=T+1`` so the lm_head
runs only at the generated positions (identical positions/values to the old
``pos = P-1 .. P+T-2`` gather, up to fp rounding).

Also fixed vs the original:
  * --prompt @file now actually reads the file (the old code stored the file NAME
    in meta['prompt'], which poisoned run_thresholds free-runs when reused)
  * dtype auto + bf16-safe tensor->numpy + perturbation cast to embedding dtype
  * layers default from the real depth (DeepSeek-1.5B reproduces 7,14,21,27)
Requires transformers >= 4.50 (logits_to_keep).
"""
import argparse, json, os, time
import numpy as np, torch
from chaos_llm.protocol_general import (
    DEVICE, default_layer_indices, load_model, make_directions, model_info,
    prompt_ids, save_json, set_window, to_np,
)


def read_prompt_arg(prompt_arg):
    text = prompt_arg
    if text.startswith("@"):
        line = open(text[1:]).read().splitlines()[0]
        text = line.split("\t", 1)[1] if "\t" in line else line
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--prompt', default='@prompts.txt')
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--max-new-tokens', type=int, default=3000)
    ap.add_argument('--fixed-tokens-from', required=True)
    ap.add_argument('--num-directions', type=int, default=2)
    ap.add_argument('--radii', default='0.1')
    ap.add_argument('--layers', default='')
    ap.add_argument('--dtype', default='auto')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    text = read_prompt_arg(args.prompt)
    fd = np.load(args.fixed_tokens_from, allow_pickle=True)
    pids = torch.as_tensor(fd['prompt_ids'], device=DEVICE, dtype=torch.long)[None]
    gen = torch.as_tensor(fd['gen_ids'][:args.max_new_tokens], device=DEVICE, dtype=torch.long)[None]
    P = int(pids.shape[1]); T = int(gen.shape[1])
    radii = [float(x) for x in args.radii.split(',')]

    model, tok, model_path = load_model(args.model, dtype=args.dtype)
    info = model_info(model, tok=tok, model_path=model_path)
    D = info['hidden_size']; NL = info['n_hidden_states']
    layers = ([int(x) for x in args.layers.split(',') if x.strip()]
              if args.layers.strip() else default_layer_indices(info['num_layers']))
    if max(layers) >= NL:
        raise ValueError(f'layer indices {layers} out of range (hidden_states has {NL} entries)')
    if int(max(fd['prompt_ids'].max(), fd['gen_ids'][:args.max_new_tokens].max())) >= info['vocab_size']:
        raise ValueError("--fixed-tokens-from contains token ids beyond this model's vocab: "
                         "tokens come from another model")
    set_window(model, args.window)
    t0 = time.time()

    def forward(emb, B=1):
        # logits only for the last T+1 positions (P-1 .. P+T-1); take the first T
        out = model(inputs_embeds=emb,
                    attention_mask=torch.ones((B, emb.shape[1]), dtype=torch.long, device=DEVICE),
                    output_hidden_states=True, use_cache=False, logits_to_keep=T + 1)
        return out, out.logits[:, :T, :]

    Efull = None
    with torch.inference_mode():
        Efull = model.get_input_embeddings()(torch.cat([pids, gen], dim=1))
    emb_dtype = Efull.dtype

    with torch.inference_mode():
        hs0, z0 = forward(Efull)
        top2 = torch.topk(z0[0], k=2, dim=-1)
        margins = (top2.values[:, 0] - top2.values[:, 1]).float().cpu().numpy().astype(np.float32)
        top1 = top2.indices[:, 0]; top2i = top2.indices[:, 1]
        h0 = {l: hs0.hidden_states[l][0, P:P + T] for l in layers}
        norm_h0 = np.stack([to_np(h0[l].norm(dim=-1)) for l in layers])
        del hs0, z0, top2
        torch.cuda.empty_cache()

    simplex, sub = make_directions(args.num_directions, P, D, 69)
    sub_t = torch.as_tensor(sub, device=DEVICE, dtype=torch.long)
    K = args.num_directions; nl = len(layers); nr = len(radii)
    dh_norm = np.zeros((nr, K, nl, T), np.float32)
    dm = np.zeros((nr, K, T), np.float32)
    top1v = top1.view(1, T); top2v = top2i.view(1, T)

    for ri, r in enumerate(radii):
        for di in range(K):
            E = Efull.clone()
            delta = torch.zeros(P * D, device=DEVICE, dtype=torch.float32)
            delta.index_copy_(0, sub_t, torch.as_tensor(simplex[di], device=DEVICE) * r)
            # fp32 accumulate, round once to the model dtype (bf16-safe)
            E[0, :P] = (E[0, :P].float() + delta.view(P, D)).to(emb_dtype)
            with torch.inference_mode():
                hs, z = forward(E)
                z1 = torch.gather(z[0], 1, top1v.T).squeeze(-1)
                z2 = torch.gather(z[0], 1, top2v.T).squeeze(-1)
                dm[ri, di] = to_np(z1.float() - z2.float()) - margins
                for li, l in enumerate(layers):
                    dh_norm[ri, di, li] = to_np(
                        (hs.hidden_states[l][0, P:P + T].float() - h0[l].float()).norm(dim=-1))
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
    save_json(os.path.join(args.out, 'meta.json'), dict(model=args.model, model_path=model_path,
               dtype=info['dtype'], params=info['params'], window=args.window,
               prompt=text, prompt_len=P, max_new_tokens=T, num_directions=K, radii=radii,
               layers=layers, hidden_size=D, num_layers=info['num_layers'],
               vocab_size=info['vocab_size'], model_class=info['model_class'],
               fixed_tokens_from=args.fixed_tokens_from,
               elapsed_sec=round(time.time() - t0, 1)))
    print('saved', args.out)


if __name__ == '__main__':
    main()
