"""Layer-resolved amplification of a perturbation injected into one layer.

For each injection layer L, add r*v to the residual stream at one position and
measure the response at every later layer.

Patched for multi-model use (protocol_general):
  * depth from model_info(): injection default = ALL blocks, measurement covers
    hidden_states[0..n_layers] (the original hardcoded 28, which silently missed
    the last layer on a 36-layer model and CRASHED with IndexError on 24 layers)
  * decoder blocks via model_layers() (no direct model.model.layers access)
  * dtype auto; hook arithmetic accumulates in fp32 and rounds once to the
    hidden dtype (bf16-safe)
  * to_np() for tensor->numpy (bf16 tensors cannot .numpy())
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol_general import (
    DEVICE, full_embeddings, load_model, model_info, model_layers, save_json,
    set_window, to_np,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='',
                    help='injection block indices; default = all blocks')
    ap.add_argument('--dirs', type=int, default=6)
    ap.add_argument('--r', type=float, default=0.05)
    ap.add_argument('--pos', type=int, default=50, help='generated position (0-based)')
    ap.add_argument('--seed', type=int, default=3000)
    ap.add_argument('--dtype', default='auto')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, 'a1.npz'))
    meta = json.load(open(os.path.join(args.run, 'meta.json')))
    pids = torch.as_tensor(d['prompt_ids'], device=DEVICE, dtype=torch.long)[None]
    gids = torch.as_tensor(d['gen_ids'], device=DEVICE, dtype=torch.long)[None]
    P = int(meta['prompt_len']); T = int(gids.shape[1])
    if not (0 <= args.pos < T):
        raise ValueError(f'--pos must be in [0, {T - 1}] for a {T}-token trajectory')
    model, tok, _ = load_model(args.model, dtype=args.dtype)
    info = model_info(model, tok=tok)
    D = info['hidden_size']
    n_layers = info['num_layers']
    NH = info['n_hidden_states']
    blocks = model_layers(model)
    if int(d['gen_ids'].max()) >= info['vocab_size']:
        raise ValueError('source run has token ids beyond this model\'s vocab (another model?)')
    inj_layers = ([int(x) for x in args.layers.split(',') if x.strip()]
                  if args.layers.strip() else list(range(n_layers)))
    if max(inj_layers) >= len(blocks) or min(inj_layers) < 0:
        raise ValueError(f'injection layer indices {inj_layers} out of range '
                         f'(model has {len(blocks)} blocks)')
    set_window(model, meta['window'])

    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    dirs = torch.stack([torch.randn(D, device=DEVICE, generator=gen) for _ in range(args.dirs)])
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    E = full_embeddings(model, pids, gids)
    with torch.inference_mode():
        o0 = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device=DEVICE),
                   output_hidden_states=True, use_cache=False)
    h0 = {l: o0.hidden_states[l][0, P:P + T].detach() for l in range(NH)}
    del o0
    torch.cuda.empty_cache()

    pos = P + args.pos
    dh = np.zeros((len(inj_layers), args.dirs, NH, T), np.float32)
    for li, L in enumerate(inj_layers):
        for di in range(args.dirs):
            delta = (args.r * dirs[di])
            def hook(module, inputs, output, L=L, delta=delta):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[:, pos, :] = (h[:, pos, :].float() + delta.float()).to(h.dtype)
                    return (h,) + output[1:]
                h = output.clone()
                h[:, pos, :] = (h[:, pos, :].float() + delta.float()).to(h.dtype)
                return h
            hnd = blocks[L].register_forward_hook(hook)
            with torch.inference_mode():
                o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device=DEVICE),
                          output_hidden_states=True, use_cache=False)
            hnd.remove()
            for l in range(NH):
                dh[li, di, l] = to_np(
                    (o.hidden_states[l][0, P:P + T].float() - h0[l].float()).norm(dim=-1))
            del o
            torch.cuda.empty_cache()
        print('injection layer', L, 'done', flush=True)

    np.savez(os.path.join(args.out, 'layer_probe.npz'), dh=dh, layers=np.asarray(inj_layers),
             pos=args.pos, r=args.r, dirs=to_np(dirs))
    save_json(os.path.join(args.out, 'meta.json'), dict(run=args.run, model=args.model,
               dtype=info['dtype'], params=info['params'], layers=inj_layers,
               num_layers=n_layers, measured_hidden_states=NH, pos=args.pos, r=args.r,
               dirs=args.dirs, seed=args.seed, hidden_size=D, vocab_size=info['vocab_size'],
               prompt_len=P, T=T, window=meta['window']))
    print('saved', args.out)


if __name__ == '__main__':
    main()
