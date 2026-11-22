"""Current-state perturbation for large models (port of run_state2.py).

Injects a perturbation into the residual stream at a chosen layer and token
position s (batched over directions), then measures how the response propagates
in depth and time. Same output schema as run_state2.py.

Differences from run_state2.py (DGX Spark):
- big_engine loader (device_map streaming; fp32 14B fits in unified memory),
- lean capture via hooks instead of output_hidden_states for the batch rows,
- margin response via the lm_head row-difference trick (no [B,T,V] logits),
- OOM-adaptive batch size (shared machine).
"""
import argparse
import json
import os

import numpy as np
import torch

from chaos_llm.big_engine import (
    Capture, baseline_pass, full_embeddings, load_big, save_json, set_window,
    teacher_batch, wdiff_tensor,
)


def make_hook(delta, pos):
    def hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[:, pos, :] = h[:, pos, :] + delta
            return (h,) + output[1:]
        h = output.clone()
        h[:, pos, :] = h[:, pos, :] + delta
        return h
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', default='8,16,32,50,75,100,150,200')
    ap.add_argument('--dirs', type=int, default=24)
    ap.add_argument('--radii', default='3e-3,1e-2,3e-2,1e-1')
    ap.add_argument('--layers', default='')
    ap.add_argument('--inject-layer', type=int, default=0)
    ap.add_argument('--batch-dirs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--window', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, 'a1.npz'))
    meta = json.load(open(os.path.join(args.run, 'meta.json')))
    pids = torch.as_tensor(d['prompt_ids'])[None].cuda()
    gids = torch.as_tensor(d['gen_ids'])[None].cuda()
    P = int(meta['prompt_len'])
    T = int(gids.shape[1])
    steps = [int(x) for x in args.steps.split(',')]
    radii = [float(x) for x in args.radii.split(',')]

    model, tok, _mp, info = load_big(args.model, dtype=args.dtype)
    set_window(model, args.window or meta['window'])
    n_layers, D = info['num_layers'], info['hidden_size']
    layers = ([int(x) for x in args.layers.split(',')] if args.layers
              else sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1])))

    E = full_embeddings(model, pids, gids)
    capture = Capture(model, layers)
    m_new, top1, top2, h0 = baseline_pass(model, E, P, T, layers, capture)
    capture.remove()
    del capture
    # must agree with the a1 run (same deterministic teacher-forced pass)
    assert np.allclose(m_new, d['margins'], atol=1e-4), "margins differ from the a1 run"
    wdiff = wdiff_tensor(model, top1, top2)
    m = d['margins']

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
            cur = max(1, args.batch_dirs)
            done = 0
            while done < K:
                B = min(cur, K - done)
                idx = list(range(done, done + B))
                deltas = dirs[idx] * r
                hnd = None
                capture = None
                try:
                    Eb = E.expand(B, -1, -1).clone()
                    if args.inject_layer == 0:
                        Eb[:, P + s, :] += deltas
                    # register the injection hook BEFORE the capture hooks, so the
                    # injected layer itself records the post-injection state
                    elif args.inject_layer > 0:
                        hnd = model.model.layers[args.inject_layer - 1].register_forward_hook(
                            make_hook(deltas, P + s))
                    capture = Capture(model, layers)
                    with torch.inference_mode():
                        res = teacher_batch(model, Eb, P, T, layers, h0, wdiff, capture)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if cur == 1:
                        raise
                    cur = max(1, cur // 2)
                    print(f'OOM -> batch {cur}', flush=True)
                    continue
                finally:
                    if hnd is not None:
                        hnd.remove()
                    if capture is not None:
                        capture.remove()
                dh[ri, si, idx] = res['dh_norm']
                cs[ri, si, idx] = res['cos']
                dm[ri, si, idx] = res['dm']
                done += B
                del Eb, deltas, res
                torch.cuda.empty_cache()
            print(f'r={r:g} s={s} done', flush=True)

    np.savez(os.path.join(args.out, 'state2.npz'), dh=dh, cos=cs, dm=dm,
             steps=np.asarray(steps), radii=np.asarray(radii), layers=np.asarray(layers),
             inject_layer=args.inject_layer, dirs=dirs.float().cpu().numpy(), margins=m)
    save_json(os.path.join(args.out, 'meta.json'), dict(
        run=args.run, steps=steps, radii=radii, dirs=args.dirs, layers=layers,
        inject_layer=args.inject_layer, seed=args.seed, prompt_len=P, T=T,
        window=args.window or meta['window'], model=args.model, dtype=args.dtype,
        engine='run_state2_big'))


if __name__ == '__main__':
    main()
