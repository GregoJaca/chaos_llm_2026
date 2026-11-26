"""Measure the angle between the response δh and the state h at every saved layer.

Self-contained: baseline + perturbed prompt embeddings, batch = K directions + one
in-batch reference row.  Saves the angle, the radial fraction |cos|, the response
norm and the state norm.  This is the measurement that was missing for most models.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from chaos_llm.protocol import (full_embeddings, load_model, make_directions,
                                prompt_ids, save_json, set_window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--num-directions", type=int, default=64)
    ap.add_argument("--radii", default="0.03,0.1")
    ap.add_argument("--layers", default="0,7,14,21,27")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--fixed-tokens-from", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    layers = [int(x) for x in args.layers.split(",")]

    model, tok, path = load_model(args.model, attn="sdpa", dtype=args.dtype)
    set_window(model, args.window)

    text = args.prompt
    if text.startswith("@"):
        line = open(text[1:]).read().splitlines()[0]
        text = line.split("\t", 1)[1] if "\t" in line else line
    pids = prompt_ids(tok, text)
    P = pids.shape[1]
    D = model.config.hidden_size
    if args.fixed_tokens_from:
        fpath = args.fixed_tokens_from
        if os.path.isdir(fpath):
            fpath = os.path.join(fpath, "a1.npz")
        fd = np.load(fpath, allow_pickle=True)
        pids = torch.as_tensor(fd["prompt_ids"], device="cuda", dtype=torch.long)[None]
        gen_ids = torch.as_tensor(fd["gen_ids"][: args.max_new_tokens],
                                  device="cuda", dtype=torch.long)[None]
        P = pids.shape[1]
    else:
        from chaos_llm.protocol import generate
        E0 = model.get_input_embeddings()(pids)
        gen = generate(model, E0, args.max_new_tokens)
        gen_ids = gen[:, P:]
    T = int(gen_ids.shape[1])
    E = full_embeddings(model, pids, gen_ids)
    L = E.shape[1]
    mask = torch.ones((1, L), dtype=torch.long, device="cuda")

    with torch.inference_mode():
        base = model(inputs_embeds=E, attention_mask=mask,
                     output_hidden_states=True, use_cache=False)
    H0 = [h[0].detach() for h in base.hidden_states]

    simplex, sub = make_directions(args.num_directions, P, D, args.seed)
    sub_t = torch.as_tensor(sub, device="cuda", dtype=torch.long)
    K = args.num_directions
    nr, nl = len(radii), len(layers)
    ang = np.zeros((nr, K, nl, T), np.float32)
    cosv = np.zeros_like(ang)
    dh_norm = np.zeros_like(ang)
    h_norm = np.zeros_like(ang)

    for ri, r in enumerate(radii):
        for k0 in range(0, K, 4):
            idx = list(range(k0, min(k0 + 4, K)))
            B = len(idx) + 1
            Eb = E.expand(B, -1, -1).clone()
            for j, di in enumerate(idx):
                delta = torch.zeros(P * D, device="cuda", dtype=torch.float32)
                delta.index_copy_(0, sub_t, torch.as_tensor(simplex[di], device="cuda") * r)
                Eb[j + 1, :P] += delta.view(P, D)
            with torch.inference_mode():
                out = model(inputs_embeds=Eb, attention_mask=mask.expand(B, -1),
                            output_hidden_states=True, use_cache=False)
            for j, di in enumerate(idx):
                for li, l in enumerate(layers):
                    hp = out.hidden_states[l][j + 1, P:P + T].float().detach()
                    h0 = H0[l][P:P + T].float().detach()
                    d = hp - h0
                    nh = h0.norm(dim=-1)
                    nd = d.norm(dim=-1)
                    c = (d * h0).sum(-1) / (nd * nh + 1e-30)
                    cosv[ri, di, li] = c.cpu().numpy()
                    ang[ri, di, li] = torch.rad2deg(torch.arccos(torch.clamp(c, -1, 1))).cpu().numpy()
                    dh_norm[ri, di, li] = nd.cpu().numpy()
                    h_norm[ri, di, li] = nh.cpu().numpy()
            print(f"r={r} dirs {idx[0]}-{idx[-1]} done", flush=True)

    np.savez(os.path.join(args.out, "angle.npz"),
             angle=ang, cos=cosv, dh_norm=dh_norm, norm_h=h_norm,
             radii=np.asarray(radii), layers=np.asarray(layers),
             prompt_ids=pids[0].cpu().numpy(), gen_ids=gen_ids[0].cpu().numpy(),
             simplex=simplex, subspace=sub)
    save_json(os.path.join(args.out, "meta.json"), dict(
        model=args.model, dtype=args.dtype, window=args.window, prompt_len=int(P),
        max_new_tokens=int(T), num_directions=int(K), radii=radii, layers=layers,
        seed=args.seed, elapsed_sec=round(time.time() - t_start, 1)))
    print("saved", args.out)


if __name__ == "__main__":
    main()


