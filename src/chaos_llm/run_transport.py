"""Transport kernel: perturb ONE prompt position, measure the response at every generated position.

Saves dh[pos, dir, layer, gen_t] = ||dh|| for a perturbation of norm r applied only at prompt
position `pos`.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from chaos_llm.protocol import full_embeddings, load_model, save_json, set_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--positions", default="all", help="all | comma list of prompt positions")
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--layers", default="1,7,14,21,last")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, "a1.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    pids = torch.as_tensor(d["prompt_ids"]).long()[None].cuda()
    gids = torch.as_tensor(d["gen_ids"]).long()[None].cuda()
    P = int(meta["prompt_len"]); T = int(gids.shape[1])
    model, tok, _ = load_model(args.model, attn="sdpa", dtype=args.dtype)
    set_window(model, meta["window"])
    info = model._chaos_info
    L = int(info["num_layers"])
    layers = [L - 1 if x == "last" else int(x) for x in args.layers.split(",")]
    positions = list(range(P)) if args.positions == "all" else [int(x) for x in args.positions.split(",")]

    E0 = full_embeddings(model, pids, gids)
    D = int(E0.shape[-1])
    with torch.inference_mode():
        o = model(inputs_embeds=E0,
                  attention_mask=torch.ones((1, E0.shape[1]), dtype=torch.long, device="cuda"),
                  output_hidden_states=True, use_cache=False)
    h0 = {l: o.hidden_states[l][0, P:P + T].detach() for l in layers}
    del o
    torch.cuda.empty_cache()

    g = torch.Generator(device="cuda").manual_seed(1234)
    U = torch.randn(len(positions), args.dirs, D, device="cuda", generator=g)
    U = U / U.norm(dim=2, keepdim=True)

    dh = np.zeros((len(positions), args.dirs, len(layers), T), dtype=np.float32)
    t0 = time.time()
    for pi, p in enumerate(positions):
        for di in range(args.dirs):
            E = E0.clone()
            E[0, p] += (args.r * U[pi, di]).to(E.dtype)
            with torch.inference_mode():
                o = model(inputs_embeds=E,
                          attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device="cuda"),
                          output_hidden_states=True, use_cache=False)
            for li, l in enumerate(layers):
                dh[pi, di, li] = (o.hidden_states[l][0, P:P + T] - h0[l]).norm(dim=-1).float().cpu().numpy()
            del o, E
        torch.cuda.empty_cache()
        if pi % 5 == 0:
            print(f"position {p}/{P}: {time.time()-t0:.0f}s", flush=True)

    np.savez(os.path.join(args.out, "transport.npz"), dh=dh, positions=np.asarray(positions),
             layers=np.asarray(layers), r=args.r, dirs=args.dirs, prompt_len=P, T=T)
    save_json(os.path.join(args.out, "meta.json"),
              dict(run=args.run, positions=positions, layers=layers, r=args.r,
                   dirs=args.dirs, elapsed=round(time.time() - t0, 1), model=args.model))
    print("saved", args.out, round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
