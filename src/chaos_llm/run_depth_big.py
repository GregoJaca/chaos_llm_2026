"""Depth tangent spectrum for large models (port of run_depth.py).

Propagates k tangent vectors through all layers per token (layer-wise JVP +
QR renormalisation, Benettin-style); reports the depth spectrum
lambda_i = (1/l) log sigma_i(product) per token, local gains, and
h_depth = sum of positive exponents.

Differences from run_depth.py: big_engine loader (device_map streaming, fp32),
eager attention kept (JVP needs the same math path as the T4 reference).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.func import jvp

from chaos_llm.big_engine import full_embeddings, load_big, save_json, set_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokens", default="0,16,50,100,150,200,250,299")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(os.path.join(args.run, "a1.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    pids = torch.as_tensor(d["prompt_ids"])[None].cuda()
    gids = torch.as_tensor(d["gen_ids"])[None].cuda()
    P = int(meta["prompt_len"])
    pos_list = [int(x) for x in args.tokens.split(",") if int(x) < gids.shape[1]]

    model, tok, _mp, info = load_big(args.model, dtype=args.dtype, attn="eager")
    set_window(model, meta.get("window") or 100)
    nL = info["num_layers"]
    E = full_embeddings(model, pids, gids)
    L = E.shape[1]
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, L), dtype=torch.long, device="cuda"),
                  output_hidden_states=True, use_cache=False)
    H0 = o.hidden_states[0]
    D = H0.shape[-1]
    pos_ids = torch.arange(L, device="cuda")[None]
    pe = model.model.rotary_emb(H0, pos_ids)
    mask = model.model._update_causal_mask(torch.ones((1, L), dtype=torch.long, device="cuda"), H0,
                                           torch.arange(L, device="cuda"), None, output_attentions=False)
    k = args.k
    lam = np.zeros((len(pos_list), nL, k), dtype=np.float32)
    local = np.zeros((len(pos_list), nL), dtype=np.float32)
    t0 = time.time()
    for ti, gen_t in enumerate(pos_list):
        t = P + gen_t
        g = torch.Generator(device="cuda").manual_seed(1234 + ti)
        Q = torch.linalg.qr(torch.randn(D, k, device="cuda", generator=g))[0]
        cum = torch.zeros(k, device="cuda")
        for l in range(nL):
            layer = model.model.layers[l]

            def f(Hin, layer=layer):
                return layer(Hin, attention_mask=mask, position_ids=pos_ids,
                             position_embeddings=pe, use_cache=False)[0]
            Us = []
            for i in range(k):
                V = torch.zeros_like(o.hidden_states[l])
                V[0, t] = Q[:, i]
                with torch.no_grad():
                    _, dy = jvp(f, (o.hidden_states[l],), (V,))
                Us.append(dy[0, t])
            U = torch.stack(Us, dim=1)  # [D,k]
            local[ti, l] = U.norm(dim=0).mean().item()
            Q, R = torch.linalg.qr(U)
            cum += torch.log(torch.abs(torch.diag(R)) + 1e-30)
            lam[ti, l] = (cum / (l + 1)).detach().float().cpu().numpy()
        print(f'token {gen_t}: final lambda {lam[ti,-1].round(3).tolist()} '
              f'h_depth {float(np.maximum(lam[ti,-1],0).sum()):.3f} t={time.time()-t0:.0f}s', flush=True)
        del Us, U
        torch.cuda.empty_cache()

    h_depth = np.maximum(lam[:, -1, :], 0).sum(axis=1)
    np.savez(os.path.join(args.out, "depth.npz"), lam=lam, local_gain=local, h_depth=h_depth,
             tokens=np.asarray(pos_list), layers=np.arange(nL), k=k)
    save_json(os.path.join(args.out, "meta.json"),
              dict(run=args.run, tokens=pos_list, k=k, n_layers=nL, hidden_dim=D,
                   model=args.model, dtype=args.dtype, window=meta.get("window"),
                   elapsed_sec=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
