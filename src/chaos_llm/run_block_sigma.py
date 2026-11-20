"""Top singular value of the FULL block Jacobian at one position, by power iteration (generic)."""
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
    ap.add_argument("--tokens", default="16,50,100")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, "a1.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    pids = torch.as_tensor(d["prompt_ids"]).long()[None].cuda()
    gids = torch.as_tensor(d["gen_ids"]).long()[None].cuda()
    P = int(meta["prompt_len"])
    pos_list = [int(x) for x in args.tokens.split(",") if int(x) < gids.shape[1]]

    model, tok, _ = load_model(args.model, attn="eager", dtype=args.dtype)
    set_window(model, meta["window"])
    blocks = list(model.model.layers)
    nL = len(blocks)
    E = full_embeddings(model, pids, gids)
    L = int(E.shape[1])
    with torch.inference_mode():
        o = model(inputs_embeds=E,
                  attention_mask=torch.ones((1, L), dtype=torch.long, device="cuda"),
                  output_hidden_states=True, use_cache=False)
    H = [h.detach() for h in o.hidden_states[:nL + 1]]
    D = int(H[0].shape[-1])
    del o
    torch.cuda.empty_cache()

    caps = {}
    handles = []

    def mk(i):
        def hook(module, a, kw):
            caps[i] = kw
        return hook

    for i, b in enumerate(blocks):
        handles.append(b.register_forward_pre_hook(mk(i), with_kwargs=True))
    with torch.inference_mode():
        model(inputs_embeds=E,
              attention_mask=torch.ones((1, L), dtype=torch.long, device="cuda"),
              output_hidden_states=False, use_cache=False)
    for h in handles:
        h.remove()

    t0 = time.time()
    smax = np.zeros((len(pos_list), nL), dtype=np.float32)
    smax_iter = np.zeros((len(pos_list), nL, args.iters), dtype=np.float32)
    for ti, gen_t in enumerate(pos_list):
        t = P + gen_t
        for l in range(nL):
            layer, kw, Hl = blocks[l], caps[l], H[l]

            def f(Hin, layer=layer, kw=kw):
                out = layer(Hin, **kw)
                return out[0] if isinstance(out, tuple) else out

            g = torch.Generator(device="cuda").manual_seed(77 + ti * 100 + l)
            v = torch.randn(D, device="cuda", generator=g)
            v = v / v.norm()
            for it in range(args.iters):
                V = torch.zeros_like(Hl); V[0, t] = v.to(Hl.dtype)
                with torch.no_grad():
                    _, u = torch.func.jvp(f, (Hl,), (V,))
                u = (u[0, t] if u.dim() == 3 else u[t]).float()
                # J^T u : vjp with a cotangent that is zero except at position t
                _, vjp_fn = torch.func.vjp(f, Hl)
                cot = torch.zeros_like(Hl); cot[0, t] = u.to(Hl.dtype)
                with torch.no_grad():
                    (w,) = vjp_fn(cot)
                w = w[0, t].float()
                nv = w.norm()
                v = w / (nv + 1e-30)
                smax_iter[ti, l, it] = float(nv.sqrt())
            smax[ti, l] = float(nv.sqrt())
        print(f"token {gen_t}: sigma1 profile {np.round(smax[ti][::5], 2).tolist()} "
              f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(os.path.join(args.out, "block_sigma.npz"), smax=smax, smax_iter=smax_iter,
             tokens=np.asarray(pos_list), layers=np.arange(nL), hidden=D)
    save_json(os.path.join(args.out, "meta.json"),
              dict(model=args.model, run=args.run, tokens=pos_list, iters=args.iters,
                   elapsed=round(time.time() - t0, 1)))
    print("saved", args.out, round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
