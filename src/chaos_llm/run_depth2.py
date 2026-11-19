"""Depth tangent spectrum, generic across decoder architectures.

Propagates k orthonormal tangent directions through every block *at one position* and records
the cumulative finite-time depth exponents  lam_l = log sigma_i(product through block l) / (l+1).

The only model-specific object is the block itself; its keyword arguments (attention mask,
position embeddings, ...) are captured from a normal forward pass with a pre-hook, so the same
code runs on Qwen2/Qwen3/Phi/Qwen3.5 without touching private APIs.
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
    ap.add_argument("--tokens", default="0,16,50,100,123,143,200,250")
    ap.add_argument("--k", type=int, default=8)
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

    # capture the exact kwargs each block receives
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
    print(f"captured kwargs for {len(caps)} blocks: "
          f"{sorted(caps[0].keys())}", flush=True)

    k = args.k
    lam = np.zeros((len(pos_list), nL, k), dtype=np.float32)
    local = np.zeros((len(pos_list), nL), dtype=np.float32)
    t0 = time.time()
    for ti, gen_t in enumerate(pos_list):
        t = P + gen_t
        g = torch.Generator(device="cuda").manual_seed(1234 + ti)
        Q = torch.linalg.qr(torch.randn(D, k, device="cuda", generator=g).float())[0]
        cum = torch.zeros(k, device="cuda")
        for l in range(nL):
            layer = blocks[l]
            kw = caps[l]
            Hl = H[l]
            def f(Hin, layer=layer, kw=kw):
                out = layer(Hin, **kw)
                return out[0] if isinstance(out, tuple) else out
            Us = []
            for i in range(k):
                V = torch.zeros_like(Hl)
                V[0, t] = Q[:, i]
                with torch.no_grad():
                    _, dy = torch.func.jvp(f, (Hl,), (V,))
                v = dy[0, t] if dy.dim() == 3 else dy[t]
                Us.append(v)
            U = torch.stack(Us, dim=1)
            local[ti, l] = U.norm(dim=0).mean().item()
            Q, R = torch.linalg.qr(U)
            cum = cum + torch.log(torch.abs(torch.diag(R)) + 1e-30)
            lam[ti, l] = (cum / (l + 1)).detach().float().cpu().numpy()
            del U, Us
        print(f"token {gen_t}: lambda1 final {float(lam[ti, -1, 0]):.3f} "
              f"h_depth {float(np.maximum(lam[ti, -1], 0).sum()):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    h_depth = np.maximum(lam[:, -1, :], 0).sum(axis=1)
    np.savez(os.path.join(args.out, "depth.npz"), lam=lam, local_gain=local, h_depth=h_depth,
             tokens=np.asarray(pos_list), layers=np.arange(nL), k=k)
    save_json(os.path.join(args.out, "meta.json"),
              dict(run=args.run, tokens=pos_list, k=k, n_layers=nL, hidden_dim=D,
                   elapsed_sec=round(time.time() - t0, 1), model=args.model))
    print("saved", args.out, "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
