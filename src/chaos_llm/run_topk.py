"""Top-k singular subspaces of each block Jacobian, and how the perturbation sits in them.

For every saved layer l and probe token t we compute, at the reference state,

    J_l = d(block_l)/d(h_l)             (the full block, residual included)
    U_l = top-k LEFT singular vectors   (subspace iteration, JVP only)
    sigma_1..k

and two families of angles:

1. rotation between consecutive blocks
       theta_k(l) = largest principal angle between span(U_l[:k]) and span(U_{l+1}[:k])
   for k = 1, 3, 10.  k=1 is the old "top singular vector rotation"; k>1 fixes the
   degeneracy problem (two near-equal singular values swapping gives a fake 90 deg).

2. alignment of the *actual perturbation* with the subspace
       a_k(l) = || P_{U_l[:k]} delta h_l || / || delta h_l ||
   i.e. the fraction of the perturbation that lives where this block can amplify.
   This is the quantity that matters for propagation: not whether the *state* is
   aligned with the top singular vector, but whether the *perturbation* is.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.func import jvp

from chaos_llm.protocol import full_embeddings, get_layers, load_model, save_json, set_window


def block_kwargs(model, E, L):
    blocks = get_layers(model)
    caps = {}
    handles = []

    def mk(i):
        def hook(module, a, kw):
            caps[i] = kw
        return hook
    for i, b in enumerate(blocks):
        handles.append(b.register_forward_pre_hook(mk(i), with_kwargs=True))
    with torch.inference_mode():
        model(inputs_embeds=E, attention_mask=torch.ones((1, L), dtype=torch.long, device="cuda"),
              output_hidden_states=False, use_cache=False)
    for h in handles:
        h.remove()
    return blocks, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokens", default="16,50,123")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, "a1.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    pids = torch.as_tensor(d["prompt_ids"]).long()[None].cuda()
    gids = torch.as_tensor(d["gen_ids"]).long()[None].cuda()
    P = int(meta["prompt_len"])
    toks = [int(x) for x in args.tokens.split(",") if int(x) < gids.shape[1]]

    model, tok, _ = load_model(args.model, attn="eager", dtype=args.dtype)
    set_window(model, meta["window"])
    E = full_embeddings(model, pids, gids)
    L = int(E.shape[1])
    blocks, caps = block_kwargs(model, E, L)
    nL = len(blocks)
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, L), dtype=torch.long, device="cuda"),
                  output_hidden_states=True, use_cache=False)
    H = [h.detach() for h in o.hidden_states[:nL + 1]]
    D = int(H[0].shape[-1])
    del o
    torch.cuda.empty_cache()

    # ---- perturbed forwards (same sparse-simplex family as run_a1)
    rng = np.random.default_rng(7)
    K = args.dirs
    flat_dim = P * D
    idx = rng.choice(flat_dim, size=(K, 20), replace=False)
    sgn = rng.choice([-1.0, 1.0], size=(K, 20))
    Eb = E[0][None].expand(K, -1, -1).clone()
    for k in range(K):
        e = Eb[k].reshape(-1)
        for j in range(20):
            e[idx[k, j]] += args.r * sgn[k, j] / np.sqrt(20)
    with torch.inference_mode():
        op = model(inputs_embeds=Eb, attention_mask=torch.ones((K, L), dtype=torch.long, device="cuda"),
                   output_hidden_states=True, use_cache=False)
    Hp = [h.detach() for h in op.hidden_states[:nL + 1]]
    del op
    torch.cuda.empty_cache()

    k = args.k
    sig = np.zeros((len(toks), nL, k), np.float32)
    align = np.zeros((len(toks), nL, K, k), np.float32)   # fraction captured by top-k
    theta = np.zeros((len(toks), max(nL - 1, 0), 3), np.float32)  # for kk = 1,3,10
    t0 = time.time()

    for ti, gt in enumerate(toks):
        t = P + gt
        g = torch.Generator(device="cuda").manual_seed(4242 + ti)
        bases = []
        for l in range(nL):
            layer = blocks[l]
            kw = caps[l]
            Hl = H[l]

            def f(Hin, layer=layer, kw=kw):
                out = layer(Hin, **kw)
                return out[0] if isinstance(out, tuple) else out

            Q = torch.linalg.qr(torch.randn(D, k, device="cuda", generator=g))[0]
            for _ in range(args.iters):
                Z = []
                for i in range(k):
                    V = torch.zeros_like(Hl)
                    V[0, t] = Q[:, i].to(Hl.dtype)
                    with torch.no_grad():
                        _, dy = jvp(f, (Hl,), (V,))
                    v = dy[0, t] if dy.dim() == 3 else dy[t]
                    Z.append(v.float())
                Z = torch.stack(Z, dim=1)
                Q, R = torch.linalg.qr(Z)
                dg = torch.abs(torch.diag(R))
            sig[ti, l] = dg.detach().cpu().numpy()[:k]
            Qc = Q.detach()                      # (D,k) orthonormal basis
            bases.append(Qc)
            # alignment of every perturbation direction with the top-k subspace
            for dd in range(K):
                delta = (Hp[l][dd, t] - H[l][0, t]).float()
                nrm = delta.norm()
                if nrm <= 0:
                    continue
                for kk in (1, 3, 10):
                    if kk > k:
                        continue
                    B = Qc[:, :kk]
                    proj = B @ (B.T @ delta)
                    align[ti, l, dd, kk - 1] = float((proj.norm() / (nrm + 1e-30)).item())
        # principal angles between consecutive layers' subspaces
        for l in range(nL - 1):
            A_ = bases[l]
            B_ = bases[l + 1]
            for j, kk in enumerate((1, 3, 10)):
                if kk > k:
                    continue
                C = (A_[:, :kk].T @ B_[:, :kk]).cpu().numpy()
                sv = np.linalg.svd(C, compute_uv=False)
                theta[ti, l, j] = float(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))
        print(f"token {gt} done ({time.time() - t0:.0f}s)", flush=True)

    np.savez(os.path.join(args.out, "topk.npz"),
             sig=sig, align=align, theta=theta, layers=np.arange(nL), tokens=np.asarray(toks),
             k=k, dirs=K, r=args.r)
    save_json(os.path.join(args.out, "meta.json"),
              dict(model=args.model, run=args.run, tokens=toks, k=k, iters=args.iters,
                   dirs=K, r=args.r, elapsed_sec=round(time.time() - t0, 1)))
    print("saved", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
