"""Non-normality for large models (port of run_nonnorm.py).

sigma_max, spectral radius rho, kappa = sigma_max/rho for MLP blocks and full
blocks, plus principal angles between the top singular directions of
consecutive layers. Same output schema as run_nonnorm.py.

Differences: big_engine loader; explicit causal/sliding mask (transformers
>=4.5x removed _update_causal_mask); cache_position passed to the layer.
MLP Jacobian is the same analytic SwiGLU formula:
    J(x) = W_down @ ( s[:,None] W_up + (up * ds)[:,None] W_gate )
with s = act(gate), ds = d act/d gate at x, evaluated at the captured input.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.func import jvp, vjp
from scipy.sparse.linalg import LinearOperator, eigs

from chaos_llm.big_engine import full_embeddings, load_big, save_json, set_window


def build_mask(L, window, device, dtype):
    idx = torch.arange(L, device=device)
    allowed = idx[None, :] <= idx[:, None]
    if window:
        allowed = allowed & (idx[None, :] > idx[:, None] - int(window))
    m = torch.zeros((L, L), device=device, dtype=dtype)
    m = m.masked_fill(~allowed, torch.finfo(dtype).min)
    return m[None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokens", default="21,50,100,189,273")
    ap.add_argument("--layers", default="")
    ap.add_argument("--full-layers", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(os.path.join(args.run, "a1.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    pids = torch.as_tensor(d["prompt_ids"])[None].cuda()
    gids = torch.as_tensor(d["gen_ids"])[None].cuda()
    P = int(meta["prompt_len"])
    toks = [int(x) for x in args.tokens.split(",")]
    model, tok, _mp, info = load_big(args.model, dtype=args.dtype, attn="eager")
    active = set_window(model, meta.get("window") or 0)
    nL = info["num_layers"]
    D = info["hidden_size"]
    sel_layers = ([int(x) for x in args.layers.split(",")] if args.layers
                  else sorted(set(list(range(0, nL, 6)) + [nL - 1])))
    full_layers = ([int(x) for x in args.full_layers.split(",")] if args.full_layers
                   else sorted(set([0, nL // 4, nL // 2, 3 * nL // 4, nL - 1])))
    E = full_embeddings(model, pids, gids)
    L = E.shape[1]

    captured = {}
    handles = []
    for l in range(nL):
        def hook(mod, inp, out, l=l):
            captured[l] = inp[0].detach()
        handles.append(model.model.layers[l].mlp.register_forward_hook(hook))
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, L), dtype=torch.long, device="cuda"),
                  output_hidden_states=True, use_cache=False)
    for h in handles:
        h.remove()
    H0 = o.hidden_states[0]
    pos_ids = torch.arange(L, device="cuda")[None]
    pe = model.model.rotary_emb(H0, pos_ids)
    mask = build_mask(L, meta.get("window") if active else 0, "cuda", torch.float32)
    cache_position = torch.arange(L, device="cuda")

    def mlp_metrics(l, t):
        mlp = model.model.layers[l].mlp
        x = captured[l][0, P + t].clone()
        with torch.no_grad():
            if hasattr(mlp, "gate_proj"):
                gate = mlp.gate_proj(x)
                up = mlp.up_proj(x)
                Wg, Wu = mlp.gate_proj.weight, mlp.up_proj.weight
            else:
                # fused gate_up_proj layout (Phi3MLP and similar)
                gu = mlp.gate_up_proj(x)
                gate, up = gu.chunk(2, dim=-1)
                W = mlp.gate_up_proj.weight
                Fh = W.shape[0] // 2
                Wg, Wu = W[:Fh], W[Fh:]
            act = getattr(mlp, "act_fn", None) or getattr(mlp, "activation_fn")
            s = act(gate)
            sig = torch.sigmoid(gate)
            ds = sig * (1 + gate * (1 - sig))
            J = (mlp.down_proj.weight @ (s[:, None] * Wu + (up * ds)[:, None] * Wg)).detach()
            v = torch.randn(D, device=J.device)
            v /= v.norm()
            for _ in range(20):
                u = J @ v
                w = J.T @ u
                v = w / (w.norm() + 1e-30)
            smax = float((J @ v).norm())
            Jc = J.float().cpu()
            A = LinearOperator((D, D),
                               matvec=lambda z: (Jc @ torch.as_tensor(z, dtype=torch.float32)).numpy(),
                               rmatvec=lambda z: (Jc.T @ torch.as_tensor(z, dtype=torch.float32)).numpy(),
                               dtype=np.float32)
            ev = eigs(A, k=1, which="LM", maxiter=40, tol=1e-3, return_eigenvectors=False)
            rho = float(abs(ev[0]))
        return smax, rho

    def funcs(l, t):
        layer = model.model.layers[l]
        H = o.hidden_states[l].clone()

        def layer_out(res):
            return res[0] if isinstance(res, tuple) else res

        def f(Hin):
            return layer_out(layer(Hin, attention_mask=mask, position_ids=pos_ids,
                                   position_embeddings=pe, use_cache=False,
                                   cache_position=cache_position))

        def jv(v):
            V = torch.zeros_like(H)
            V[0, P + t] = v
            with torch.no_grad():
                _, dy = jvp(f, (H,), (V,))
            return dy[0, P + t]

        def jtv(u):
            U = torch.zeros_like(H)
            U[0, P + t] = u
            _, vjp_fn = vjp(f, H)
            (V,) = vjp_fn(U)
            return V[0, P + t]
        return jv, jtv

    def power(l, t, iters=10):
        jv, jtv = funcs(l, t)
        v = torch.randn(D, device="cuda")
        v /= v.norm()
        for _ in range(iters):
            u = jv(v)
            w = jtv(u)
            v = w / (w.norm() + 1e-30)
        return float(jv(v).norm()), v

    def rho_full(l, t):
        jv, jtv = funcs(l, t)
        A = LinearOperator((D, D),
                           matvec=lambda x: jv(torch.as_tensor(x, device="cuda", dtype=torch.float32)).float().cpu().numpy(),
                           rmatvec=lambda x: jtv(torch.as_tensor(x, device="cuda", dtype=torch.float32)).float().cpu().numpy(),
                           dtype=np.float32)
        ev = eigs(A, k=1, which="LM", maxiter=25, tol=1e-3, return_eigenvectors=False)
        return float(abs(ev[0]))

    t0 = time.time()
    mlp_s = np.zeros((len(toks), nL), np.float32)
    mlp_r = np.zeros_like(mlp_s)
    mlp_k = np.zeros_like(mlp_s)
    for ti, t in enumerate(toks):
        for l in sel_layers:
            s, r = mlp_metrics(l, t)
            mlp_s[ti, l] = s
            mlp_r[ti, l] = r
            mlp_k[ti, l] = s / max(r, 1e-30)
        print("MLP token", t, "done", flush=True)

    fs = np.full((len(toks), nL), np.nan, np.float32)
    fr = np.full_like(fs, np.nan)
    fk = np.full_like(fs, np.nan)
    for ti, t in enumerate(toks):
        for l in full_layers:
            s, _ = power(l, t)
            r = rho_full(l, t)
            fs[ti, l] = s
            fr[ti, l] = r
            fk[ti, l] = s / max(r, 1e-30)
            print(f"full token {t} layer {l}: sigma {s:.3f} rho {r:.3f} kappa {s/max(r,1e-30):.3f}", flush=True)

    # principal angles between top singular vectors of consecutive layers
    # (subsample layers if the network is deep: every layer is one power sweep)
    angle_layers = sorted(set(list(range(0, nL, 4)) + [nL - 1]))
    angles = np.full((len(toks), len(angle_layers) - 1), np.nan, np.float32)
    for ti, t in enumerate(toks):
        vecs = []
        for l in angle_layers:
            _, v = power(l, t, iters=8)
            vecs.append(v.detach().float().cpu().numpy())
        for i in range(len(angle_layers) - 1):
            c = abs(float(np.dot(vecs[i], vecs[i + 1])))
            c = min(1.0, max(-1.0, c))
            angles[ti, i] = np.degrees(np.arccos(c))
        print("angles token", t, "median", float(np.nanmedian(angles[ti])), flush=True)

    np.savez(os.path.join(args.out, "nonnorm.npz"), mlp_smax=mlp_s, mlp_rho=mlp_r, mlp_kappa=mlp_k,
             full_smax=fs, full_rho=fr, full_kappa=fk, angles=angles,
             angle_layers=np.asarray(angle_layers), tokens=np.asarray(toks),
             layers=np.arange(nL), sel_layers=np.asarray(sel_layers))
    save_json(os.path.join(args.out, "meta.json"),
             dict(run=args.run, tokens=toks, layers=sel_layers, full_layers=full_layers,
                  n_layers=nL, model=args.model, dtype=args.dtype,
                  window=meta.get("window"), elapsed_sec=round(time.time() - t0, 1)))
    print("saved", args.out, "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
