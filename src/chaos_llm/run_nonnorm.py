"""Non-normality: sigma_max, spectral radius rho, kappa, principal angles."""
import argparse, json, os, time
import numpy as np, torch
from torch.func import jvp, vjp
from scipy.sparse.linalg import LinearOperator, eigs
from chaos_llm.protocol import load_model, set_window, full_embeddings, save_json, get_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokens", default="16,50,123,143,200")
    ap.add_argument("--layers", default="0,7,14,21,27")
    ap.add_argument("--full-layers", default="0,14,27")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(os.path.join(args.run, "a1.npz")); meta = json.load(open(os.path.join(args.run, "meta.json")))
    pids = torch.tensor(d["prompt_ids"])[None].cuda(); gids = torch.tensor(d["gen_ids"])[None].cuda()
    P = int(meta["prompt_len"]); toks = [int(x) for x in args.tokens.split(",")]
    sel_layers = [int(x) for x in args.layers.split(",")]
    full_layers = [int(x) for x in args.full_layers.split(",")]
    model, tok, _ = load_model(args.model, attn="eager"); set_window(model, meta["window"])
    E = full_embeddings(model, pids, gids); L = E.shape[1]
    captured = {}
    handles = []
    blocks = get_layers(model)
    for l in range(len(blocks)):
        def hook(mod, inp, out, l=l):
            captured[l] = inp[0].detach()
        handles.append(blocks[l].mlp.register_forward_hook(hook))
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, L), dtype=torch.long, device="cuda"),
                  output_hidden_states=True, use_cache=False)
    for h in handles: h.remove()
    H0 = o.hidden_states[0]; D = H0.shape[-1]; nL = len(o.hidden_states) - 1
    pos_ids = torch.arange(L, device="cuda")[None]
    pe = model.model.rotary_emb(H0, pos_ids)
    mask = model.model._update_causal_mask(torch.ones((1, L), dtype=torch.long, device="cuda"), H0,
                                           torch.arange(L, device="cuda"), None, output_attentions=False)

    def mlp_metrics(l, t):
        mlp = blocks[l].mlp; x = captured[l][0, P + t].clone()
        with torch.no_grad():
            gate = mlp.gate_proj(x); up = mlp.up_proj(x); s = mlp.act_fn(gate)
            sig = torch.sigmoid(gate); ds = sig * (1 + gate * (1 - sig))
            J = (mlp.down_proj.weight @ (s[:, None] * mlp.up_proj.weight + (up * ds)[:, None] * mlp.gate_proj.weight)).detach().cpu()
            # sigma_max via power iteration on J^T J
            v = torch.randn(D); v /= v.norm()
            for _ in range(20):
                u = J @ v; w = J.T @ u; v = w / (w.norm() + 1e-30)
            smax = float((J @ v).norm())
            A = LinearOperator((D, D),
                               matvec=lambda z: (J @ torch.as_tensor(z, dtype=torch.float32)).numpy(),
                               rmatvec=lambda z: (J.T @ torch.as_tensor(z, dtype=torch.float32)).numpy(),
                               dtype=np.float32)
            ev = eigs(A, k=1, which="LM", maxiter=40, tol=1e-3, return_eigenvectors=False)
            rho = float(abs(ev[0]))
        return smax, rho

    def funcs(l, t):
        layer = blocks[l]; H = o.hidden_states[l].clone()
        def f(Hin):
            return layer(Hin, attention_mask=mask, position_ids=pos_ids, position_embeddings=pe, use_cache=False)[0]
        def jv(v):
            V = torch.zeros_like(H); V[0, P + t] = v
            with torch.no_grad(): _, dy = jvp(f, (H,), (V,))
            return dy[0, P + t]
        def jtv(u):
            U = torch.zeros_like(H); U[0, P + t] = u
            _, vjp_fn = vjp(f, H)
            (V,) = vjp_fn(U)
            return V[0, P + t]
        return jv, jtv

    def power(l, t, iters=10):
        jv, jtv = funcs(l, t); v = torch.randn(D, device="cuda"); v /= v.norm()
        for _ in range(iters):
            u = jv(v); w = jtv(u); v = w / (w.norm() + 1e-30)
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
    mlp_s = np.zeros((len(toks), nL), np.float32); mlp_r = np.zeros_like(mlp_s); mlp_k = np.zeros_like(mlp_s)
    for ti, t in enumerate(toks):
        for l in sel_layers:
            s, r = mlp_metrics(l, t); mlp_s[ti, l] = s; mlp_r[ti, l] = r; mlp_k[ti, l] = s / max(r, 1e-30)
        print("MLP token", t, "done", flush=True)

    fs = np.full((len(toks), nL), np.nan, np.float32); fr = np.full_like(fs, np.nan); fk = np.full_like(fs, np.nan)
    for ti, t in enumerate(toks):
        for l in full_layers:
            s, _ = power(l, t)
            try:
                r = rho_full(l, t)
            except Exception:
                r = float("nan")
            fs[ti, l] = s; fr[ti, l] = r; fk[ti, l] = s / max(r, 1e-30) if np.isfinite(r) else float("nan")
            print(f"full token {t} layer {l}: sigma {s:.3f} rho {r:.3f} kappa {s/max(r,1e-30):.3f}", flush=True)

    # principal angles between top singular vectors of consecutive layers
    angles = np.full((len(toks), nL - 1), np.nan, np.float32)
    for ti, t in enumerate(toks):
        vecs = []
        for l in range(nL):
            _, v = power(l, t, iters=8); vecs.append(v.detach().float().cpu().numpy())
        for l in range(nL - 1):
            c = abs(float(np.dot(vecs[l], vecs[l + 1]))); c = min(1.0, max(-1.0, c))
            angles[ti, l] = np.degrees(np.arccos(c))
        print("angles token", t, "median", float(np.nanmedian(angles[ti])), flush=True)

    np.savez(os.path.join(args.out, "nonnorm.npz"), mlp_smax=mlp_s, mlp_rho=mlp_r, mlp_kappa=mlp_k,
             full_smax=fs, full_rho=fr, full_kappa=fk, angles=angles, tokens=np.asarray(toks),
             layers=np.arange(nL), sel_layers=np.asarray(sel_layers))
    save_json(os.path.join(args.out, "meta.json"),
              dict(run=args.run, tokens=toks, layers=sel_layers, n_layers=nL,
                   elapsed_sec=round(time.time() - t0, 1), model=args.model))
    print("saved", args.out, "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
