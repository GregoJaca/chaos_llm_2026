"""Linearity / superposition test for the transport kernel.

Perturbation: the SAME random unit vector u (length d) added to every prompt position.

We compare three things at a fixed layer:
  A[t] = response when u is added to ALL prompt positions at once,
  S[t] = sum over p of the response when u is added to position p alone,
  E[t] = ||A - S|| / ||A||   (relative superposition error).

If the response is linear in the perturbation, A == S and E = 0.
"""
import argparse
import json
import os

import numpy as np
import torch

from chaos_llm.protocol import full_embeddings, load_model, save_json, set_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--layers", default="1,7,14,last")
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
    L = int(model._chaos_info["num_layers"])
    layers = [L - 1 if x == "last" else int(x) for x in args.layers.split(",")]
    D = int(model._chaos_info["hidden_size"])

    E0 = full_embeddings(model, pids, gids)
    mask = torch.ones((1, E0.shape[1]), dtype=torch.long, device="cuda")

    def fwd(E):
        with torch.inference_mode():
            o = model(inputs_embeds=E, attention_mask=mask, output_hidden_states=True, use_cache=False)
        return {l: o.hidden_states[l][0, P:P + T].detach() for l in layers}

    h0 = fwd(E0)

    g = torch.Generator(device="cuda").manual_seed(2024)
    res = {l: dict(err=np.zeros((args.trials, T)), cos=np.zeros((args.trials, T)),
                   nA=np.zeros((args.trials, T)), nS=np.zeros((args.trials, T))) for l in layers}
    for k in range(args.trials):
        u = torch.randn(D, device="cuda", generator=g)
        u = u / u.norm()
        du = (args.r * u).to(E0.dtype)

        E = E0.clone(); E[0, :P] += du
        hA = fwd(E)
        S = {l: torch.zeros_like(h0[l]) for l in layers}
        for p in range(P):
            E = E0.clone(); E[0, p] += du
            hp = fwd(E)
            for l in layers:
                S[l] += hp[l] - h0[l]
            del hp
        for l in layers:
            A = hA[l] - h0[l]
            nA = A.norm(dim=-1); nS = S[l].norm(dim=-1)
            diff = (A - S[l]).norm(dim=-1)
            res[l]["err"][k] = (diff / torch.clamp(nA, min=1e-12)).cpu().numpy()
            res[l]["cos"][k] = torch.nn.functional.cosine_similarity(A, S[l], dim=-1).cpu().numpy()
            res[l]["nA"][k] = nA.cpu().numpy(); res[l]["nS"][k] = nS.cpu().numpy()
        del hA, S
        torch.cuda.empty_cache()
        print(f"trial {k}: err(last layer) median {np.median(res[layers[-1]]['err'][k]):.4f} "
              f"cos median {np.median(res[layers[-1]]['cos'][k]):.5f}", flush=True)

    out = {}
    for l in layers:
        out[f"layer{l}"] = {k: v for k, v in res[l].items()}
    np.savez(os.path.join(args.out, "superposition.npz"), layers=np.asarray(layers), r=args.r,
             trials=args.trials, prompt_len=P, T=T,
             **{f"err_L{l}": res[l]["err"] for l in layers},
             **{f"cos_L{l}": res[l]["cos"] for l in layers},
             **{f"nA_L{l}": res[l]["nA"] for l in layers},
             **{f"nS_L{l}": res[l]["nS"] for l in layers})
    save_json(os.path.join(args.out, "meta.json"),
              dict(model=args.model, run=args.run, trials=args.trials, r=args.r,
                   layers=layers, prompt_len=P, T=T,
                   err_median={int(l): float(np.median(res[l]["err"])) for l in layers}))
    print("saved", args.out)


if __name__ == "__main__":
    main()
