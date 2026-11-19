"""Dimensionality dataset for one model.

Produces, in --out:
  vocab.npz   : Gram matrices of the input embedding and the unembedding (D x D),
                singular values, row norms, a 1024-row sample of each matrix.
  acts.npz    : hidden states at generated positions, all layers, 3 prompts x 400 tokens
                (float16), plus token ids and per-layer norms.
  logits.npz  : logits for 128 positions of prompt 1 (float16), with token ids.

Everything else (spectra, intrinsic dimension, anisotropy) is computed offline on CPU.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from chaos_llm.protocol import full_embeddings, generate, load_model, prompt_ids, set_window

PROMPTS = [
    "Explain how ocean tides are produced by the Moon and the Sun.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Solve: a train travels 240 km in 3 hours, then 180 km in 2 hours. What is its average speed?",
]
LAYER_CHUNK = 8192          # rows per Gram accumulation


def gram(W):
    """D x D Gram matrix of a possibly huge [V, D] matrix, accumulated in chunks."""
    D = W.shape[1]
    G = torch.zeros((D, D), device=W.device, dtype=torch.float32)
    for i in range(0, W.shape[0], LAYER_CHUNK):
        c = W[i:i + LAYER_CHUNK].float()
        G += c.T @ c
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--window", type=int, default=4096)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    model, tok, path = load_model(args.model, dtype=args.dtype)
    info = model._chaos_info
    L = info["num_layers"]
    D = info["hidden_size"]
    V = info["vocab_size"]
    print(f"[{args.tag}] loaded {L} layers, D={D}, V={V}, dtype={next(model.parameters()).dtype}", flush=True)

    # ---------------- vocabulary matrices ----------------
    E = model.get_input_embeddings().weight.detach()
    W_U = model.get_output_embeddings().weight.detach()
    tied = E.data_ptr() == W_U.data_ptr()
    print(f"[{args.tag}] embedding {tuple(E.shape)} unembedding {tuple(W_U.shape)} tied={tied}", flush=True)

    GE = gram(E).double().cpu().numpy()
    GW = GE if tied else gram(W_U).double().cpu().numpy()
    sE = np.sqrt(np.maximum(np.linalg.eigvalsh(GE)[::-1], 0))
    sW = sE if tied else np.sqrt(np.maximum(np.linalg.eigvalsh(GW)[::-1], 0))
    step = max(1, E.shape[0] // 1024)
    sample_E = E[::step][:1024].float().cpu().numpy().astype(np.float16)
    sample_W = sample_E if tied else W_U[::step][:1024].float().cpu().numpy().astype(np.float16)
    np.savez_compressed(
        os.path.join(args.out, "vocab.npz"),
        GE=GE.astype(np.float32), GW=GW.astype(np.float32),
        sE=sE.astype(np.float32), sW=sW.astype(np.float32),
        normE=E.norm(dim=1).float().cpu().numpy(),
        normW=E.norm(dim=1).float().cpu().numpy() if tied else W_U.norm(dim=1).float().cpu().numpy(),
        sample_E=sample_E, sample_W=sample_W,
        tied=np.array([tied]), D=np.array([D]), V=np.array([V]), L=np.array([L]),
    )
    print(f"[{args.tag}] vocab saved sE[:3]={sE[:3]} t={time.time()-t0:.0f}s", flush=True)
    del GE, GW
    torch.cuda.empty_cache()

    # ---------------- activations ----------------
    set_window(model, args.window)
    acts = []          # list over prompts of [L+1, T, D] float16
    ids_all = []
    norms = []
    logit_pack = None
    for pi, text in enumerate(PROMPTS):
        pids = prompt_ids(tok, text)
        P = int(pids.shape[1])
        with torch.inference_mode():
            emb = model.get_input_embeddings()(pids)
            gen = generate(model, emb, args.tokens)
        T = int(gen.shape[1])
        Efull = full_embeddings(model, pids, gen)
        with torch.inference_mode():
            o = model(inputs_embeds=Efull,
                      attention_mask=torch.ones((1, Efull.shape[1]), dtype=torch.long, device="cuda"),
                      output_hidden_states=True, use_cache=False)
        hs = o.hidden_states
        n = min(len(hs), L + 1)
        A = torch.stack([hs[l][0, P:P + T].float() for l in range(n)])       # [n, T, D]
        acts.append(A.to(torch.float16).detach().cpu().numpy())
        norms.append(A.norm(dim=-1).detach().cpu().numpy().astype(np.float32))
        ids_all.append(gen[0].detach().cpu().numpy().astype(np.int32))
        if pi == 0:
            K = min(128, T)
            h = hs[-1][0, P:P + K]
            z = model.lm_head(h).float()
            logit_pack = dict(logits=z.detach().to(torch.float16).cpu().numpy(),
                              ids=gen[0][:K].detach().cpu().numpy().astype(np.int32),
                              prompt_ids=pids[0].detach().cpu().numpy().astype(np.int32))
        del o, hs, Efull, A
        torch.cuda.empty_cache()
        print(f"[{args.tag}] prompt {pi}: P={P} T={T} t={time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(
        os.path.join(args.out, "acts.npz"),
        acts=np.stack(acts),                     # [n_prompts, L+1, T, D] fp16
        norms=np.stack(norms),                   # [n_prompts, L+1, T] fp32
        gen_ids=np.stack(ids_all),               # [n_prompts, T]
        layers=np.arange(acts[0].shape[0]), D=np.array([D]), L=np.array([L]),
    )
    np.savez_compressed(os.path.join(args.out, "logits.npz"), **logit_pack)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(dict(model=args.model, tag=args.tag, path=path, layers=L, hidden=D,
                       vocab=V, tied=bool(tied), tokens=args.tokens, window=args.window,
                       prompts=PROMPTS, dtype=str(next(model.parameters()).dtype)), f, indent=1)
    print(f"[{args.tag}] DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
