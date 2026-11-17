"""A1: prompt perturbation, teacher-forced response, margins, thresholds.

DeepSeek-1.5B, FP32. Saves arrays for A1/FSLE/threshold analyses.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from chaos_llm.protocol import (
    build_simplex, full_embeddings, generate, load_model, make_directions,
    prompt_ids, save_json, set_window, subspace_indices,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--num-directions", type=int, default=20)
    ap.add_argument("--radii", default="1e-6,2.15e-6,4.64e-6,1e-5,2.15e-5,4.64e-5,1e-4,2.15e-4,4.64e-4,1e-3,2.15e-3,4.64e-3,1e-2")
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--batch-dirs", type=int, default=4)
    ap.add_argument("--layers", default="0,7,14,21,27")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    text = args.prompt
    if text.startswith("@"):
        line = open(text[1:]).read().splitlines()[0]
        text = line.split("\t", 1)[1] if "\t" in line else line
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    layers = [int(x) for x in args.layers.split(",")]

    model, tok, model_path = load_model(args.model)
    set_window(model, args.window)

    pids = prompt_ids(tok, text)
    P = pids.shape[1]
    D = model.config.hidden_size
    with torch.inference_mode():
        base_prompt = model.get_input_embeddings()(pids)[0]

    t0 = time.time()
    # baseline: two identical rows in one batch, verify
    Eref = base_prompt[None].expand(2, -1, -1)
    gen = generate(model, Eref, args.max_new_tokens)
    assert torch.equal(gen[0], gen[1]), "r=0 duplicate rows differ"
    gen_ids = gen[0:1]
    T = gen_ids.shape[1]

    Efull = full_embeddings(model, pids, gen_ids)
    out0 = model(inputs_embeds=Efull, attention_mask=torch.ones((1, Efull.shape[1]), dtype=torch.long, device="cuda"),
                 output_hidden_states=True, use_cache=False)
    logits0 = out0.logits[0]  # [L,V]
    pos = torch.arange(P - 1, P + T - 1, device="cuda")
    z0 = logits0[pos]  # [T,V]
    top2 = torch.topk(z0, k=2, dim=-1)
    margins = (top2.values[:, 0] - top2.values[:, 1]).detach().cpu().numpy().astype(np.float32)
    top1 = top2.indices[:, 0]
    top2i = top2.indices[:, 1]
    h0 = {l: out0.hidden_states[l][0, P:P + T].detach() for l in layers}  # [T,D]
    norm_h0 = np.stack([h0[l].norm(dim=-1).cpu().numpy() for l in layers])  # [n_layers,T]
    del out0, logits0, z0, top2

    simplex, sub = make_directions(args.num_directions, P, D, args.seed)
    sub_t = torch.as_tensor(sub, device="cuda", dtype=torch.long)

    nr, K, nl = len(radii), args.num_directions, len(layers)
    dh_norm = np.zeros((nr, K, nl, T), dtype=np.float32)
    dm = np.zeros((nr, K, T), dtype=np.float32)
    cos = np.zeros((nr, K, nl, T), dtype=np.float32)
    dh_last = np.zeros((nr, min(3, K), T, D), dtype=np.float32)

    top1v = top1.view(1, T, 1)
    top2v = top2i.view(1, T, 1)
    m_t = torch.as_tensor(margins, device="cuda")

    for ri, r in enumerate(radii):
        for start in range(0, K, args.batch_dirs):
            idx = list(range(start, min(start + args.batch_dirs, K)))
            B = len(idx)
            E = Efull.expand(B, -1, -1).clone()
            for j, di in enumerate(idx):
                delta = torch.zeros(P * D, device="cuda", dtype=torch.float32)
                delta.index_copy_(0, sub_t, torch.as_tensor(simplex[di], device="cuda") * r)
                E[j, :P] += delta.view(P, D)
            out = model(inputs_embeds=E,
                        attention_mask=torch.ones((B, E.shape[1]), dtype=torch.long, device="cuda"),
                        output_hidden_states=True, use_cache=False)
            zb = out.logits[:, pos, :]  # [B,T,V]
            z1 = torch.gather(zb, 2, top1v.expand(B, -1, -1)).squeeze(-1)
            z2 = torch.gather(zb, 2, top2v.expand(B, -1, -1)).squeeze(-1)
            dm[ri, idx] = ((z1 - z2) - m_t[None, :]).detach().cpu().numpy()  # [B,T]
            del zb, z1, z2, out.logits
            for li, l in enumerate(layers):
                h = out.hidden_states[l][:, P:P + T, :]  # [B,T,D]
                diff = h - h0[l][None]
                dh_norm[ri, idx, li] = diff.norm(dim=-1).detach().cpu().numpy()
                cos[ri, idx, li] = torch.nn.functional.cosine_similarity(
                    h, h0[l][None].expand_as(h), dim=-1).detach().cpu().numpy()
            for j, di in enumerate(idx):
                if di < 3:
                    dh_last[ri, di] = (out.hidden_states[layers[-1]][j, P:P + T] - h0[layers[-1]]).detach().cpu().numpy()
            del out, E
            torch.cuda.empty_cache()

    # local thresholds and predicted first flip (linear response at each r)
    eps = 1e-30
    r_c = margins[None, None, :] / np.maximum(np.abs(dm) / np.asarray(radii)[:, None, None], eps)
    pred_T = np.full((nr, K), -1, dtype=np.int32)
    for ri, r in enumerate(radii):
        hit = r_c[ri] <= r
        anyhit = hit.any(axis=1)
        pred_T[ri, anyhit] = hit.argmax(axis=1)[anyhit]

    meta = dict(model=args.model, model_path=model_path, dtype="float32", window=args.window,
                prompt=text, prompt_len=int(P), max_new_tokens=int(T), num_directions=K,
                radii=radii, layers=layers, seed=args.seed,
                baseline_duplicate_identical=True, torch=torch.__version__,
                transformers=__import__("transformers").__version__, elapsed_sec=round(time.time() - t0, 1))
    np.savez(os.path.join(args.out, "a1.npz"),
             prompt_ids=pids[0].cpu().numpy(), gen_ids=gen_ids[0].cpu().numpy(),
             margins=margins, top1=top1.cpu().numpy(), top2=top2i.cpu().numpy(),
             norm_h0=norm_h0, dh_norm=dh_norm, dm=dm, cos=cos,
             dh_last=dh_last[: , :dh_last.shape[1]], r_c=r_c, pred_T=pred_T,
             radii=np.asarray(radii), layers=np.asarray(layers),
             simplex=simplex, subspace=sub)
    save_json(os.path.join(args.out, "meta.json"), meta)
    print(json.dumps(meta, indent=2))
    print("min margins (step,value):", sorted([(int(i), float(v)) for i, v in enumerate(margins)], key=lambda x: x[1])[:8])


if __name__ == "__main__":
    main()
