"""A1: prompt perturbation, teacher-forced response, margins, thresholds.

Patched for multi-model use (protocol_general):
  * dtype auto (fp32 <= 2.5B params, bf16 above; --dtype overrides)
  * hidden size / layers / vocab from model_info() (no hardcoded 1536/27/28)
  * bf16-safe: to_np() for every tensor->numpy, perturbation cast to the
    embedding dtype, hard-window mask uses the model dtype
  * --layers defaults to a spread over the actual depth (n=28 reproduces 0,7,14,21,27)
  * guards --fixed-tokens-from against cross-model token reuse
Backward compatible with the DeepSeek protocol (same defaults, same outputs in fp32).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from chaos_llm.protocol_general import (
    DEVICE, default_layer_indices, full_embeddings, generate, hard_window_mask,
    load_model, make_directions, model_info, prompt_ids, save_json, set_window, to_np,
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
    ap.add_argument("--hard-window", action="store_true")
    ap.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16")
    ap.add_argument("--fixed-tokens-from", default="",
                    help="npz file (or dir containing a1.npz) whose prompt_ids/gen_ids are reused")
    ap.add_argument("--layers", default="",
                    help="comma list of hidden-state indices; default = spread over depth "
                         "(0, n/4, n/2, 3n/4, n-1); DeepSeek-1.5B reproduces 0,7,14,21,27")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    text = args.prompt
    if text.startswith("@"):
        line = open(text[1:]).read().splitlines()[0]
        text = line.split("\t", 1)[1] if "\t" in line else line
    radii = [float(x) for x in args.radii.split(",") if x.strip()]

    model, tok, model_path = load_model(args.model, dtype=args.dtype)
    info = model_info(model, tok=tok, model_path=model_path)
    D = info["hidden_size"]
    NL = info["n_hidden_states"]
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers.strip() else default_layer_indices(info["num_layers"]))
    set_window(model, args.window)

    pids = prompt_ids(tok, text)
    P = pids.shape[1]
    if max(layers) >= NL or min(layers) < 0:
        raise ValueError(f"layer indices {layers} out of range for a {info['num_layers']}-block "
                         f"model (hidden_states has {NL} entries: 0..{NL - 1})")
    with torch.inference_mode():
        base_prompt = model.get_input_embeddings()(pids)[0]

    t0 = time.time()
    if args.fixed_tokens_from:
        fpath = args.fixed_tokens_from
        if os.path.isdir(fpath):
            fpath = os.path.join(fpath, "a1.npz")
        fd = np.load(fpath, allow_pickle=True)
        _src_max = int(max(fd["prompt_ids"].max(), fd["gen_ids"].max()))
        if _src_max >= info["vocab_size"]:
            raise ValueError(
                f"--fixed-tokens-from contains token ids up to {_src_max} but this model's "
                f"vocab_size is {info['vocab_size']}: those tokens come from another model")
        fd_meta = os.path.join(os.path.dirname(fpath), "meta.json")
        if os.path.isfile(fd_meta):
            src_model = json.load(open(fd_meta)).get("model")
            if src_model and os.path.basename(str(src_model)) != os.path.basename(args.model):
                print(f"WARNING: fixed tokens were produced by {src_model!r}, running on "
                      f"{args.model!r}: teacher-forcing another model's tokens", flush=True)
        pids = torch.as_tensor(fd["prompt_ids"], device=DEVICE, dtype=torch.long)[None]
        P = int(pids.shape[1])
        gen_ids = torch.as_tensor(fd["gen_ids"], device=DEVICE, dtype=torch.long)[None, :args.max_new_tokens]
        T = int(gen_ids.shape[1])
        with torch.inference_mode():
            base_prompt = model.get_input_embeddings()(pids)[0]
    else:
        # baseline: two identical rows in one batch, verify
        Eref = base_prompt[None].expand(2, -1, -1).contiguous()
        gen = generate(model, Eref, args.max_new_tokens)
        assert torch.equal(gen[0], gen[1]), \
            f"r=0 duplicate rows differ ({info['dtype']}): determinism is broken, do not run"
        gen_ids = gen[0:1]
        T = int(gen_ids.shape[1])
        del Eref, gen
        torch.cuda.empty_cache()

    Efull = full_embeddings(model, pids, gen_ids)
    emb_dtype = Efull.dtype
    am = hard_window_mask(Efull.shape[1], args.window, dtype=emb_dtype, device=DEVICE) \
        if args.hard_window else torch.ones((1, Efull.shape[1]), dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        out0 = model(inputs_embeds=Efull, attention_mask=am,
                     output_hidden_states=True, use_cache=False)
        logits0 = out0.logits[0]  # [L,V]
        pos = torch.arange(P - 1, P + T - 1, device=DEVICE)
        z0 = logits0[pos]  # [T,V]
        top2 = torch.topk(z0, k=2, dim=-1)
        margins = (top2.values[:, 0] - top2.values[:, 1]).float().cpu().numpy().astype(np.float32)
        top1 = top2.indices[:, 0]
        top2i = top2.indices[:, 1]
        h0 = {l: out0.hidden_states[l][0, P:P + T].detach() for l in layers}  # [T,D]
        norm_h0 = np.stack([to_np(h0[l].norm(dim=-1)) for l in layers])  # [n_layers,T]
        del out0, logits0, z0, top2

    simplex, sub = make_directions(args.num_directions, P, D, args.seed)
    sub_t = torch.as_tensor(sub, device=DEVICE, dtype=torch.long)

    nr, K, nl = len(radii), args.num_directions, len(layers)
    dh_norm = np.zeros((nr, K, nl, T), dtype=np.float32)
    dm = np.zeros((nr, K, T), dtype=np.float32)
    cos = np.zeros((nr, K, nl, T), dtype=np.float32)
    dh_last = np.zeros((nr, min(3, K), T, D), dtype=np.float32)

    top1v = top1.view(1, T, 1)
    top2v = top2i.view(1, T, 1)
    m_t = torch.as_tensor(margins, device=DEVICE)

    for ri, r in enumerate(radii):
        for start in range(0, K, args.batch_dirs):
            idx = list(range(start, min(start + args.batch_dirs, K)))
            B = len(idx)
            E = Efull.expand(B, -1, -1).clone()
            for j, di in enumerate(idx):
                delta = torch.zeros(P * D, device=DEVICE, dtype=torch.float32)
                delta.index_copy_(0, sub_t,
                                  torch.as_tensor(simplex[di], device=DEVICE) * r)
                # fp32 accumulate, round once to the model dtype (bf16-safe)
                E[j, :P] = (E[j, :P].float() + delta.view(P, D)).to(emb_dtype)
            with torch.inference_mode():
                out = model(inputs_embeds=E,
                            attention_mask=torch.ones((B, E.shape[1]), dtype=torch.long, device=DEVICE),
                            output_hidden_states=True, use_cache=False)
            zb = out.logits[:, pos, :]  # [B,T,V]
            z1 = torch.gather(zb, 2, top1v.expand(B, -1, -1)).squeeze(-1)
            z2 = torch.gather(zb, 2, top2v.expand(B, -1, -1)).squeeze(-1)
            dm[ri, idx] = to_np((z1 - z2).float() - m_t[None, :])
            del zb, z1, z2, out.logits
            for li, l in enumerate(layers):
                h = out.hidden_states[l][:, P:P + T, :]  # [B,T,D]
                diff = h - h0[l][None]
                dh_norm[ri, idx, li] = to_np(diff.norm(dim=-1))
                cos[ri, idx, li] = to_np(torch.nn.functional.cosine_similarity(
                    h, h0[l][None].expand_as(h), dim=-1))
            for j, di in enumerate(idx):
                if di < 3:
                    dh_last[ri, di] = to_np(
                        out.hidden_states[layers[-1]][j, P:P + T].float() - h0[layers[-1]].float())
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

    meta = dict(model=args.model, model_path=model_path,
                dtype=info["dtype"], params=info["params"], window=args.window,
                prompt=text, prompt_len=int(P), max_new_tokens=int(T), num_directions=K,
                radii=radii, layers=layers, seed=args.seed,
                hidden_size=D, num_layers=info["num_layers"], vocab_size=info["vocab_size"],
                model_class=info["model_class"],
                fixed_tokens_from=args.fixed_tokens_from or None,
                baseline_duplicate_identical=True, torch=torch.__version__,
                transformers=__import__("transformers").__version__,
                elapsed_sec=round(time.time() - t0, 1))
    np.savez(os.path.join(args.out, "a1.npz"),
             prompt_ids=pids[0].cpu().numpy(), gen_ids=gen_ids[0].cpu().numpy(),
             margins=margins, top1=top1.cpu().numpy(), top2=top2i.cpu().numpy(),
             norm_h0=norm_h0, dh_norm=dh_norm, dm=dm, cos=cos,
             dh_last=dh_last[:, :dh_last.shape[1]], r_c=r_c, pred_T=pred_T,
             radii=np.asarray(radii), layers=np.asarray(layers),
             simplex=simplex, subspace=sub)
    save_json(os.path.join(args.out, "meta.json"), meta)
    print(json.dumps(meta, indent=2))
    print("min margins (step,value):", sorted([(int(i), float(v)) for i, v in enumerate(margins)], key=lambda x: x[1])[:8])


if __name__ == "__main__":
    main()
