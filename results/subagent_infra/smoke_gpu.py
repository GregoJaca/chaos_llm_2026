"""smoke_gpu.py -- one-time validation of a NEW model before any real run.

Implements the smoke sequence from CHECKLIST.md:
  1. load with automatic dtype (prints model_info: hidden size, layers, vocab, dtype)
  2. one forward: logits finite, shapes sane, hidden_states count == num_layers+1
  3. r=0 determinism: two identical rows in ONE batch must generate identical tokens,
     and generation must run the full horizon (eos_token_id=None)
  4. window check (with --window > 0): empirical proof that the window clips attention
  5. r-linearity probe: teacher-forced margin response at r=1e-3..1 for 2 directions;
     prints dm(r)/r and recommends a radii ladder for run_a1 (bf16 models: expect a
     numeric floor around r~1e-2 -- the perturbation rounds away below it)

Run this BEFORE launching any queue on a new model. GPU recommended; CPU works for
tiny models. Saves a json report next to --out if given.

Usage:
    python smoke_gpu.py --model Qwen/Qwen3-4B-Instruct-2507 --window 100 \
        --prompt @prompts.txt --out results/smoke/qwen3_4b
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from chaos_llm.protocol_general import (
    DEVICE, full_embeddings, generate, load_model, make_directions, model_info,
    prompt_ids, save_json, set_window, to_np, window_is_effective,
)


def read_prompt(prompt_arg):
    text = prompt_arg
    if text.startswith("@"):
        line = open(text[1:], encoding="utf-8").read().splitlines()[0]
        text = line.split("\t", 1)[1] if "\t" in line else line
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--window", type=int, default=100,
                    help="window to validate; pass 0 to skip the window check")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--radii", default="1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0")
    ap.add_argument("--dirs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.dirs < 2:
        raise SystemExit("--dirs must be >= 2 (degenerate simplex)")
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    text = read_prompt(args.prompt)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    report = {"model": args.model, "prompt": text, "checks": {}}

    # ---- 1. load -----------------------------------------------------------
    t0 = time.time()
    model, tok, path = load_model(args.model, dtype=args.dtype)
    info = model_info(model, tok=tok, model_path=path)
    print("[1] loaded in %.1fs" % (time.time() - t0))
    print(json.dumps(info, indent=2, default=str))
    report["model_info"] = info
    D, NL = info["hidden_size"], info["n_hidden_states"]
    report["checks"]["load"] = "PASS"
    set_window(model, args.window)

    # ---- 2. one forward ----------------------------------------------------
    pids = prompt_ids(tok, text)
    P = pids.shape[1]
    with torch.inference_mode():
        emb = model.get_input_embeddings()(pids)
        o = model(inputs_embeds=emb, attention_mask=torch.ones_like(pids),
                  output_hidden_states=True, use_cache=False)
    assert torch.isfinite(o.logits).all(), "logits contain NaN/Inf"
    assert o.logits.dim() == 3 and o.logits.shape[0] == 1 and o.logits.shape[1] == P, \
        f"logit shape {tuple(o.logits.shape)} unexpected for prompt len {P}"
    if o.logits.shape[-1] != info["embedding_rows"]:
        print(f"    note: logits vocab dim {o.logits.shape[-1]} != embedding rows "
              f"{info['embedding_rows']} (padded embeddings); config vocab_size="
              f"{info['vocab_size']}")
    print(f"[2] forward OK: logits finite, shape {tuple(o.logits.shape)}, "
          f"{len(o.hidden_states)} hidden states (expected {NL})")
    if len(o.hidden_states) != NL:
        print(f"    WARNING: hidden_states count {len(o.hidden_states)} != num_layers+1 ({NL}); "
              "layer indexing in the run scripts assumes embeddings + one entry per block")
    report["checks"]["forward"] = "PASS"
    del o, emb
    torch.cuda.empty_cache()

    # ---- 3. r=0 determinism --------------------------------------------------
    with torch.inference_mode():
        base = model.get_input_embeddings()(pids)[0]
    Eref = base[None].expand(2, -1, -1).contiguous()
    gen = generate(model, Eref, args.max_new_tokens)  # asserts the full horizon
    identical = bool(torch.equal(gen[0], gen[1]))
    print(f"[3] r=0 duplicate rows identical: {identical}; "
          f"generation ran the full {gen.shape[1]} tokens (no early stop)")
    if not identical:
        raise SystemExit("r=0 duplicates differ: DO NOT RUN THIS MODEL/TOLERANCE COMBO; "
                         "see HANDOFF 'bf16: r=0 diverges' -- never mix precisions")
    report["checks"]["r0_duplicate_identical"] = identical
    report["checks"]["generation_full_length"] = int(gen.shape[1])
    gen_ids = gen[0:1]
    T = int(gen_ids.shape[1])
    del Eref, gen
    torch.cuda.empty_cache()

    # ---- 4. window check -----------------------------------------------------
    if args.window > 0:
        win = window_is_effective(model, args.window)
        print(f"[4] window check (W={args.window}): effective={win['effective']} "
              f"windowed_diff={win['windowed_max_diff']:g} "
              f"full_attention_diff={win['full_attention_max_diff']:g}")
        print(f"    {win['note']}")
        report["checks"]["window_effective"] = win
    else:
        print("[4] window check skipped (--window 0)")
    set_window(model, args.window)

    # ---- 5. r-linearity probe -------------------------------------------------
    Efull = full_embeddings(model, pids, gen_ids)
    emb_dtype = Efull.dtype
    with torch.inference_mode():
        o0 = model(inputs_embeds=Efull,
                   attention_mask=torch.ones((1, Efull.shape[1]), dtype=torch.long, device=DEVICE),
                   output_hidden_states=False, use_cache=False)
        pos = torch.arange(P - 1, P + T - 1, device=DEVICE)
        top2 = torch.topk(o0.logits[0][pos], k=2, dim=-1)
        margins = to_np(top2.values[:, 0] - top2.values[:, 1])
        top1v = top2.indices[:, 0].view(1, T, 1)   # [1,T,1] -> expand to [K,T,1]
        top2v = top2.indices[:, 1].view(1, T, 1)
        del o0

    simplex, sub = make_directions(args.dirs, P, D, args.seed)
    sub_t = torch.as_tensor(sub, device=DEVICE, dtype=torch.long)
    K = args.dirs
    rows = []
    for r in radii:
        E = Efull.expand(K, -1, -1).clone()
        with torch.inference_mode():
            for j in range(K):
                delta = torch.zeros(P * D, device=DEVICE, dtype=torch.float32)
                delta.index_copy_(0, sub_t, torch.as_tensor(simplex[j], device=DEVICE) * r)
                E[j, :P] = (E[j, :P].float() + delta.view(P, D)).to(emb_dtype)
            out = model(inputs_embeds=E,
                        attention_mask=torch.ones((K, E.shape[1]), dtype=torch.long, device=DEVICE),
                        output_hidden_states=False, use_cache=False)
            zb = out.logits[:, pos, :]
            z1 = torch.gather(zb, 2, top1v.expand(K, -1, -1)).squeeze(-1)
            z2 = torch.gather(zb, 2, top2v.expand(K, -1, -1)).squeeze(-1)
            dm = to_np(z1 - z2) - margins[None, :]   # [K,T]
            flips = int((np.abs(dm) >= margins[None, :]).sum())
        rows.append(dict(r=r, dm_mean=float(np.abs(dm).mean()), dm_max=float(np.abs(dm).max()),
                         dm_over_r=float(np.abs(dm).mean() / r), flips=flips))
        del out, zb, z1, z2, dm, E
        torch.cuda.empty_cache()

    print("[5] r-linearity (teacher-forced margin response, 2 directions):")
    print(f"    {'r':>8s} {'mean|dm|':>12s} {'max|dm|':>10s} {'mean|dm|/r':>12s} flips>=margin")
    for row in rows:
        print(f"    {row['r']:8g} {row['dm_mean']:12.3e} {row['dm_max']:10.3e} "
              f"{row['dm_over_r']:12.3e} {row['flips']:>6d}")
    # a response is 'zero' when not a single margin moved
    nonzero = [row for row in rows if row["dm_max"] > 0.0]
    band = []
    if nonzero:
        rho = np.array([row["dm_over_r"] for row in nonzero])
        med = float(np.median(rho))
        band = [row for row in nonzero if 0.8 * med <= row["dm_over_r"] <= 1.25 * med]
    if band:
        lo, hi = band[0]["r"], band[-1]["r"]
        ladder = np.geomspace(lo, max(hi, lo * 100), num=12)
        rec = [float(f"{x:.3g}") for x in ladder]
        print(f"    linear band: r in [{lo:g}, {hi:g}] (mean|dm|/r within +-20% of {med:.3g})")
        print(f"    SUGGESTED run_a1 --radii: {','.join(f'{x:g}' for x in rec)}")
        report["recommended_radii"] = rec
        report["linear_band"] = [lo, hi]
    else:
        print("    NO radius gave a measurable response: prompt margins may be huge or "
              "the dtype floor is above the tested radii (bf16: start the sweep at r~1e-2)")
        report["recommended_radii"] = None
    report["linearity"] = rows

    if args.out:
        save_json(os.path.join(args.out, "smoke.json"), report)
        print(f"report saved to {os.path.join(args.out, 'smoke.json')}")
    print("SMOKE OK -- proceed with the CHECKLIST (choose radii from the linear band, "
          "then launch run_a1)")


if __name__ == "__main__":
    main()
