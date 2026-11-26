"""Attention mass on prompt vs generated tokens, per layer and generated step."""
import argparse, json, os
import numpy as np
import torch
from chaos_llm.protocol import (full_embeddings, generate, load_model, prompt_ids,
                                save_json, set_window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--layers", default="0,7,14,21,27")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    text = open(args.prompt[1:]).read().splitlines()[0]
    if "\t" in text:
        text = text.split("\t", 1)[1]

    model, tok, path = load_model(args.model, attn="eager", dtype=args.dtype)
    set_window(model, args.window)
    pids = prompt_ids(tok, text)
    P = pids.shape[1]
    with torch.inference_mode():
        E0 = model.get_input_embeddings()(pids)
        gen = generate(model, E0, args.max_new_tokens)
    gen_ids = gen[:, P:]
    E = full_embeddings(model, pids, gen_ids)
    L = E.shape[1]
    mask = torch.ones((1, L), dtype=torch.long, device="cuda")
    with torch.inference_mode():
        out = model(inputs_embeds=E, attention_mask=mask, output_attentions=True, use_cache=False)
    T = int(gen_ids.shape[1])
    frac = np.zeros((len(layers), T), np.float32)
    for li, l in enumerate(layers):
        A = out.attentions[l][0].float()          # (heads, L, L)
        for t in range(T):
            pos = P + t
            a = A[:, pos, :].mean(0)              # average over heads
            frac[li, t] = float(a[:P].sum() / (a.sum() + 1e-30))
    np.savez(os.path.join(args.out, "attn_frac.npz"), frac=frac,
             layers=np.asarray(layers), prompt_len=P, gen_ids=gen_ids[0].cpu().numpy())
    save_json(os.path.join(args.out, "meta.json"), dict(
        model=args.model, window=args.window, prompt_len=int(P), T=T, layers=layers,
        dtype=args.dtype))
    print("saved", args.out, frac.shape)


if __name__ == "__main__":
    main()
