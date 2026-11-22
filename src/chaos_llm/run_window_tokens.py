"""Free-run token streams at several window sizes (native window; DeepSeek only)."""
import argparse
import json
import os

import numpy as np
import torch

from chaos_llm.protocol import generate, load_model, prompt_ids, save_json, set_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", default="prompts.txt")
    ap.add_argument("--windows", default="60,100,150,200,300,500,1000,4096")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    text = open(args.prompt_file).read().splitlines()[0]
    if "\t" in text:
        text = text.split("\t", 1)[1]
    model, tok, _ = load_model(args.model, dtype=args.dtype)
    pids = prompt_ids(tok, text)
    ids = {}
    for w in [int(x) for x in args.windows.split(",")]:
        set_window(model, w)
        with torch.inference_mode():
            emb = model.get_input_embeddings()(pids)
            gen = generate(model, emb, args.tokens)
        ids[str(w)] = gen[0].cpu().numpy().astype(np.int32)
        print(f"W={w}: first 6 tokens {ids[str(w)][:6].tolist()}", flush=True)
    np.savez(os.path.join(args.out, "tokens.npz"),
             **{f"w{k}": v for k, v in ids.items()})
    save_json(os.path.join(args.out, "meta.json"),
              dict(model=args.model, prompt=text, prompt_len=int(pids.shape[1]),
                   windows=[int(x) for x in args.windows.split(",")], tokens=args.tokens))
    print("saved", args.out)


if __name__ == "__main__":
    main()
