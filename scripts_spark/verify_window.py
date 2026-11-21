"""Verify the sliding window is ACTUALLY applied (transformers >=4.5x gates it
on config.layer_types, which stock Qwen2 configs never set — set_window must
patch it, and this test proves it worked).

Method: teacher-force the same 300 tokens twice, once full attention, once
window 100. States must be bit-identical while nothing falls out of the
window, and must differ from the first position where a token is excluded.
For P=65, W=100: generated position t (absolute 65+t) excludes position
65+t-100 = t-35 for the first time at t=35 -> exact departure expected at
t=35 (first state index where max|dh| > 0), zero before.

Exit code 0 and 'WINDOW_OK' on success; nonzero and 'WINDOW_BROKEN' otherwise.
"""
import argparse
import os
import sys

import numpy as np
import torch

from chaos_llm.big_engine import full_embeddings, load_big, set_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--fixed-tokens-from", required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--tokens", type=int, default=150)
    args = ap.parse_args()

    model, tok, _mp, info = load_big(args.model)
    text = open(args.prompt[1:]).read().splitlines()[0]
    text = text.split("\t", 1)[1] if "\t" in text else text
    pids = tok(text, return_tensors="pt").input_ids.cuda()
    P = pids.shape[1]
    d = np.load(args.fixed_tokens_from)
    gids = torch.as_tensor(d["gen_ids"], device="cuda", dtype=torch.long)[None, :args.tokens]
    T = gids.shape[1]
    E = full_embeddings(model, pids, gids)
    L = E.shape[1]
    mask2d = torch.ones((1, L), dtype=torch.long, device="cuda")

    with torch.inference_mode():
        set_window(model, 0)
        h_full = model(inputs_embeds=E, attention_mask=mask2d, use_cache=False,
                       output_hidden_states=True).hidden_states[-1][0, P:P + T].clone()
        set_window(model, args.window)
        h_win = model(inputs_embeds=E, attention_mask=mask2d, use_cache=False,
                      output_hidden_states=True).hidden_states[-1][0, P:P + T].clone()

    dh = (h_win - h_full).norm(dim=-1).float().cpu().numpy()
    first = int(np.argmax(dh > 0)) if (dh > 0).any() else -1
    expect = args.window - P
    print("P =", P, "W =", args.window, "expected first departure t* =", expect)
    print("first differing generated step:", first)
    print("max|dh| for t < t*:", float(dh[:max(first, 0)].max()) if first > 0 else 0.0)
    print("||dh|| at t*, t*+5, t*+20:", [round(float(x), 6) for x in dh[first:first + 21:5]] if first >= 0 else None)
    ok = first == expect
    print("WINDOW_OK" if ok else "WINDOW_BROKEN")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
