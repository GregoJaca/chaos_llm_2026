"""Per-step gate strength of a recorded greedy/loop orbit.

G_t = sigma_max of the rank-1 pair product (e_i - e_j)(w_i - w_j)^T
    = ||e_i - e_j|| * ||w_i - w_j||   (i = top1, j = top2 at step t)

With the margin m_t this gives the two-token readout-gain prediction
sigma_max(Sigma_p) ~ p(1-p) * G_t,  p = sigmoid(-m_t / T),
for any temperature, without running the loop.
"""
import argparse
import json
import os

import numpy as np
import torch

from chaos_llm.protocol import load_model, save_json


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--a1", default="", help="a1.npz with top1/top2/margins")
    ap.add_argument("--loop", default="", help="softloop.npz with top1_ref/top2_ref/m_ref")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.loop:
        d = np.load(args.loop)
        t1 = torch.as_tensor(d["top1_ref"], device="cuda").long()
        t2 = torch.as_tensor(d["top2_ref"], device="cuda").long()
        m = d["m_ref"].astype(np.float64)
        tag = args.loop
    elif args.a1:
        d = np.load(args.a1)
        t1 = torch.as_tensor(d["top1"], device="cuda").long()
        t2 = torch.as_tensor(d["top2"], device="cuda").long()
        m = d["margins"].astype(np.float64)
        tag = args.a1
    else:
        raise SystemExit("need --a1 or --loop")
    T = len(m)

    model, tok, path = load_model(args.model)
    E = model.get_input_embeddings().weight.detach()
    W = model.lm_head.weight.detach() if getattr(model, "lm_head", None) is not None else E

    de = (E[t1] - E[t2]).float()
    dw = (W[t1] - W[t2]).float()
    G = (de.norm(dim=1) * dw.norm(dim=1)).cpu().numpy().astype(np.float64)
    np.savez(os.path.join(args.out, "gates.npz"),
             G=G, m=m, top1=t1.cpu().numpy(), top2=t2.cpu().numpy())
    save_json(os.path.join(args.out, "meta.json"), dict(model=path, source=tag, steps=T))
    print("saved", args.out)


if __name__ == "__main__":
    main()
