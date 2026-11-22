"""Transport kernel K(p,t) and superposition test for Mamba-130m.

The control-side replica of master.md section 7:
  - K(p,t) = ||delta h_{t,L}|| / r when ONLY prompt position p is perturbed
    (dense random unit vector at that position), 8 directions, r=0.1,
    teacher-forced 300 tokens, last layer, in-batch reference.
  - Superposition: joint run (same unit u added to every prompt position,
    radius per position r) vs the sum of single-position runs; compare the
    full response vectors A_t and S_t = sum_p B_{p,t}.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--state-spaces--mamba-130m-hf")
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--sup-radii", default="0.01,0.03,0.1")
    ap.add_argument("--sup-trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0all = time.time()

    path = os.path.expanduser(args.model)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, local_files_only=True,
        attn_implementation="eager", low_cpu_mem_usage=True).cuda().eval()
    D = model.config.hidden_size
    L = model.config.num_hidden_layers

    text = args.prompt
    if text.startswith("@"):
        line = open(text[1:]).read().splitlines()[0]
        text = line.split("\t", 1)[1] if "\t" in line else line
    pids = tok(text, return_tensors="pt").input_ids.cuda()
    P = int(pids.shape[1])

    # baseline greedy generation
    with torch.inference_mode():
        out = model.generate(input_ids=pids.expand(2, -1).contiguous(),
                             max_new_tokens=args.max_new_tokens, do_sample=False,
                             eos_token_id=None, pad_token_id=0)
        assert torch.equal(out[0, P:], out[1, P:])
    gen = out[0, P:]
    T = int(gen.shape[0])
    E = model.get_input_embeddings()(torch.cat([pids[0], gen]))[None]  # [1, P+T, D]
    mask = torch.ones((1, P + T), dtype=torch.long, device="cuda")

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    K = args.dirs

    # ---------------- transport kernel K(p,t) ----------------
    kern = np.zeros((P, K, T), np.float32)
    for p in range(P):
        B = K + 1
        Eb = E.expand(B, -1, -1).clone()
        for k in range(K):
            v = torch.randn(D, device="cuda", generator=g)
            v = v / v.norm()
            Eb[k + 1, p] += args.r * v
        with torch.inference_mode():
            o = model(inputs_embeds=Eb, attention_mask=mask.expand(B, -1),
                      output_hidden_states=True, use_cache=False)
        hlast = o.hidden_states[-1][:, P:P + T, :]
        href = hlast[0]
        for k in range(K):
            kern[p, k] = (hlast[k + 1] - href).norm(dim=-1).float().cpu().numpy()
        del o, Eb
        if p % 16 == 0:
            print("kernel p=%d/%d  %.1fs" % (p, P, time.time() - t0all), flush=True)
    np.save(os.path.join(args.out, "kernel.npy"), kern)
    Kmed = np.median(kern, axis=1) / args.r  # [P, T]
    print("kernel done: K median at p=0: t=1 %.3f t=100 %.3f t=299 %.3f; "
          "peak t at p=P-1: %d" % (Kmed[0, 1], Kmed[0, 100], Kmed[0, 299],
                                   int(Kmed[-1].argmax())), flush=True)

    # ---------------- superposition ----------------
    sup = {}
    for r_pos in [float(x) for x in args.sup_radii.split(",")]:
        rel_errs = []
        coses = []
        for trial in range(args.sup_trials):
            u = torch.randn(D, device="cuda", generator=g)
            u = u / u.norm()
            # joint: u at every position
            B = 2
            Ej = E.expand(B, -1, -1).clone()
            Ej[1, :P] += r_pos * u
            with torch.inference_mode():
                oj = model(inputs_embeds=Ej, attention_mask=mask.expand(B, -1),
                           output_hidden_states=True, use_cache=False)
            A = (oj.hidden_states[-1][1, P:P + T] - oj.hidden_states[-1][0, P:P + T])  # [T,D]
            # single-position runs, accumulated position by position
            S = torch.zeros(T, D, device="cuda")
            for p in range(P):
                Eb = E.expand(2, -1, -1).clone()
                Eb[1, p] += r_pos * u
                with torch.inference_mode():
                    ob = model(inputs_embeds=Eb, attention_mask=mask.expand(2, -1),
                                output_hidden_states=True, use_cache=False)
                S += (ob.hidden_states[-1][1, P:P + T] - ob.hidden_states[-1][0, P:P + T])
                del ob, Eb
            rel = ((A - S).norm(dim=-1) / A.norm(dim=-1).clamp(min=1e-30)).float().cpu().numpy()
            cos = torch.nn.functional.cosine_similarity(A, S, dim=-1).float().cpu().numpy()
            rel_errs.append(rel)
            coses.append(cos)
            print("sup r=%g trial %d: median ||A-S||/||A|| = %.4f, median cos = %.5f" % (
                r_pos, trial, float(np.median(rel)), float(np.median(cos))), flush=True)
        sup[r_pos] = dict(rel_err=[float(np.median(x)) for x in rel_errs],
                          cos=[float(np.median(x)) for x in coses],
                          rel_all=np.stack(rel_errs).mean(0).tolist(),
                          anorm=None)
    # save A norms for one trial (error vs amplitude figure)
    json_sup = {("%g" % k): v for k, v in sup.items()}
    with open(os.path.join(args.out, "superposition.json"), "w") as f:
        json.dump(json_sup, f, indent=1)
    meta = dict(model=args.model, prompt=text, prompt_len=P, T=T, r_kernel=args.r,
                dirs=K, sup_radii=args.sup_radii, seed=args.seed,
                elapsed_sec=round(time.time() - t0all, 1))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
