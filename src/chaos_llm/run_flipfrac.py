"""Tie-corrected flip fractions for a control model (Mamba-130m).

Replicates the master.md section-8 protocol: for each of 8 prompts, 192
random dense unit directions u in R^{P x d} (norm 1 over the whole tensor),
4 radii; free-running greedy generation of 300 tokens; record whether the
generated sequence differs from the unperturbed run, and the step + the
unperturbed margin at the first difference.  Analysis excludes flips at
steps with margin <= 0.02 (quantisation ties; none expected in fp32).
In-batch reference row in every batch.
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
    ap.add_argument("--prompts", required=True, help="file with 8 named prompts (name<TAB>text)")
    ap.add_argument("--radii", default="0.1,0.3,0.6,1.0")
    ap.add_argument("--dirs", type=int, default=192)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=5)
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

    lines = [l for l in open(args.prompts).read().splitlines() if l.strip()]
    radii = [float(x) for x in args.radii.split(",")]
    results = {}
    for line in lines:
        name, text = line.split("\t", 1) if "\t" in line else (f"p{len(results)}", line)
        pids = tok(text, return_tensors="pt").input_ids.cuda()
        P = int(pids.shape[1])
        emb = model.get_input_embeddings()(pids)[0]  # [P, D]
        # baseline generation + margins
        with torch.inference_mode():
            out = model.generate(inputs_embeds=emb[None].expand(2, -1, -1).contiguous(),
                                  attention_mask=torch.ones((2, P), dtype=torch.long, device="cuda"),
                                  max_new_tokens=args.max_new_tokens, do_sample=False,
                                  eos_token_id=None, pad_token_id=0)
            assert torch.equal(out[0], out[1])
            gen = out[0:1]
            Ef = model.get_input_embeddings()(torch.cat([pids, gen], dim=1))
            o0 = model(inputs_embeds=Ef, attention_mask=torch.ones((1, Ef.shape[1]), dtype=torch.long, device="cuda"),
                       use_cache=False)
            z0 = o0.logits[0, P - 1:P - 1 + args.max_new_tokens]
            top2 = torch.topk(z0, k=2, dim=-1)
            margins = (top2.values[:, 0] - top2.values[:, 1]).float().cpu().numpy()
        del o0, out
        torch.cuda.empty_cache()
        T = int(gen.shape[1])
        res = {}
        for r in radii:
            rng = np.random.default_rng(args.seed + int(r * 1000))
            flags = np.zeros((args.dirs,), bool)
            steps = np.full((args.dirs,), -1, np.int32)
            done = 0
            while done < args.dirs:
                B = min(args.chunk, args.dirs - done)
                # dense random unit directions over the whole P x D tensor
                V = torch.as_tensor(rng.standard_normal((B, P, D)), device="cuda", dtype=torch.float32)
                V = V / V.flatten(1).norm(dim=1).view(B, 1, 1)
                Eb = emb[None].expand(B + 1, -1, -1).clone()
                Eb[1:] += r * V
                with torch.inference_mode():
                    o = model.generate(inputs_embeds=Eb,
                                       attention_mask=torch.ones((B + 1, P), dtype=torch.long, device="cuda"),
                                       max_new_tokens=T, do_sample=False,
                                       eos_token_id=None, pad_token_id=0)
                ref = o[0].cpu()
                for k in range(B):
                    row = o[k + 1].cpu()
                    mm = np.where(row.numpy() != ref.numpy())[0]
                    if len(mm):
                        flags[done + k] = True
                        steps[done + k] = int(mm[0])
                done += B
                del o, Eb
                torch.cuda.empty_cache()
            res["%g" % r] = dict(flags=flags.tolist(), steps=steps.tolist())
            flips = int(flags.sum())
            print("%s r=%g: %d/%d flipped" % (name, r, flips, args.dirs), flush=True)
        results[name] = dict(margins=margins.tolist(), radii=res, prompt_len=P, T=T)
        np.savez(os.path.join(args.out, "flipfrac_%s.npz" % name),
                 margins=margins,
                 **{("flags_%g" % r): np.asarray(res["%g" % r]["flags"]) for r in radii},
                 **{("steps_%g" % r): np.asarray(res["%g" % r]["steps"]) for r in radii})
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(dict(model=args.model, prompts_file=args.prompts, radii=radii,
                       dirs=args.dirs, max_new_tokens=args.max_new_tokens, seed=args.seed,
                       names=list(results.keys()),
                       elapsed_sec=round(time.time() - t0all, 1)), f, indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
