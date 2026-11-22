"""KV-cache state kicks for a Transformer (DeepSeek-1.5B, FP32, W=100).

The exact analog of the Mamba state kicks: perturb the model's internal
memory (the KV cache) at a chosen step t0, then teacher-force the baseline
tokens and measure whether the response grows or decays.  Every batch
contains the unperturbed reference in row 0 and the kicked copy in row 1,
so all differences are exact in-batch.

Direction: a random dense vector over the concatenated (k, v) entries of the
kicked layers, restricted to the LIVE window positions (those the sliding
window still attends to), total norm = eps.

Saves dh[t] (last layer), dh at three depths, per config.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def prefill_and_step(model, pids, ids_full, P, t0):
    """Build a 2-row cache by feeding the prompt and baseline tokens up to t0-1."""
    B = 2
    o = model(input_ids=pids.expand(B, -1).contiguous(), use_cache=True)
    cache = o.past_key_values
    tcur = P
    for t in range(P, t0):
        o = model(input_ids=ids_full[t - 1:t][None].expand(B, -1).contiguous(),
                  past_key_values=cache, use_cache=True,
                  cache_position=torch.tensor([tcur], device="cuda"),
                  attention_mask=torch.ones((B, tcur + 1), dtype=torch.long, device="cuda"))
        cache = o.past_key_values
        tcur += 1
    return cache, tcur, o


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--kick-steps", default="70,150")
    ap.add_argument("--eps-list", default="0.01,0.1,1.0")
    ap.add_argument("--layer-modes", default="all,0,14,27")
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0all = time.time()

    path = os.path.expanduser(args.model)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True).cuda().eval()
    cfg = model.config
    cfg.sliding_window = int(args.window)
    cfg.use_sliding_window = True
    L = cfg.num_hidden_layers
    print("layers", L, "window", args.window, flush=True)

    text = args.prompt
    if text.startswith("@"):
        line = open(text[1:]).read().splitlines()[0]
        text = line.split("\t", 1)[1] if "\t" in line else line
    pids = tok(text, return_tensors="pt").input_ids.cuda()
    P = pids.shape[1]

    with torch.inference_mode():
        out = model.generate(input_ids=pids.expand(2, -1).contiguous(),
                             max_new_tokens=args.max_new_tokens, do_sample=False,
                             eos_token_id=None, pad_token_id=0)
        assert torch.equal(out[0, P:], out[1, P:]), "r=0 duplicate rows differ"
    gen = out[0, P:]
    ids_full = torch.cat([pids[0], gen])
    T = int(gen.shape[0])

    with torch.inference_mode():
        o0 = model(input_ids=ids_full[None], use_cache=False)
        z0 = o0.logits[0, P - 1:P - 1 + T]
        top2 = torch.topk(z0, k=2, dim=-1)
        margins = (top2.values[:, 0] - top2.values[:, 1]).float().cpu().numpy()
    del o0
    torch.cuda.empty_cache()
    print("baseline done; min margin %.4f; validating cache path" % margins.min(), flush=True)

    # ---- validation: unperturbed stepped pass must reproduce argmax tokens ----
    t_check = min(80, P + T - 2)
    cache, tcur, o = prefill_and_step(model, pids, ids_full, P, t_check)
    with torch.inference_mode():
        o = model(input_ids=ids_full[t_check - 1:t_check][None],
                  past_key_values=cache, use_cache=False, output_hidden_states=True,
                  cache_position=torch.tensor([tcur], device="cuda"),
                  attention_mask=torch.ones((1, tcur + 1), dtype=torch.long, device="cuda"))
        nxt = int(o.logits[0, -1].argmax())
    ok = nxt == int(ids_full[t_check])
    print("cache-stepped argmax reproduces baseline token:", ok, flush=True)
    assert ok, "cache stepping is not an exact simulator; aborting"
    del cache, o
    torch.cuda.empty_cache()

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    K = args.dirs
    W = int(args.window)

    def run_config(eps, t0, layers):
        dh_all = np.zeros((K, args.horizon), np.float32)
        dh_l = np.zeros((K, 3, args.horizon), np.float32)
        for k in range(K):
            cache, tcur, _ = prefill_and_step(model, pids, ids_full, P, t0)
            # live positions in the cache at step t0 (sliding window)
            lo = max(0, tcur - W)
            with torch.no_grad():
                n_ent = 0
                for l in layers:
                    kc = cache.key_cache[l]
                    n_ent += 2 * kc.shape[1] * (tcur - lo) * kc.shape[3]
                scale = eps / np.sqrt(n_ent)
                for l in layers:
                    kc = cache.key_cache[l]
                    vc = cache.value_cache[l]
                    dk = torch.randn(kc.shape, device="cuda", generator=g) * scale
                    dv = torch.randn(vc.shape, device="cuda", generator=g) * scale
                    kc[1, :, lo:, :] += dk[1, :, lo:, :]
                    vc[1, :, lo:, :] += dv[1, :, lo:, :]
            # sanity: rows still identical before any live attention? not required;
            # teacher-force the horizon
            for j in range(args.horizon):
                t = t0 + j
                if t - 1 >= len(ids_full):
                    break
                with torch.inference_mode():
                    o = model(input_ids=ids_full[t - 1:t][None].expand(2, -1).contiguous(),
                              past_key_values=cache, use_cache=True,
                              cache_position=torch.tensor([tcur], device="cuda"),
                              attention_mask=torch.ones((2, tcur + 1), dtype=torch.long, device="cuda"),
                              output_hidden_states=True)
                    cache = o.past_key_values
                    dh_all[k, j] = float((o.hidden_states[-1][1, 0] - o.hidden_states[-1][0, 0]).norm())
                    for li3, l3 in enumerate([0, L // 2, L - 1]):
                        dh_l[k, li3, j] = float(
                            (o.hidden_states[l3][1, 0] - o.hidden_states[l3][0, 0]).norm())
                    tcur += 1
            del cache, o
            torch.cuda.empty_cache()
        return dh_all, dh_l

    results = {}
    eps_list = [float(x) for x in args.eps_list.split(",")]
    t0_list = [int(x) for x in args.kick_steps.split(",")]
    layer_modes = []
    for m in args.layer_modes.split(","):
        layer_modes.append(list(range(L)) if m == "all" else [int(m)])
    for eps in eps_list:
        for t0 in t0_list:
            for li, layers in enumerate(layer_modes):
                key = f"eps{eps:g}_t0{t0}_l{args.layer_modes.split(',')[li]}"
                dh, dhl = run_config(eps, t0, layers)
                results[key] = dict(dh=dh, dh_layers=dhl)
                med = np.median(dh, axis=0)
                print("%-24s dh(0)=%.4g dh(+20)=%.4g dh(+%d)=%.4g (median %d dirs)" % (
                    key, med[0], med[min(20, len(med) - 1)], len(med) - 1, med[-1], K), flush=True)

    np.savez(os.path.join(args.out, "kv_kick.npz"),
             **{k: v["dh"] for k, v in results.items()},
             **{k + "_layers": v["dh_layers"] for k, v in results.items()},
             margins=margins, prompt_len=P, T=T)
    meta = dict(model=args.model, window=W, prompt=text, prompt_len=P, max_new_tokens=T,
                eps_list=eps_list, t0_list=t0_list, layer_modes=args.layer_modes,
                dirs=K, horizon=args.horizon, seed=args.seed, live_only=True,
                elapsed_sec=round(time.time() - t0all, 1))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
