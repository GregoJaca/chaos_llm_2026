"""First-contact smoke test for a big model on the DGX Spark.

Staged, with memory readings between every stage; run this BEFORE any real
experiment. Usage (from the repo root on spark):

  PYTHONPATH=src ~/LLM/venv/bin/python scripts_spark/smoke_big.py \
      --model ~/LLM/models/DeepSeek-R1-Distill-Qwen-14B [--tokens 40]

Stages:
  0. memory before
  1. fp32 load (device_map streaming)   -> weights_gb vs MemAvailable
  2. 2-row greedy generation            -> bit-identity (protocol validity)
  3. teacher-forced batch, 2 radii      -> linearity ratio, timing per row
  4. free-run batch, chunk rows         -> timing, first-mismatch sanity
  5. model freed                        -> memory returned?
"""
import argparse
import time

import numpy as np
import torch

from chaos_llm.big_engine import (
    Capture, baseline_pass, build_direction_blocks, check_mem, free_generate,
    full_embeddings, load_big, mem_available_gb, perturb_rows, set_window,
    teacher_batch, wdiff_tensor,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--batch-dirs", type=int, default=8)
    args = ap.parse_args()

    print(f"[0] MemAvailable before load: {mem_available_gb():.1f} GB", flush=True)
    t0 = time.time()
    model, tok, path, info = load_big(args.model, dtype=args.dtype)
    print(f"[1] loaded {info} in {time.time()-t0:.0f}s; "
          f"cuda allocated {torch.cuda.memory_allocated()/2**30:.1f} GiB; "
          f"MemAvailable now {mem_available_gb():.1f} GB", flush=True)
    if args.window:
        set_window(model, args.window)
    D = info["hidden_size"]

    pids = tok("Write a short essay about the ocean.", return_tensors="pt").input_ids.cuda()
    P = pids.shape[1]
    with torch.inference_mode():
        base_prompt = model.get_input_embeddings()(pids)[0]

    # stage 2: bit-identity of two identical rows
    Eref = base_prompt[None].expand(2, -1, -1)
    t0 = time.time()
    gen = free_generate(model, Eref, args.tokens)
    dt = time.time() - t0
    ok = bool(torch.equal(gen[0], gen[1]))
    print(f"[2] generation 2x{args.tokens} tokens: {dt:.1f}s "
          f"({dt/args.tokens*1000:.0f} ms/step) bit-identity={ok}", flush=True)
    assert ok, "r=0 rows differ"
    gen_ids = gen[0:1]
    del Eref, gen
    torch.cuda.empty_cache()

    # stage 3: teacher-forced batches at two radii
    Efull = full_embeddings(model, pids, gen_ids)
    layers = [0, info["num_layers"] - 1]
    capture = Capture(model, layers)
    margins, top1, top2, h0 = baseline_pass(model, Efull, P, args.tokens, layers, capture)
    wdiff = wdiff_tensor(model, top1, top2)
    print(f"[3] baseline margins: min={margins.min():.4f} "
          f"median={np.median(margins):.3f}", flush=True)
    simplex, subspace = build_direction_blocks(16, P, D, block=16, seed=7)
    res_by_r = {}
    for r in (0.03, 0.1):
        t0 = time.time()
        E = perturb_rows(Efull, P, D, simplex, subspace, list(range(args.batch_dirs)), r)
        with torch.inference_mode():
            res = teacher_batch(model, E, P, args.tokens, layers, h0, wdiff, capture)
        dt = time.time() - t0
        res_by_r[r] = res
        print(f"[3] teacher batch {args.batch_dirs} rows r={r}: {dt:.1f}s "
              f"({dt/args.batch_dirs:.2f}s/row) dh_last median={np.median(res['dh_norm'][:, -1]):.4f} "
              f"MemAvailable={mem_available_gb():.1f} GB", flush=True)
        del E, res
        torch.cuda.empty_cache()
    a = np.median(res_by_r[0.03]["dh_norm"][:, -1])
    b = np.median(res_by_r[0.1]["dh_norm"][:, -1])
    print(f"[3] linearity ratio dh(0.1)/dh(0.03) = {b/max(a,1e-30):.2f} (expect ~{0.1/0.03:.2f})", flush=True)
    capture.remove()
    del Efull, h0, wdiff
    torch.cuda.empty_cache()

    # stage 4: batched free run
    t0 = time.time()
    E = base_prompt[None].expand(args.batch_dirs, -1, -1).clone()
    out = free_generate(model, E, args.tokens)
    dt = time.time() - t0
    same = sum(int(torch.equal(out[0], out[j])) for j in range(args.batch_dirs))
    print(f"[4] free run {args.batch_dirs}x{args.tokens}: {dt:.1f}s "
          f"({dt/args.tokens*1000:.0f} ms/step for the whole batch) identical rows={same}/{args.batch_dirs}", flush=True)
    del E, out
    torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[5] after free: MemAvailable {mem_available_gb():.1f} GB "
          f"(cuda allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB)", flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
