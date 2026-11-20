"""A1 for large models (DGX Spark): prompt perturbation, teacher-forced response,
margins, thresholds — same npz schema as run_a1.py, so all downstream analysis
scripts work unchanged.

Efficiency design (bandwidth-bound GB10, see big_engine.py):
- Baseline free run once, then teacher-force its tokens for every (direction, radius):
  one full-sequence forward per batch of --batch-dirs rows.
- Response is linear in r (validated separately by --linearity mode), so the main
  ensembles run ONE radius per direction; the threshold ladder r_c = m*r/|dm| is
  r-independent via dm/r.
- Never materialises [B,T,V] logits (lm_head row-difference trick).

Resume-safe: each batch writes results/parts/rr{ri}_d{start}.npz and is skipped on
rerun; parts merge into a1.npz at the end.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from chaos_llm.big_engine import (
    Capture, baseline_pass, build_direction_blocks, check_mem, free_generate,
    full_embeddings, load_big, perturb_rows, save_json, set_window, teacher_batch,
    wdiff_tensor,
)


def read_prompt(spec):
    """Prompt spec: plain text, @file (first line), or @file::name (line with
    that tab-separated name)."""
    text = spec
    if text.startswith("@"):
        fname = text[1:]
        want = None
        if "::" in fname:
            fname, want = fname.split("::", 1)
        for line in open(fname).read().splitlines():
            if not line.strip():
                continue
            name, body = (line.split("\t", 1) + [""])[:2] if "\t" in line else ("", line)
            if want is None or name == want:
                return body
        raise ValueError(f"prompt {want!r} not found in {fname}")
    return text


def merge_parts(parts_dir, nr, K, nl, T, D, nvec, radii):
    dh_norm = np.zeros((nr, K, nl, T), dtype=np.float32)
    dm = np.zeros((nr, K, T), dtype=np.float32)
    cos = np.zeros((nr, K, nl, T), dtype=np.float32)
    dh_last = np.zeros((nr, nvec, T, D), dtype=np.float32) if nvec else None
    filled = np.zeros((nr, K), dtype=bool)
    for ri in range(nr):
        for name in sorted(os.listdir(parts_dir)):
            if not name.startswith(f"r{ri}_"):
                continue
            d = np.load(os.path.join(parts_dir, name))
            starts = d["starts"]
            n = len(starts)
            if filled[ri, starts].any():
                raise RuntimeError(f"overlapping part {name}")
            dh_norm[ri, starts] = d["dh_norm"]
            dm[ri, starts] = d["dm"]
            cos[ri, starts] = d["cos"]
            if "vec_starts" in d and d["vec_starts"].size:
                for j, di in enumerate(d["vec_starts"]):
                    dh_last[ri, di] = d["dh_final"][j]
            filled[ri, starts] = True
        missing = int((~filled[ri]).sum())
        if missing:
            raise RuntimeError(f"radius {radii[ri]}: {missing} directions missing from parts")
    return dh_norm, dm, cos, dh_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--num-directions", type=int, default=1000)
    ap.add_argument("--dirs-per-block", type=int, default=20)
    ap.add_argument("--radii", default="0.1")
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--positions", default="",
                    help="lo:hi restrict perturbed prompt positions (flush support scan)")
    ap.add_argument("--batch-dirs", type=int, default=16)
    ap.add_argument("--vector-rows", type=int, default=24)
    ap.add_argument("--layers", default="")
    ap.add_argument("--fixed-tokens-from", default="")
    ap.add_argument("--min-mem-gb", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    final_path = os.path.join(args.out, "a1.npz")
    if os.path.exists(final_path):
        print("a1.npz exists; nothing to do", flush=True)
        return
    parts_dir = os.path.join(args.out, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    text = read_prompt(args.prompt)
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    nr, K = len(radii), args.num_directions
    t0 = time.time()

    model, tok, model_path, info = load_big(args.model, dtype=args.dtype)
    L_total = info["num_layers"]
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        step = max(1, L_total // 8)
        layers = sorted(set(list(range(0, L_total, step)) + [L_total - 1]))
    if args.window:
        set_window(model, args.window)

    from transformers import AutoTokenizer
    pids = tok(text, return_tensors="pt").input_ids.cuda()
    P = pids.shape[1]
    D = info["hidden_size"]
    with torch.inference_mode():
        base_prompt = model.get_input_embeddings()(pids)[0]

    fpath = args.fixed_tokens_from
    if fpath:
        if os.path.isdir(fpath):
            fpath = os.path.join(fpath, "a1.npz")
        fd = np.load(fpath, allow_pickle=True)
        pids = torch.as_tensor(fd["prompt_ids"], device="cuda", dtype=torch.long)[None]
        P = int(pids.shape[1])
        gen_ids = torch.as_tensor(fd["gen_ids"], device="cuda", dtype=torch.long)[None, :args.max_new_tokens]
        T = int(gen_ids.shape[1])
        with torch.inference_mode():
            base_prompt = model.get_input_embeddings()(pids)[0]
    else:
        # baseline: two identical rows in one batch, verify bit-identity
        check_mem(args.min_mem_gb + 1, "baseline generation")
        Eref = base_prompt[None].expand(2, -1, -1)
        gen = free_generate(model, Eref, args.max_new_tokens)
        assert torch.equal(gen[0], gen[1]), "r=0 duplicate rows differ"
        gen_ids = gen[0:1]
        T = gen_ids.shape[1]
        del Eref, gen
        torch.cuda.empty_cache()

    Efull = full_embeddings(model, pids, gen_ids)
    capture = Capture(model, layers)
    margins, top1, top2, h0 = baseline_pass(model, Efull, P, T, layers, capture)
    norm_h0 = np.stack([h0[l].norm(dim=-1).float().cpu().numpy() for l in layers])
    wdiff = wdiff_tensor(model, top1, top2)
    print(f"baseline done: P={P} T={T} min margins {np.sort(margins)[:4].tolist()}", flush=True)

    positions = None
    if args.positions:
        lo, hi = args.positions.split(":")
        positions = (int(lo), int(hi))
    simplex, subspace = build_direction_blocks(K, P, D, block=args.dirs_per_block,
                                               seed=args.seed, positions=positions)
    nl = len(layers)

    for ri, r in enumerate(radii):
        # resume: directions already covered by existing parts for this radius
        covered = np.zeros(K, dtype=bool)
        for name in os.listdir(parts_dir):
            if name.startswith(f"r{ri}_"):
                covered[np.load(os.path.join(parts_dir, name))["starts"]] = True
        todo = [di for di in range(K) if not covered[di]]
        cur = max(1, args.batch_dirs)
        i = 0
        while i < len(todo):
            chunk = todo[i:i + cur]
            try:
                E = perturb_rows(Efull, P, D, simplex, subspace, chunk, r)
                with torch.inference_mode():
                    res = teacher_batch(
                        model, E, P, T, layers, h0, wdiff, capture,
                        vector_rows=[j for j, di in enumerate(chunk) if di < min(args.vector_rows, K)])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur == 1:
                    raise
                cur = max(1, cur // 2)
                print(f"OOM at r={r:g}; halving batch to {cur}", flush=True)
                continue
            vec_idx = [di for di in chunk if di < min(args.vector_rows, K)]
            np.savez(os.path.join(parts_dir, f"r{ri}_d{chunk[0]:05d}.npz"),
                     starts=np.asarray(chunk, dtype=np.int64),
                     dh_norm=res["dh_norm"], dm=res["dm"], cos=res["cos"],
                     vec_starts=np.asarray(vec_idx, dtype=np.int64),
                     dh_final=(np.stack([res["dh_final_rows"][j] for j, di in enumerate(chunk)
                                          if di < min(args.vector_rows, K)])
                               if vec_idx else np.zeros((0, T, D), dtype=np.float32)))
            print(f"r={r:g} dirs {chunk[0]}-{chunk[-1]} (n={len(chunk)})  t={time.time()-t0:.0f}s "
                  f"mem_avail={check_mem(0):.1f}G", flush=True)
            i += len(chunk)
            del E, res
            torch.cuda.empty_cache()

    capture.remove()
    del wdiff, h0, Efull
    torch.cuda.empty_cache()

    nvec = min(args.vector_rows, K)
    dh_norm, dm, cos, dh_last = merge_parts(parts_dir, nr, K, nl, T, D, nvec, radii)

    eps = 1e-30
    r_c = margins[None, None, :] / np.maximum(np.abs(dm) / np.asarray(radii)[:, None, None], eps)
    pred_T = np.full((nr, K), -1, dtype=np.int32)
    for ri, r in enumerate(radii):
        hit = r_c[ri] <= r
        anyhit = hit.any(axis=1)
        pred_T[ri, anyhit] = hit.argmax(axis=1)[anyhit]

    meta = dict(model=args.model, model_path=model_path, dtype=args.dtype,
                window=args.window, prompt=text, prompt_len=int(P),
                max_new_tokens=int(T), num_directions=K, dirs_per_block=args.dirs_per_block,
                radii=radii, layers=layers, seed=args.seed,
                positions=positions,
                fixed_tokens_from=args.fixed_tokens_from or None,
                baseline_duplicate_identical=True, batch_dirs=args.batch_dirs,
                engine="run_a1_big", arch=info["arch"],
                torch=torch.__version__, host=os.uname().nodename,
                transformers=__import__("transformers").__version__,
                elapsed_sec=round(time.time() - t0, 1))
    np.savez(final_path,
             prompt_ids=pids[0].cpu().numpy(), gen_ids=gen_ids[0].cpu().numpy(),
             margins=margins, top1=top1, top2=top2, norm_h0=norm_h0,
             dh_norm=dh_norm, dm=dm, cos=cos,
             dh_last=dh_last if dh_last is not None else np.zeros(0),
             r_c=r_c, pred_T=pred_T,
             radii=np.asarray(radii), layers=np.asarray(layers),
             simplex=simplex, subspace=subspace)
    save_json(os.path.join(args.out, "meta.json"), meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "prompt"}, indent=2), flush=True)
    print("min margins (step,value):", sorted(
        [(int(i), float(v)) for i, v in enumerate(margins)], key=lambda x: x[1])[:8], flush=True)


if __name__ == "__main__":
    main()
