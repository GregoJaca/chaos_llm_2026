"""Calibration for a large model on the DGX Spark (fast, ~10-20 min).

Produces the numbers needed to design the real experiments:
1. bit-identity of two r=0 rows in one batch (protocol validity),
2. baseline margin spectrum m_t (near-ties -> the interesting r scale),
3. response linearity in r for a few directions (floor and ceiling),
4. a preview of the signed threshold ladder r_* (flux-tube radius scale).

Prints a recommended radius window. Nothing is saved except a small npz.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--radii", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0")
    ap.add_argument("--num-directions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--min-mem-gb", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    model, tok, model_path, info = load_big(args.model, dtype=args.dtype)
    print(f"loaded {info}  mem_avail={check_mem(0):.1f}GB", flush=True)
    if args.window:
        set_window(model, args.window)
    L_total, D = info["num_layers"], info["hidden_size"]
    layers = sorted(set([0, L_total // 4, L_total // 2, 3 * L_total // 4, L_total - 1]))

    text = open(args.prompt[1:]).read().splitlines()[0]
    text = text.split("\t", 1)[1] if "\t" in text else text
    pids = tok(text, return_tensors="pt").input_ids.cuda()
    P = pids.shape[1]
    with torch.inference_mode():
        base_prompt = model.get_input_embeddings()(pids)[0]

    check_mem(args.min_mem_gb + 1, "baseline generation")
    Eref = base_prompt[None].expand(2, -1, -1)
    gen = free_generate(model, Eref, args.max_new_tokens)
    identical = bool(torch.equal(gen[0], gen[1]))
    assert identical, "r=0 duplicate rows differ"
    gen_ids = gen[0:1]
    T = gen_ids.shape[1]
    del Eref, gen
    torch.cuda.empty_cache()
    print(f"baseline: P={P} T={T} bit-identity OK", flush=True)

    Efull = full_embeddings(model, pids, gen_ids)
    capture = Capture(model, layers)
    margins, top1, top2, h0 = baseline_pass(model, Efull, P, T, layers, capture)
    wdiff = wdiff_tensor(model, top1, top2)

    # margin spectrum
    order = np.argsort(margins)
    print("margins: min={:.4f} 5 smallest (step,val): {}".format(
        margins.min(), [(int(i), round(float(margins[i]), 4)) for i in order[:5]]), flush=True)
    for thr in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        print(f"  margins < {thr}: {int((margins < thr).sum())}/{T}", flush=True)

    # linearity over radii for a few directions
    radii = [float(x) for x in args.radii.split(",")]
    K = args.num_directions
    simplex, subspace = build_direction_blocks(K, P, D, block=max(K, 2), seed=args.seed)
    nl = len(layers)
    dh_norm = np.zeros((len(radii), K, nl, T), dtype=np.float32)
    dm = np.zeros((len(radii), K, T), dtype=np.float32)
    for ri, r in enumerate(radii):
        E = perturb_rows(Efull, P, D, simplex, subspace, list(range(K)), r)
        with torch.inference_mode():
            res = teacher_batch(model, E, P, T, layers, h0, wdiff, capture)
        dh_norm[ri] = res["dh_norm"]
        dm[ri] = res["dm"]
        del E, res
        torch.cuda.empty_cache()

    # dh_last (final norm output) per radius, median over directions
    print("\nlinearity check (last layer, median over directions, t in [T/4, T)):", flush=True)
    li = layers.index(L_total - 1)
    med = np.median(dh_norm[:, :, li, T // 4:], axis=(1, 2))
    base_floor = np.float32(0)
    for r, m in zip(radii, med):
        print(f"  r={r:<8g} dh={m:.4e}  dh/r={m / r:.4e}", flush=True)
    ratios = med / np.asarray(radii)
    lo, hi = ratios[0], ratios.max()
    print(f"  dh/r spread across radii: {lo:.3e} .. {hi:.3e} (ratio {hi / max(lo, 1e-30):.2f})", flush=True)

    # signed threshold preview at the largest radius below nonlinearity
    r_star = np.zeros((len(radii), K), dtype=np.float32)
    for ri, r in enumerate(radii):
        rc = margins[None, :] * r / np.maximum(-dm[ri], 1e-30)
        r_star[ri] = rc.min(axis=1)
    print("\nsigned r_* per direction (rows = radii):", flush=True)
    for ri, r in enumerate(radii):
        print(f"  r={r:<8g} r_*: min={r_star[ri].min():.3e} med={np.median(r_star[ri]):.3e} "
              f"max={r_star[ri].max():.3e}  (finite where dm<0: {(dm[ri] < 0).any(axis=1).sum()}/{K})", flush=True)

    np.savez(os.path.join(args.out, "calib.npz"), margins=margins, top1=top1, top2=top2,
             radii=np.asarray(radii), dh_norm=dh_norm, dm=dm, layers=np.asarray(layers),
             gen_ids=gen_ids[0].cpu().numpy(), prompt_ids=pids[0].cpu().numpy(),
             r_star=r_star, simplex=simplex, subspace=subspace)
    save_json(os.path.join(args.out, "meta.json"), dict(
        model=args.model, dtype=args.dtype, window=args.window, prompt_len=int(P),
        max_new_tokens=int(T), radii=radii, num_directions=K, seed=args.seed,
        bit_identity=identical, arch=info["arch"],
        elapsed_sec=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
