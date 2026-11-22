"""Free-run validation of predicted T(r,u) for large models (batched, in-batch r=0).

Same outputs as run_thresholds.py: obs[ri, di] = first mismatch step (-1 = never),
pred from the a1 run, plus full token sequences for post-flip analysis.
Batched greedy generation with KV cache: on the GB10 one generate() call for
--chunk rows costs T weight sweeps total, so bigger chunks are strictly better.
"""
import argparse
import json
import os

import numpy as np
import torch

from chaos_llm.big_engine import free_generate, load_big, save_json, set_window


def first_mismatch(ref, row):
    n = min(len(ref), len(row))
    mm = np.where(row[:n] != ref[:n])[0]
    return int(mm[0]) if len(mm) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--run", required=True)
    ap.add_argument("--radii", required=True)
    ap.add_argument("--dirs", default="", help="comma list of direction indices; default all")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0 = use the a1 run length")
    ap.add_argument("--min-mem-gb", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(os.path.join(args.run, "a1.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    simplex = d["simplex"]
    subspace = d["subspace"]
    P = int(meta["prompt_len"])
    T = int(args.max_new_tokens) or int(meta["max_new_tokens"])
    K = simplex.shape[0]
    radii = [float(x) for x in args.radii.split(",")]
    dirs = [int(x) for x in args.dirs.split(",")] if args.dirs else list(range(K))

    model, tok, _mp, info = load_big(args.model, dtype=args.dtype)
    if meta.get("window"):
        set_window(model, meta["window"])
    D = info["hidden_size"]
    pids = tok(meta["prompt"], return_tensors="pt").input_ids.cuda()
    assert pids.shape[1] == P, "prompt tokenisation differs from the a1 run"
    with torch.inference_mode():
        base = model.get_input_embeddings()(pids)[0]
    sub_t = torch.as_tensor(subspace, device="cuda", dtype=torch.long)

    obs = np.full((len(radii), K), -1, dtype=np.int32)
    agreement = np.full((len(radii), K), np.nan, dtype=np.float32)
    ref_ids = None
    all_gen = {}
    for ri, r in enumerate(radii):
        gen_rows = []
        for start in range(0, len(dirs), args.chunk - 1):
            batch_dirs = dirs[start:start + args.chunk - 1]
            plan = [dict(kind="ref", di=-1)] + [dict(kind="pert", di=di) for di in batch_dirs]
            B = len(plan)
            E = base[None].expand(B, -1, -1).clone()
            for j, row in enumerate(plan):
                if row["kind"] == "pert":
                    delta = torch.zeros(P * D, device="cuda", dtype=torch.float32)
                    delta.index_copy_(0, sub_t, torch.as_tensor(simplex[row["di"]], device="cuda") * r)
                    E[j] += delta.view(P, D)
            gen = free_generate(model, E, T).float().cpu().numpy()
            ref = gen[0]
            if ref_ids is None:
                ref_ids = ref.copy()
            elif not np.array_equal(ref_ids, ref):
                print(f"WARNING: r=0 reference differs from the first chunk at r={r}", flush=True)
            for j, row in enumerate(plan):
                if row["kind"] == "pert":
                    di = row["di"]
                    obs[ri, di] = first_mismatch(ref, gen[j])
                    n = min(len(ref), len(gen[j]))
                    agreement[ri, di] = float((gen[j][:n] == ref[:n]).mean())
            gen_rows.append(np.vstack([gen[1:]]))
            print(f"r={r:g} dirs {batch_dirs[0]}-{batch_dirs[-1]}: "
                  f"T={[(di, int(obs[ri, di])) for di in batch_dirs]}", flush=True)
            del E, gen
            torch.cuda.empty_cache()
        all_gen[f"r{ri}"] = np.vstack(gen_rows)

    pred = d["pred_T"]
    mradii = np.asarray(meta["radii"])
    pred_rows = np.array([int(np.argmin(np.abs(mradii - r))) for r in radii])
    np.savez(os.path.join(args.out, "free_run.npz"), obs=obs, agreement=agreement,
             pred=pred[pred_rows], radii=np.asarray(radii),
             dirs=np.asarray(dirs), gen_ids=np.array(list(all_gen.values()), dtype=object))
    flip = obs >= 0
    save_json(os.path.join(args.out, "summary.json"), dict(
        run=args.run, radii=radii, dirs=dirs, obs=obs.tolist(),
        flip_frac=flip.mean(axis=1).tolist(),
        mean_flip_step=[obs[ri][flip[ri]].mean().round(1) if flip[ri].any() else None
                        for ri in range(len(radii))],
        mean_postflip_agreement=[float(np.nanmean(agreement[ri])) for ri in range(len(radii))],
        n_dirs=len(dirs), window=meta.get("window"), prompt_len=P, T=T))
    print("saved", args.out)


if __name__ == "__main__":
    main()
