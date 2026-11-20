"""GPU precompute for the correlation-dimension (D2) program.

For each trajectory npz produced by collect_trajectories_big, computes the full
pairwise Euclidean distance matrix on the GPU (chunked, fp32) and saves it as
.npy, plus a compact summary (N, norms, repetition-screen numbers). The actual
D2/RQA estimators then run locally on the laptop from these matrices — no model
needed there, no GPU conflict here.

Usage:
  PYTHONPATH=src python scripts_spark/d2_distances.py \
      --run-dir results_big/ds14/traj_w4096 --states h_last --out results_big/ds14/d2
"""
import argparse
import glob
import json
import os

import numpy as np
import torch


@torch.inference_mode()
def pairwise_gpu(x, chunk=512):
    """Full [N,N] Euclidean distances on the GPU, chunked over rows."""
    n = x.shape[0]
    xt = torch.as_tensor(x, device="cuda", dtype=torch.float32)
    out = torch.empty((n, n), device="cuda", dtype=torch.float32)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        d = torch.cdist(xt[i:j, None, :], xt[None, :, :])  # [c, n]
        out[i:j] = d
    return out.cpu().numpy()


def repetition_screen(h):
    """Relative distance of each state to its nearest non-adjacent neighbour.
    Near-exact loops give values ~1e-3..1e-5; healthy trajectories ~1e-2+."""
    n = h.shape[0]
    hh = h[n // 4:]  # skip the prompt-conditioned head
    m = pairwise_gpu(hh)
    np.fill_diagonal(m, np.inf)
    # Theiler window 50: exclude |i-j| < 50
    idx = np.arange(m.shape[0])
    for w in range(1, 50):
        np.fill_diagonal(m[w:], np.inf)
        np.fill_diagonal(m[:, w:], np.inf)
    nearest = m.min(axis=1)
    scale = np.median(np.linalg.norm(hh - hh.mean(axis=0), axis=1)) + 1e-30
    return float(np.median(nearest) / scale), float(np.quantile(nearest, 0.05) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--states", default="h_last")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-n", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.run_dir, "*.npz")))
    summary = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)
        if args.states not in d:
            print(f"skip {name}: no {args.states}", flush=True)
            continue
        h = d[args.states][:args.max_n].astype(np.float32)
        n = h.shape[0]
        outp = os.path.join(args.out, f"dist_{name}_{args.states}.npy")
        if not os.path.exists(outp):
            m = pairwise_gpu(h)
            np.save(outp, m.astype(np.float32))
        else:
            m = np.load(outp)
        rel, p05 = repetition_screen(h)
        norms = np.linalg.norm(h, axis=1)
        rec = dict(name=name, n=n, dim=h.shape[1],
                   norm_median=float(np.median(norms)),
                   norm_growth=float(norms[-500:].mean() / max(norms[:500].mean(), 1e-30)),
                   rel_nearest=rel, rel_nearest_p05=p05)
        summary.append(rec)
        print(name, rec, flush=True)
        del m
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
