"""Screen long trajectories for near-exact repetition loops."""
import glob, os, sys
import numpy as np

ROOTS = ["results/long_traj", "results/long_traj2", "results/long_traj3", "results/long_traj4"]
rows = []
for root in ROOTS:
    for path in sorted(glob.glob(os.path.join(root, "*.npz"))):
        d = np.load(path)
        if "h_last" not in d:
            continue
        x = d["h_last"].astype(np.float64)
        T = x.shape[0]
        x = x - x.mean(0, keepdims=True)
        scale = float(np.median(np.linalg.norm(x, axis=1)))
        n = 1 << (2 * T - 1).bit_length()
        F = np.fft.rfft(x, n=n, axis=0)
        ac = np.fft.irfft((F * np.conj(F)).sum(1), n=n)[:T]
        sq = (x * x).sum(1)
        cs = np.concatenate([[0.0], np.cumsum(sq)])
        best = (9e9, -1)
        for P in range(5, min(260, T - 10)):
            # sum over t<T-P of ||x_t - x_{t+P}||^2
            s = cs[T - P] + (cs[T] - cs[P]) - 2 * ac[P]
            m = s / (T - P)
            d2 = np.sqrt(max(m, 0.0)) / scale
            if d2 < best[0]:
                best = (d2, P)
        rows.append((os.path.basename(path), best[1], best[0]))
rows.sort(key=lambda r: r[2])
print(f"{'trajectory':32s} {'period':>6s} {'rel.dist':>9s}  verdict")
for name, P, v in rows:
    verdict = "LOOP" if v < 0.10 else ("weak" if v < 0.25 else "ok")
    print(f"{name:32s} {P:6d} {v:9.3f}  {verdict}")
print("loops:", sum(1 for _, _, v in rows if v < 0.10), "/", len(rows))
