"""Dump keys, shapes, dtypes and small summaries of every npz we use."""
import numpy as np

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
FILES = [
    r"w100_layers\a1.npz",
    r"state.npz",
    r"state_mid.npz",
    r"fr_w100_large.npz",
    r"nonnorm.npz",
    r"a1.npz",
]

for rel in FILES:
    path = BASE + "\\" + rel
    print("=" * 70)
    print("FILE:", rel)
    z = np.load(path, allow_pickle=True)
    for k in z.files:
        a = z[k]
        print(f"  {k:12s} shape={str(a.shape):22s} dtype={a.dtype}")
        if a.ndim <= 1 and a.size <= 40:
            print("      values:", np.array2string(np.asarray(a).ravel(), precision=4, max_line_width=100))
        elif a.ndim == 2 and min(a.shape) <= 30 and a.size < 900:
            pass  # skip heavy print
        # quick stats for numeric
        try:
            if np.issubdtype(a.dtype, np.number):
                print(f"      min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} mean={np.nanmean(a):.4g}")
        except Exception as e:
            print("      (stats err)", e)
    z.close()
