import numpy as np, json, os
base = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
for rel in [r"w100_layers\a1.npz", r"w200\a1.npz", r"w500\a1.npz",
            r"window_fixed\w100\a1.npz", r"window_fixed\w1000\a1.npz"]:
    z = np.load(os.path.join(base, rel), allow_pickle=True)
    print("=== ", rel)
    for k in z.files:
        a = z[k]
        extra = ""
        if a.ndim <= 1 and a.size < 40:
            extra = "  vals=" + np.array2string(a.ravel()[:40], precision=4, max_line_width=200)
        print(f"  {k:12s} shape={str(a.shape):24s} dtype={a.dtype}{extra}")

for rel in [r"fr_w100_large.npz", r"fr_w100_mid.npz", r"fr_childhood.npz"]:
    z = np.load(os.path.join(base, rel), allow_pickle=True)
    print("=== ", rel)
    for k in z.files:
        a = z[k]
        extra = ""
        if a.ndim <= 1 and a.size < 40:
            extra = "  vals=" + np.array2string(a.ravel()[:40], precision=4, max_line_width=200)
        print(f"  {k:12s} shape={str(a.shape):24s} dtype={a.dtype}{extra}")

for rel in ["free_large.json", "free_w100_large.json", "free_w200.json", "free_w500.json", "free_w100_layers_large.json"]:
    p = os.path.join(base, rel)
    with open(p) as f:
        d = json.load(f)
    print("=== ", rel, type(d))
    if isinstance(d, dict):
        for k, v in d.items():
            print("   ", k, type(v), (str(v)[:200] if not isinstance(v, list) else f"list len={len(v)} first={str(v[:2])[:200]}"))
    elif isinstance(d, list):
        print("   list len", len(d), "first:", str(d[:1])[:400])
