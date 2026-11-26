import numpy as np, os, json

BASE = os.path.expanduser("~/LLM/chaos_llm_2026")
RUNS = {
    "Mamba-130m": "results/ctrl/mamba130/a1i_txt",
    "Pythia-410m": "results/ctrl/pythia410/a1i_txt",
    "randQwen": "results/ctrl/randqwen_g1.0/a1i_txt",
}
for name, rel in RUNS.items():
    p = os.path.join(BASE, rel, "a1.npz")
    if not os.path.exists(p):
        print("missing", p); continue
    z = np.load(p, allow_pickle=True)
    meta = json.load(open(os.path.join(BASE, rel, "meta.json")))
    radii = z["radii"].astype(float)
    dh = z["dh_norm"].astype(np.float32)
    hn = z["norm_h0"].astype(np.float32)
    med = np.median(dh, axis=1)
    lo, hi = np.percentile(dh, [25, 75], axis=1)
    np.savez(os.path.join(BASE, "results_big_small", f"ctrl_{name.replace('-', '_')}.npz"),
             radii=radii, layers=z["layers"].astype(int), dh_med=med, dh_lo=lo, dh_hi=hi,
             norm_h0=hn, margins=z["margins"].astype(float),
             T=int(dh.shape[-1]), n_dirs=int(dh.shape[1]))
    print(name, dh.shape, "r", radii, "layers", z["layers"][:5], "...")
print("done")
