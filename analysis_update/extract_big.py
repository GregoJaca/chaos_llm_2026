import numpy as np, json, os

BASE = os.path.expanduser("~/LLM/chaos_llm_2026")
RUNS = {
    "Qwen3-14B": "results_big/qwen314/a1_inter_w0",
    "Phi-4": "results_big/phi4/a1_inter_w0",
    "DeepSeek-14B": "results_big/ds14/a1_inter_w100",
}
out = {}
for name, rel in RUNS.items():
    p = os.path.join(BASE, rel, "a1.npz")
    z = np.load(p, allow_pickle=True)
    meta = json.load(open(os.path.join(BASE, rel, "meta.json")))
    radii = z["radii"].astype(float)
    r_target = 0.3 if name == "Phi-4" else 0.1
    ri = int(np.argmin(np.abs(radii - r_target)))
    dh = z["dh_norm"].astype(float)          # (r, dir, layer, token)
    hn = z["norm_h0"].astype(float)          # (layer, token)
    nL, T = hn.shape
    med = np.median(dh[ri], axis=0)          # (layer, token)
    rel_depth = med[:, 50] / hn[:, 50]
    rel_depth = rel_depth / max(rel_depth[1], 1e-30)
    resp_last = med[-1] / radii[ri]
    out[name] = dict(
        r=float(radii[ri]), nlayer=int(nL), T=int(T),
        rel_depth=rel_depth.tolist(),
        resp_last=resp_last.tolist(),
        norm_h0_mid=float(np.median(hn[:, 50])),
        a1=float(resp_last[1]),
        amax=float(resp_last.max()),
    )
    print(name, "r", radii[ri], "layers", nL, "rel_depth end", rel_depth[-1],
          "resp a1", resp_last[1], "max", resp_last.max(), flush=True)

json.dump(out, open(os.path.join(BASE, "results_big_small", "big_a1_summary.json"), "w"), indent=1)
print("saved")

