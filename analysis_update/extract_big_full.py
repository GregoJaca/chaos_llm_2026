import numpy as np, json, os

BASE = os.path.expanduser("~/LLM/chaos_llm_2026")
OUT = os.path.join(BASE, "results_big_small")
os.makedirs(OUT, exist_ok=True)
RUNS = {
    "Qwen3-14B": [("inter", "results_big/qwen314/a1_inter_w0"),
                  ("childhood", "results_big/qwen314/a1_childhood_w0"),
                  ("cyber", "results_big/qwen314/a1_cyber_w0")],
    "Phi-4": [("inter", "results_big/phi4/a1_inter_w0"),
              ("childhood", "results_big/phi4/a1_childhood_w0"),
              ("cyber", "results_big/phi4/a1_cyber_w0")],
    "DeepSeek-14B": [("inter", "results_big/ds14/a1_inter_w100"),
                     ("childhood", "results_big/ds14/a1_childhood_w100"),
                     ("cyber", "results_big/ds14/a1_cyber_w100")],
}

for name, runs in RUNS.items():
    for tag, rel in runs:
        p = os.path.join(BASE, rel, "a1.npz")
        if not os.path.exists(p):
            print("missing", p); continue
        d = np.load(p, allow_pickle=True)
        meta = json.load(open(os.path.join(BASE, rel, "meta.json")))
        radii = d["radii"].astype(float)
        ri = int(np.argmin(np.abs(radii - 0.1)))
        r0 = float(radii[ri])
        dh = d["dh_norm"][ri].astype(np.float32)
        hn = d["norm_h0"].astype(np.float32)
        dm = d["dm"][ri].astype(np.float32)
        m = d["margins"].astype(np.float32)
        rc = d["r_c"][ri].astype(np.float64)
        rc[:, m <= 0.02] = np.inf
        j = np.argmin(rc, axis=1)
        rstar = rc[np.arange(rc.shape[0]), j]
        tstar = j.astype(np.int32)
        med = np.median(dh, axis=0)
        lo, hi = np.percentile(dh, [25, 75], axis=0)
        relmed = med / np.maximum(hn, 1e-30)
        h0 = d["h0_last"].astype(np.float32)
        dhl = d["dh_last"][0].astype(np.float32)
        cosang = (dhl * h0[None]).sum(-1) / (
            np.linalg.norm(dhl, axis=-1) * np.linalg.norm(h0, axis=-1)[None] + 1e-30)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1))).astype(np.float32)
        G_u = dh[:, -1, 50] / r0
        np.savez(os.path.join(OUT, f"big_{name.replace('-', '_')}_{tag}.npz"),
                 radii=np.asarray([r0]), layers=d["layers"].astype(np.int32),
                 dh_med=med, dh_lo=lo, dh_hi=hi, norm_h0=hn, rel_med=relmed,
                 margins=m, dm=dm, rstar=rstar, tstar=tstar, mstar=m[j],
                 G_u=G_u, angle=ang, cosang=cosang,
                 prompt_len=int(len(d["prompt_ids"])), window=meta.get("window"),
                 n_dirs=int(dh.shape[0]), n_angle=int(dhl.shape[0]),
                 n_layer=int(dh.shape[1]), T=int(dh.shape[2]))
        print(name, tag, "median r*", float(np.median(rstar[np.isfinite(rstar)])),
              "angle", float(np.nanmedian(ang)), flush=True)
print("done")
