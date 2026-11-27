"""Table behind the depth story: what each metric says, on the same models."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *

CTRL = {"Mamba-130m": "results_big_small/ctrl_Mamba_130m.npz",
        "Pythia-410m": "results_big_small/ctrl_Pythia_410m.npz",
        "randQwen": "results_big_small/ctrl_randQwen.npz"}
ORDER = all_names() + list(CTRL)


def load_all(n, r=0.1):
    """returns dh (D,L,T) med over dirs, lo, hi, hn (L,T), layers"""
    if n in SMALL:
        z = a1(n)
        ri = int(np.argmin(np.abs(z["radii"] - r)))
        # recompute percentiles of the RELATIVE response
        dh = z["dh"][ri]                      # (dir, layer, T)
        hn = z["hn"]                          # (layer, T)
        rel = dh / np.maximum(hn, 1e-30)[None]
        med = np.median(dh, axis=0); lo, hi = np.percentile(dh, [25, 75], axis=0)
        rmed = np.median(rel, axis=0); rlo, rhi = np.percentile(rel, [25, 75], axis=0)
        return dict(dh=med, lo=lo, hi=hi, rel=rmed, rlo=rlo, rhi=rhi, hn=hn,
                    layers=z["layers"].astype(float), T=dh.shape[-1])
    if n in BIG:
        z = big(n)
        dh = z["dh_med"].astype(float); lo = z["dh_lo"].astype(float); hi = z["dh_hi"].astype(float)
        hn = z["norm_h0"].astype(float)
        rel = dh / np.maximum(hn, 1e-30); rlo = lo / np.maximum(hn, 1e-30); rhi = hi / np.maximum(hn, 1e-30)
        return dict(dh=dh, lo=lo, hi=hi, rel=rel, rlo=rlo, rhi=rhi, hn=hn,
                    layers=z["layers"].astype(float), T=dh.shape[-1])
    z = load(CTRL[n])
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    dh = z["dh_med"][ri].astype(float); lo = z["dh_lo"][ri].astype(float); hi = z["dh_hi"][ri].astype(float)
    hn = z["norm_h0"].astype(float)
    rel = dh / np.maximum(hn, 1e-30)
    return dict(dh=dh, lo=lo, hi=hi, rel=rel, rlo=lo / np.maximum(hn, 1e-30),
                rhi=hi / np.maximum(hn, 1e-30), hn=hn,
                layers=z["layers"].astype(float), T=dh.shape[-1])


DEP = {"DeepSeek-1.5B": "results/depth2/deepseek/depth.npz",
       "DeepSeek-14B": "results_big_small/results_big/ds14/depth_w100/depth.npz",
       "Qwen2.5-3B": "results/depth2/qwen25_3b/depth.npz",
       "Qwen3-4B": "results/depth2/qwen3_4b/depth.npz",
       "Qwen3-14B": "results_big_small/results_big/qwen314/depth_w0/depth.npz",
       "Qwen3.5-2B": "results/depth2/qwen3_5/depth.npz",
       "Phi-4-mini": "results/depth2/phi4mini/depth.npz",
       "Phi-4": "results_big_small/results_big/phi4/depth_w0/depth.npz"}

print("%-14s %8s %8s %8s %8s %8s %8s %8s" % (
    "model", "rel(1)", "rel(mid)", "rel(L)", "relL/rel1", "|hL|/|h1|", "dhL/dh1", "cum_L"))
rows = {}
for n in ORDER:
    try:
        d = load_all(n)
    except Exception as e:
        print("%-14s ERR %s" % (n, e)); continue
    L = len(d["layers"])
    t = min(50, d["T"] - 1)
    # first saved layer with a nonzero response at this token
    l0 = 1 if L > 1 else 0
    rel = d["rel"][:, t]
    dh = d["dh"][:, t]
    hn = d["hn"][:, t]
    rows[n] = dict(rel1=float(rel[l0]), relmid=float(rel[L // 2]), relL=float(rel[-1]),
                   ratio_rel=float(rel[-1] / max(rel[l0], 1e-30)),
                   ratio_hn=float(hn[-1] / max(hn[l0], 1e-30)),
                   ratio_dh=float(dh[-1] / max(dh[l0], 1e-30)),
                   layers=[int(x) for x in d["layers"]])
    r = rows[n]
    print("%-14s %8.2e %8.2e %8.2e %8.2f %8.2f %8.2f" % (
        n, r["rel1"], r["relmid"], r["relL"], r["ratio_rel"], r["ratio_hn"], r["ratio_dh"]))

print("\n=== tangent map: cum_L, sum of log local gains, block-0 share")
for n, p in DEP.items():
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    lam = z["lam"].astype(float)                     # (token, layer)
    L = lam.shape[1]
    x = np.arange(L)
    cum = lam * (x + 1)[None, :]
    med = np.median(cum, axis=0)
    s = np.median(np.log(np.maximum(lam, 1e-30)), axis=0)   # log g per block
    tot = med[-1]
    print("%-14s cum_L=%.2f  sum_log_g=%.2f  block0_share=%.2f  g0=%.2f  g_mid=%.2f  g_last=%.2f" % (
        n, tot, s.sum(), (s[0] / s.sum()) if s.sum() else float("nan"),
        np.exp(s[0]), np.exp(np.median(s[1:len(s)//2 + 1])), np.exp(s[-1])))

json.dump(rows, open(os.path.join(HERE, "analysis_update", "tmp_depth_table.json"), "w"), indent=1)
