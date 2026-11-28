"""Shared discovery/allocation helpers for the closed-loop (soft loop) analyses.

A "pair" is a loop run and a replay run recorded in the SAME directory (same
reference orbit, same dtype, same temperature).  Ratios are only ever formed
inside a pair.  Different pairs of the same model/T may come from different
chunks (a chunk = one run with --dirs K); the reference row is unperturbed, so
all chunks of one campaign share the same orbit and their directions can be
pooled.
"""
import os
import glob
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TAGS = {
    "DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25", "Qwen3-4B": "qwen34",
    "Qwen3.5-2B": "qwen35", "Qwen3.5-9B": "qwen35_9b", "Phi-4-mini": "phi",
    "DeepSeek-14B": "ds14", "Qwen3-14B": "qwen314", "Phi-4": "phi4",
    "Mamba-130m": "mamba", "Pythia-410m": "pythia", "randQwen": "randqwen",
}
LEGACY = {"Qwen2.5-3B": "qwen25_3b", "Qwen3-4B": "qwen3_4b", "Qwen3.5-2B": "qwen3_5",
          "Phi-4-mini": "phi4mini", "Mamba-130m": "mamba130", "Pythia-410m": "pythia410"}
ROOTS = ["results/longloop_v3b", "results/longloop_v3", "results/longloop", "results/softloop"]


def model_tags(name):
    return list(dict.fromkeys([TAGS[name]] + ([LEGACY[name]] if name in LEGACY else [])))


def find_dirs(tag_list, T="T1.0"):
    out = []
    for root in ROOTS:
        for tag in tag_list:
            for d in glob.glob(os.path.join(HERE, root, tag, T)):
                if os.path.exists(os.path.join(d, "loop", "softloop.npz")) or \
                   os.path.exists(os.path.join(d, "softloop.npz")):
                    out.append(d)
                # one level deeper: chunked campaigns store k1/, k2/, ...
                for sub in glob.glob(os.path.join(d, "*")):
                    if os.path.exists(os.path.join(sub, "loop", "softloop.npz")):
                        out.append(sub)
    return sorted(set(out))


def loop_path(d):
    p = os.path.join(d, "loop", "softloop.npz")
    return p if os.path.exists(p) else os.path.join(d, "softloop.npz")


def replay_path(d):
    p = os.path.join(d, "replay", "softloop.npz")
    return p if os.path.exists(p) else None


def meta_layers(d):
    import json
    p = os.path.join(d, "meta.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    lay = m.get("layers")
    return list(lay) if lay else None


def meta_layers_of(npz_path):
    """layers list from meta.json next to the npz, or one level up."""
    import json
    d = os.path.dirname(npz_path)
    for cand in (d, os.path.dirname(d)):
        m = os.path.join(cand, "meta.json")
        if os.path.exists(m):
            lay = json.load(open(m)).get("layers")
            if lay:
                return list(lay)
    return None


def pairs(name, T="T1.0", require_layers=False):
    """[(loop, replay)] for this model and temperature, all complete pairs."""
    out = []
    for d in find_dirs(model_tags(name), T):
        pr = replay_path(d)
        if pr is None:
            continue
        pl = loop_path(d)
        if require_layers:
            lay = meta_layers(d)
            if not lay or len(lay) < 2:
                continue
        out.append((pl, pr))
    return out


def gather(name, T="T1.0", layer=-1, layer_num=None, prefer_root=None):
    """pooled per-direction curves (a_closed, a_replay), both normalised at t=1.

    Pairs are pooled **only within one campaign root**, because different roots
    differ in dtype and horizon (fp16/800 steps vs fp32/3000 steps) and their
    orbits are not the same realisation.  We pick the root with the most
    directions (or `prefer_root` when given).
    """
    by_root = {}
    for pl, pr in pairs(name, T):
        parts = pl.replace("\\", "/").split("/")
        root = parts[parts.index("results") + 1] if "results" in parts else "?"
        by_root.setdefault(root, []).append((pl, pr))

    def total(pairs_):
        n, s = 0, 0
        for pl, pr in pairs_:
            try:
                z = np.load(pl, allow_pickle=True)["dh"]
                n += z.shape[0]
                s = max(s, z.shape[-1])
            except Exception:
                pass
        return n, s

    stats = {r: total(v) for r, v in by_root.items()}
    # never fall back to the 600-step legacy campaign for the main curves: keep
    # roots with a horizon of at least 800 steps, then take the most directions.
    good = {r: st for r, st in stats.items() if st[1] >= 800}
    if not good:
        good = stats
    if prefer_root and prefer_root in good:
        chosen = prefer_root
    elif good:
        chosen = max(good, key=lambda r: (good[r][0], good[r][1]))
    else:
        return None, None, []

    acs, ars, used = [], [], []
    for pl, pr in by_root[chosen]:
        zc = np.load(pl, allow_pickle=True)
        zr = np.load(pr, allow_pickle=True)
        lay = meta_layers_of(pl) or []
        if layer_num is not None:
            if layer_num not in lay:
                continue
            ic = lay.index(layer_num)
            layr = meta_layers_of(pr) or lay
            if layer_num not in layr:
                continue
            ir = layr.index(layer_num)
        else:
            nlc, nlr = zc["dh"].shape[1], zr["dh"].shape[1]
            if nlc == 0 or nlr == 0:
                continue
            ic, ir = -1, -1
        ac = zc["dh"][:, ic, :].astype(float)
        ar = zr["dh"][:, ir, :].astype(float)
        n = min(ac.shape[1], ar.shape[1])
        ac, ar = ac[:, :n], ar[:, :n]
        ac = ac / np.maximum(ac[:, [1]], 1e-30)
        ar = ar / np.maximum(ar[:, [1]], 1e-30)
        ok = np.all(np.isfinite(ac), 1) & np.all(np.isfinite(ar), 1)
        if ok.sum() == 0:
            continue
        acs.append(ac[ok]); ars.append(ar[ok]); used.append(pl)
    if not acs:
        return None, None, []
    L = max(a.shape[1] for a in acs)
    def pad(a):
        out = np.full((a.shape[0], L), np.nan)
        out[:, :a.shape[1]] = a
        return out
    return (np.concatenate([pad(a) for a in acs], 0),
            np.concatenate([pad(a) for a in ars], 0), used)


def median_iqr(x, axis=0):
    return np.median(x, axis=axis), np.percentile(x, [25, 75], axis=axis)
