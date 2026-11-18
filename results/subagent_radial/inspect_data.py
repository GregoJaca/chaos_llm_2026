"""Inspect the local data files for the radial-drift explanation task.

Prints the quantities needed to design the figures/tests. Read-only.
"""
import json
import numpy as np

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"

print("=" * 78)
print("w100_layers/a1.npz")
print("=" * 78)
d = np.load(BASE + r"\w100_layers\a1.npz", allow_pickle=True)
radii = d["radii"]
layers = d["layers"]
norm_h0 = d["norm_h0"]          # (8, 300)
dh_norm = d["dh_norm"]          # (12, 20, 8, 300)
cos = d["cos"]                  # (12, 20, 8, 300)
dm = d["dm"]                    # (12, 20, 300)
dh_last = d["dh_last"]          # (12, 3, 300, 1536)
margins = d["margins"]

print("radii:", radii)
print("layers:", layers)
print()

# median state norm per layer
print("median over 300 tokens of ||h_l|| per layer:")
for i, l in enumerate(layers):
    print(f"  layer {l:2d}: median={np.median(norm_h0[i]):10.4g}   "
          f"q10={np.quantile(norm_h0[i], 0.1):10.4g}  q90={np.quantile(norm_h0[i], 0.9):10.4g}")
print()

# dh_norm per layer for a few radii (median over dirs & tokens)
print("median ||dh_l|| (over 20 dirs, 300 tokens) per layer, per radius:")
for ri in range(len(radii)):
    row = [np.median(dh_norm[ri, :, li, :]) for li in range(len(layers))]
    print(f"  r={radii[ri]:<8g}: " + "  ".join(f"{v:9.4g}" for v in row))
print()

# scaling check: is dh_norm linear in r?  ratio dh(r)/dh(r=0.01)
print("linearity check: median dh_norm(r) / median dh_norm(r=1e-2), per layer:")
ref = np.array([np.median(dh_norm[4, :, li, :]) for li in range(len(layers))])
for ri in range(len(radii)):
    row = np.array([np.median(dh_norm[ri, :, li, :]) for li in range(len(layers))])
    print(f"  r={radii[ri]:<8g}: " + "  ".join(f"{v:9.4g}" for v in row / ref))
print()

# cos stats per layer for r=0.01 (idx 4?) -> find index of 0.01
ri01 = int(np.argmin(np.abs(radii - 0.01)))
print(f"index of r=0.01: {ri01}, value {radii[ri01]}")
print("cos(perturbed, baseline) stats at r=0.01, per layer:")
for li, l in enumerate(layers):
    c = cos[ri01, :, li, :].ravel()
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    print(f"  layer {l:2d}: cos q50={np.median(c):.6f} q10={np.quantile(c,0.1):.6f} min={c.min():.6f}"
          f" | angle(deg) q50={np.median(ang):.4f} q90={np.quantile(ang,0.9):.4f} max={ang.max():.3f}")
print()

print("cos min per radius (over dirs/tokens/layers):")
for ri in range(len(radii)):
    c = cos[ri].ravel()
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    print(f"  r={radii[ri]:<8g}: cos min={c.min():.4f} | angle q50={np.median(ang):.4f} deg  q99={np.quantile(ang,0.99):.3f} max={ang.max():.2f}")
print()

# dh_last: figure out which of the 20 simplex directions were saved
print("dh_last: match saved vectors to dh_norm direction indices at layer 27.")
n_last = np.linalg.norm(dh_last, axis=-1)   # (12, 3, 300)
for j in range(3):
    # compare per (radius, token) against each of 20 dirs
    best, err = None, np.inf
    for di in range(20):
        cand = dh_norm[:, di, 7, :]          # (12, 300)
        e = np.median(np.abs(n_last[:, j, :] - cand) / (cand + 1e-12))
        if e < err:
            err, best = e, di
    print(f"  dh_last dir slot {j}: best match = simplex dir {best}, median rel err={err:.3g}")
print()

print("margins: q10/q50/q90:", np.quantile(margins, [0.1, 0.5, 0.9]))
print("dm stats at r=0.01:", np.quantile(dm[ri01], [0.1, 0.5, 0.9]))
print()

print("=" * 78)
print("depth/depth.npz")
print("=" * 78)
dd = np.load(BASE + r"\depth\depth.npz", allow_pickle=True)
lam = dd["lam"]          # (8 tokens, 28 layers, 8 exponents)
local_gain = dd["local_gain"]  # (8, 28)
h_depth = dd["h_depth"]  # (8,)
toks = dd["tokens"]
lay = dd["layers"]
print("tokens:", toks, "layers:", lay[:5], "...", lay[-3:])
print("h_depth:", h_depth)
print("local_gain per layer (token 0):", np.round(local_gain[0], 3))
print("local_gain per layer (median over tokens):", np.round(np.median(local_gain, axis=0), 3))
print("lam[:, 0, :] token 0:", lam[0, 0])
print("lam[:, 5, :] token 0:", lam[0, 5])
print("lam[:, 27, :] token 0:", lam[0, 27])
print()

# Candidate "two numbers": absolute sum of positive exponents vs relative
pos = np.maximum(lam, 0.0).sum(axis=1)          # (8 tokens, 28 layers) summed over exponents
abs_sum = pos.sum(axis=1)                        # total log growth per token
print("per-token sum over layers of sum_k max(lam_k,0):", np.round(abs_sum, 3))
print("  -> per-layer (divide 27):", np.round(abs_sum / 27, 4))
print("h_depth vs exp(abs_sum/27)?  exp:", np.round(np.exp(abs_sum / 27), 3))
print("h_depth vs exp(abs_sum)?     exp:", np.round(np.exp(abs_sum), 3))

# relative: subtract state growth per layer = log(local_gain)?
rel = np.maximum(lam - np.log(local_gain)[:, :, None], 0.0)
rel_sum = rel.sum(axis=(1, 2))
print("per-token sum of max(lam - log local_gain, 0):", np.round(rel_sum, 3))
print("h_depth vs exp(-rel_sum/27)?:", np.round(np.exp(-rel_sum / 27), 3))
print("h_depth vs exp((abs_sum-rel_sum)/27):", np.round(np.exp((abs_sum - rel_sum) / 27), 3))
print("local_gain product per token:", np.round(np.prod(local_gain, axis=1), 3))
print("log(prod local_gain):", np.round(np.log(np.prod(local_gain, axis=1)), 3))
print()

print("=" * 78)
print("nonnorm.npz")
print("=" * 78)
dn = np.load(BASE + r"\nonnorm.npz", allow_pickle=True)
print("tokens:", dn["tokens"], "sel_layers:", dn["sel_layers"])
print("full_smax median per sel layer:", np.median(dn["full_smax"], axis=0))
print("full_rho  median per sel layer:", np.median(dn["full_rho"], axis=0))
print("full_kappa median per sel layer:", np.median(dn["full_kappa"], axis=0))
print("angles (principal, deg) q50 per layer-gap:",
      np.round(np.degrees(np.median(dn["angles"], axis=0)), 1))
print()

print("=" * 78)
print("state.npz / state_mid.npz (keys only, to see if full 28-layer norms exist)")
print("=" * 78)
try:
    ds = np.load(BASE + r"\state.npz", allow_pickle=True)
    print("state.npz keys:", {k: ds[k].shape for k in ds.files})
except Exception as e:
    print("state.npz:", e)
try:
    dsm = np.load(BASE + r"\state_mid.npz", allow_pickle=True)
    print("state_mid.npz keys:", {k: dsm[k].shape for k in dsm.files})
except Exception as e:
    print("state_mid.npz:", e)
print()
try:
    import torch
    print("torch available:", torch.__version__)
except Exception as e:
    print("torch NOT available:", e)
