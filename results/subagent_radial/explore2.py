"""Second exploration: the decisive per-token checks before making final figures.

Q1  per-token depth growth of dh (is the x112 median growth an aggregation artifact?)
Q2  margins/w100.npz: same run as a1 w100_layers? (gen_ids, ||h27|| vs norm_h0)
Q3  exact radial/tangential split of dh at layer 27 using the real baseline h27
Q4  pairwise collinearity of the 3 saved dh_last directions
Q5  fp32 noise floor at r=1e-3 (per-entry linearity)
Q6  measured state angle vs the fully-tangential prediction, per layer and radius
Q7  per-token relative growth (dh27/dh3)/(s27/s3)
"""
import numpy as np

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_radial"

d = np.load(BASE + r"\w100_layers\a1.npz", allow_pickle=True)
radii = np.asarray(d["radii"]); layers = np.asarray(d["layers"])
norm_h0 = d["norm_h0"].astype(np.float64)   # (8, 300)
dh_norm = d["dh_norm"].astype(np.float64)   # (12, 20, 8, 300)
cos = d["cos"].astype(np.float64)           # (12, 20, 8, 300)
dm = d["dm"].astype(np.float64)             # (12, 20, 300)
dh_last = d["dh_last"].astype(np.float64)   # (12, 3, 300, 1536)
L = {int(l): i for i, l in enumerate(layers)}
np.set_printoptions(suppress=True, precision=4)

# ---------------------------------------------------------------- Q2 margins run
m = np.load(BASE + r"\margins\w100.npz", allow_pickle=True)
same = np.array_equal(m["gen_ids"][:300], d["gen_ids"])
print("Q2 margins gen_ids[:300] == a1 gen_ids:", same)
h27 = m["h27"][:300].astype(np.float64) if same else None
if h27 is not None:
    n27 = np.linalg.norm(h27, axis=-1)
    rel = np.abs(n27 - norm_h0[L[27]]) / norm_h0[L[27]]
    print("   ||h27(margins)|| vs norm_h0[27]: median rel diff = %.3e  max = %.3e"
          % (np.median(rel), rel.max()))

# ---------------------------------------------------------------- Q1/Q7 per-token growth
for ri, rname in [(2, 0.01), (6, 0.3)]:
    dh = np.median(dh_norm[ri], axis=0)          # (8, 300) median over 20 dirs
    s = norm_h0
    for base_l in [3, 7, 14]:
        g_abs = dh[L[27]] / dh[L[base_l]]
        g_scl = s[L[27]] / s[L[base_l]]
        g_rel = g_abs / g_scl
        ok = np.isfinite(g_abs) & (dh[L[base_l]] > 0)
        near = ok & (np.arange(300) < 164)
        far = ok & (np.arange(300) >= 164)
        print(f"Q1/Q7 r={rname}, base layer {base_l}:")
        print(f"   per-token dh27/dh_base : q10/50/90 = "
              f"{np.quantile(g_abs[ok],[.1,.5,.9])}")
        print(f"   per-token scale growth : q10/50/90 = "
              f"{np.quantile(g_scl[ok],[.1,.5,.9])}")
        print(f"   per-token RELATIVE     : q10/50/90 = "
              f"{np.quantile(g_rel[ok],[.1,.5,.9])}  near-tok med={np.median(g_rel[near]):.3f} far-tok med={np.median(g_rel[far]):.3f}")

# growth layer-by-layer for a mid token and the median profile
t_sel = 100
print("\nQ1 token t=100 dh by layer (r=0.01, med over dirs):",
      np.median(dh_norm[2, :, :, t_sel], axis=0).round(6))
print("   median-over-tokens dh by layer (r=0.01):",
      np.median(dh_norm[2], axis=(0, 2)).round(6))
print("   ||h|| median by layer:", np.median(norm_h0, axis=1).round(2))
print("   token t=100 ||h|| by layer:", norm_h0[:, t_sel].round(2))

# ---------------------------------------------------------------- Q6 state angle vs tangential prediction
print("\nQ6 measured state angle vs fully-tangential prediction (median over dirs,tokens):")
for ri in [2, 5, 6, 7, 8, 9, 10, 11]:
    r = radii[ri]
    row = []
    for li, l in enumerate(layers):
        n = norm_h0[li][None, :]              # (1,300)
        dv = dh_norm[ri, :, li, :]             # (20,300)
        C = cos[ri, :, li, :]
        theta_meas = np.degrees(np.arccos(np.clip(C, -1, 1)))
        theta_tan = np.degrees(np.arcsin(np.clip(dv / n, 0, 1)))
        m_meas = np.median(theta_meas)
        m_tan = np.median(theta_tan)
        row.append((l, m_meas, m_tan))
    print(f"  r={r:g}: " + " | ".join(f"L{l}: meas {a:.3f} tan-pred {b:.3f}" for l, a, b in row))

# fp32 floor for cos: 1 ulp at 1.0
floor_deg = np.degrees(np.sqrt(2 * 2 ** -23))
print(f"   cos fp32 floor angle ~ {floor_deg:.4f} deg")

# ---------------------------------------------------------------- Q3 exact split with real h27
if h27 is not None:
    print("\nQ3 exact radial/tangential split of dh at layer 27 (real baseline h):")
    e = h27 / np.linalg.norm(h27, axis=-1, keepdims=True)   # (300,1536)
    for ri in [2, 4, 6, 8, 9, 10, 11]:
        r = radii[ri]
        for j in range(3):
            v = dh_last[ri, j]                               # (300,1536)
            d_tot = np.linalg.norm(v, axis=-1)
            a = np.einsum("td,td->t", v, e)                  # radial component length (signed)
            b = np.sqrt(np.maximum(d_tot ** 2 - a ** 2, 0))  # tangential length
            ang = np.degrees(np.arctan2(b, np.abs(a)))
            if j == 0:
                print(f"  r={r:g} dir0: |radial|/total q10/50/90 = "
                      f"{np.quantile(np.abs(a)/d_tot,[.1,.5,.9]).round(3)}"
                      f"  tan/total q10/50/90 = {np.quantile(b/d_tot,[.1,.5,.9]).round(3)}"
                      f"  angle(dh,h) q10/50/90 deg = {np.quantile(ang,[.1,.5,.9]).round(1)}")
            if ri in (2, 6, 11) and j == 0:
                # cross-check against the cos-based split
                li = L[27]
                n = norm_h0[li]
                C = cos[ri, j, li]
                dv = dh_norm[ri, j, li]
                arg = dv ** 2 - n ** 2 * (1 - C ** 2)
                s_norm = C * n + np.sqrt(np.maximum(arg, 0))
                a_cos = s_norm * C - n
                b_cos = s_norm * np.sqrt(np.maximum(1 - C ** 2, 0))
                print(f"      cross-check vs cos-based: median |a_cos|/d = "
                      f"{np.median(np.abs(a_cos)/dv):.3f} vs exact {np.median(np.abs(a)/d_tot):.3f}; "
                      f"median b_cos/d = {np.median(b_cos/dv):.3f} vs exact {np.median(b/d_tot):.3f}")

# ---------------------------------------------------------------- Q4 collinearity of saved dirs
print("\nQ4 pairwise cos between dh_last directions (layer 27):")
for ri in [2, 6, 11]:
    v = dh_last[ri]
    cs = []
    for i in range(3):
        for j2 in range(i + 1, 3):
            c = np.einsum("td,td->t", v[i], v[j2]) / (
                np.linalg.norm(v[i], axis=-1) * np.linalg.norm(v[j2], axis=-1))
            cs.append(c)
    cs = np.asarray(cs)
    print(f"  r={radii[ri]:g}: cos q10/50/90 = {np.quantile(cs,[.1,.5,.9]).round(4)}"
          f"  frac |cos|>0.9 = {(np.abs(cs)>0.9).mean():.3f}")

# ---------------------------------------------------------------- Q5 fp32 floor at r=1e-3
print("\nQ5 per-entry linearity at layer 27 (fp32 floor check):")
from scipy import stats
for ri_small, ri_ref in [(0, 2)]:
    x = dh_norm[ri_ref, :, L[27], :] * 0.1        # linear prediction for r=1e-3
    y = dh_norm[ri_small, :, L[27], :]
    rho = stats.spearmanr(x.ravel(), y.ravel()).statistic
    dev = y / np.maximum(x, 1e-30)
    print(f"  r=1e-3 vs 0.1*r=1e-2 at L27: spearman {rho:.3f}; "
          f"y/pred q10/50/90 = {np.quantile(dev,[.1,.5,.9]).round(2)}; "
          f"frac deviating >2x: {(dev>2)|(dev<0.5):.3f}")
    # per-layer fraction of entries at/below a ~1e-3 absolute floor
    for li, l in enumerate(layers):
        yl = dh_norm[0, :, li, :]
        print(f"    L{l}: frac dh(r=1e-3) < 1e-3: {(yl<1e-3).mean():.3f}"
              f"  median dh = {np.median(yl):.2e}")
