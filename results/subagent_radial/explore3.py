"""Third exploration: per-band relative gains, dm correlations, fp32 floor stats."""
import numpy as np
from scipy import stats

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
d = np.load(BASE + r"\w100_layers\a1.npz", allow_pickle=True)
radii = np.asarray(d["radii"]); layers = np.asarray(d["layers"])
norm_h0 = d["norm_h0"].astype(np.float64)
dh_norm = d["dh_norm"].astype(np.float64)
cos = d["cos"].astype(np.float64)
dm = d["dm"].astype(np.float64)
dh_last = d["dh_last"].astype(np.float64)
Lidx = {int(l): i for i, l in enumerate(layers)}
m = np.load(BASE + r"\margins\w100.npz", allow_pickle=True)
h27 = m["h27"][:300].astype(np.float64)
e = h27 / np.linalg.norm(h27, axis=-1, keepdims=True)

# ---------------------------------------------------------------- per-band relative gains
print("per-band per-token RELATIVE gain (dh ratio / scale ratio), r=0.01:")
bands = [(3, 7), (7, 14), (14, 21), (21, 27)]
dh = np.median(dh_norm[2], axis=0)   # (8,300)
for a, b in bands:
    ia, ib = Lidx[a], Lidx[b]
    ok = dh[ia] > 0
    relg = (dh[ib] / dh[ia]) / (norm_h0[ib] / norm_h0[ia])
    print(f"  L{a}->L{b}: q10/50/90 = {np.quantile(relg[ok],[.1,.5,.9]).round(3)}"
          f"  frac<1 = {(relg[ok]<1).mean():.3f}")

# ---------------------------------------------------------------- dm vs radial/tangential
print("\n|dm| vs tangential b and radial |a| at deepest saved layer (dirs 0-2):")
for ri in [2, 4, 6, 8, 11]:
    b_all, a_all, dm_all, d_all = [], [], [], []
    for j in range(3):
        v = dh_last[ri, j]
        d_tot = np.linalg.norm(v, axis=-1)
        a = np.abs(np.einsum("td,td->t", v, e))
        b = np.sqrt(np.maximum(d_tot ** 2 - a ** 2, 0))
        b_all.append(b); a_all.append(a); d_all.append(d_tot)
        dm_all.append(np.abs(dm[ri, j]))
    b_all = np.concatenate(b_all); a_all = np.concatenate(a_all)
    dm_all = np.concatenate(dm_all); d_all = np.concatenate(d_all)
    rb = stats.spearmanr(b_all, dm_all).statistic
    ra = stats.spearmanr(a_all, dm_all).statistic
    rd = stats.spearmanr(d_all, dm_all).statistic
    nrep = np.tile(np.linalg.norm(h27, axis=-1), 3)
    rbh = stats.spearmanr(b_all / nrep, dm_all).statistic
    print(f"  r={radii[ri]:<5g}: spearman |dm|~b {rb:.3f} | ~|a| {ra:.3f} | ~d {rd:.3f} | ~b/||h|| {rbh:.3f}")

# dm scaling with r: is |dm| linear in b?
print("\n|dm| vs b scaling across radii (pooled dirs 0-2):")
for ri in range(0, 12):
    if dh_last[ri].max() == 0:
        continue
    b_med, dm_med = [], []
    for j in range(3):
        v = dh_last[ri, j]
        d_tot = np.linalg.norm(v, axis=-1)
        a = np.abs(np.einsum("td,td->t", v, e))
        b = np.sqrt(np.maximum(d_tot ** 2 - a ** 2, 0))
        b_med.append(np.median(b)); dm_med.append(np.median(np.abs(dm[ri, j])))
    print(f"  r={radii[ri]:<6g}: median b {np.mean(b_med):.3e}  median |dm| {np.mean(dm_med):.3e}")

# ---------------------------------------------------------------- fp32 floor, per-entry
print("\nfp32 floor, per-entry linearity (r=1e-3 vs 0.1*r=1e-2):")
for li, l in enumerate(layers):
    x = dh_norm[2, :, li, :] * 0.1
    y = dh_norm[0, :, li, :]
    ok = x > 0
    if ok.sum() < 30:
        print(f"  L{l}: (no nonzero reference entries at r=1e-2; skipped)")
        continue
    rho = stats.spearmanr(x[ok], y[ok]).statistic
    dev = y[ok] / x[ok]
    print(f"  L{l}: spearman {rho:.3f}  y/pred q10/50/90 {np.quantile(dev,[.1,.5,.9]).round(2)}"
          f"  frac>2x off {float(((dev>2)|(dev<0.5)).mean()):.3f}"
          f"  frac dh<1e-3 {float((y<1e-3).mean()):.3f}")

# exact angle at r=1e-3 (vector floor check)
print("\nexact angle(dh,h) at r=1e-3 (is the vector itself above noise?):")
v = dh_last[0, 0]
d_tot = np.linalg.norm(v, axis=-1)
a = np.abs(np.einsum("td,td->t", v, e))
b = np.sqrt(np.maximum(d_tot ** 2 - a ** 2, 0))
ang = np.degrees(np.arctan2(b, a))
print(f"  q10/50/90 angle {np.quantile(ang,[.1,.5,.9]).round(1)}  median d {np.median(d_tot):.2e}")

# validation: stored cos vs direct recomputation from dh_last & h27
print("\nvalidation: stored cos(s,h) vs direct cos(h+dh, h) at r=2, dirs 0-2:")
n = np.linalg.norm(h27, axis=-1)
for j in range(3):
    v = dh_last[11, j]
    s = h27 + v
    c_direct = np.einsum("td,td->t", s, h27) / (np.linalg.norm(s, axis=-1) * n)
    c_stored = cos[11, j, Lidx[27]]
    print(f"  dir{j}: median |direct-stored| = {np.median(np.abs(c_direct-c_stored)):.2e}"
          f"  max {np.abs(c_direct-c_stored).max():.2e}")

# ---------------------------------------------------------------- fsle relative curve
import json
fsle = json.load(open(BASE + r"\depth\fsle_depth.json"))
log_s = np.log(np.median(norm_h0, axis=1))
ls_full = np.interp(np.arange(28), layers, log_s)
print("\nFSLE depth curves: absolute & relative (divided by state-norm growth):")
for rk, c in fsle["curves"].items():
    c = np.asarray(c)
    abs_prof = c / c[0]
    rel_prof = abs_prof / np.exp(ls_full - ls_full[0])
    print(f"  r={float(rk):g}: abs final {abs_prof[-1]:.1f} (peak {abs_prof.max():.1f} at L{int(np.argmax(abs_prof))})"
          f" | rel: L2 {rel_prof[2]:.2f}  L14 {rel_prof[14]:.3f}  final {rel_prof[-1]:.3f}")
