"""Audit: per-direction local thresholds r_*, controlling steps, and how they
line up with observed flips. Writes audit.txt."""
import numpy as np

from common import Case, load_margin_traces, obs_pred_at, OUT

lines = []


def P(*a):
    print(*a)
    lines.append(" ".join(str(x) for x in a))


# ---------- window margin traces ----------
traces, summary = load_margin_traces()
P("== baseline margin traces (margins/w*.npz, T=600) ==")
P(f"{'win':>6} {'min m':>10} {'@step':>6} {'p01':>10} {'p05':>10} {'median':>8} {'n<0.01':>7} {'n<0.05':>7}")
for w in sorted(traces):
    s = summary[str(w)]
    P(f"{w:>6} {s['min_margin']:>10.4f} {s['min_step']:>6} {s['p01']:>10.4f} {s['p05']:>10.4f} "
      f"{s['median']:>8.3f} {s['n_below_0_01']:>7} {s['n_below_0_05']:>7}")

# ---------- per-window cases ----------
for name in ["w100", "w200", "w500"]:
    c = Case(name)
    rc = c.rc_ref()
    rstar, tstar = c.r_star()
    P("")
    P(f"== {name}: T={c.T}, r_ref={c.r_ref}, margin min {c.m.min():.4f} @ t={c.m.argmin()} "
      f"(top1={c.top1[c.m.argmin()]}, top2={c.top2[c.m.argmin()]}) ==")
    order = np.argsort(rstar)
    P("   r_* (sorted): " + ", ".join(f"d{d}:{rstar[d]:.3f}@t{tstar[d]}" for d in order))
    cnt = np.bincount(tstar[tstar >= 0], minlength=c.T)
    nz = np.where(cnt > 0)[0]
    P("   controlling-step histogram (t*: n_dirs): " + str(list(zip(nz.tolist(), cnt[nz].tolist()))))
    # observed flips from free runs
    for r in c.free()["radii"]:
        obs, pred = obs_pred_at(name, r)
        fl = np.where(obs >= 0)[0]
        P(f"   free r={r}: obs flips {len(fl)}/{c.n_dirs} -> "
          + ", ".join(f"d{d}@{obs[d]}" for d in fl)
          + f" | pred {int((pred>=0).sum())} flips, exact match {int((obs==pred).sum())}/{c.n_dirs}")
        # cross-check: r_* should rank the flipping directions first
        rank = np.argsort(np.argsort(rstar))  # 0 = smallest r_*
        P(f"      r_* rank of flipping dirs: {sorted(rank[fl].tolist())}  (low rank = should flip first)")

# ---------- linearity check (w100 has 12 radii) ----------
c = Case("w100")
P("")
P("== linearity: g(r)/g(0.01) and r_c(r)/r_c(0.01), w100, 20 dirs x 300 steps ==")
base_g = c.gain[2]
base_rc = c.rc[2]
for ir in range(len(c.radii)):
    r = c.radii[ir]
    ok = np.isfinite(c.rc[ir]) & np.isfinite(base_rc)
    gr = np.zeros_like(c.gain[ir])
    fin = base_g > 0
    gr[fin] = c.gain[ir][fin] / base_g[fin]
    rr = c.rc[ir][ok] / base_rc[ok]
    P(f"  r={r:6.3g}: gain ratio q05/q50/q95 = "
      f"{np.quantile(gr[fin],0.05):6.3f}/{np.quantile(gr[fin],0.5):6.3f}/{np.quantile(gr[fin],0.95):6.3f} "
      f"| r_c ratio q05/q50/q95 = {np.quantile(rr,0.05):6.3f}/{np.quantile(rr,0.5):6.3f}/{np.quantile(rr,0.95):6.3f} "
      f"| frac r_c ratio >2: {(rr>2).mean():.3f}  <0.5: {(rr<0.5).mean():.3f}")

with open(OUT + r"\audit.txt", "w") as f:
    f.write("\n".join(lines))
print("\nwrote audit.txt")
