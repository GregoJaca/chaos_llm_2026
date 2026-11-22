"""Numeric summary for the write-up: everything the Part 2/3 figures show."""
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = os.path.join(ROOT, "results", "ctrl_local", "mamba130")
DS = os.path.join(ROOT, "results", "local_data_ds", "w100_layers", "a1.npz")

mam = {t: np.load(os.path.join(CTRL, f"a1i_{t}.npz")) for t in ["txt", "childhood", "cyber"]}
ds = np.load(DS)

def signed_rc(z, r):
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    m = z["margins"][None, :]
    neg = np.maximum(-z["dm"][ri], 0.0)
    rc = np.where(neg > 0, m * float(z["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    return rc, float(z["radii"][ri])

print("== margins ==")
for t, z in mam.items():
    print(f"mamba {t}: min={z['margins'].min():.4g} q10={np.quantile(z['margins'],.1):.3g} med={np.median(z['margins']):.3g}")
print(f"ds txt: min={ds['margins'].min():.4g} q10={np.quantile(ds['margins'],.1):.3g} med={np.median(ds['margins']):.3g}")

print("\n== signed r* (measured at sub-flip radius) ==")
for t in ["txt", "childhood", "cyber"]:
    rc, rmeas = signed_rc(mam[t], 1.0)
    print(f"mamba {t}: r* = {rc.min():.3g} (measured at r={rmeas}), argmin step t={int(np.unravel_index(rc.argmin(), rc.shape)[1])}")
rc, rmeas = signed_rc(ds, 0.1)
print(f"ds txt: r* = {rc.min():.3g} (measured at r={rmeas})")

print("\n== response level a(t)/r at last layer, r=0.1 ==")
for t, z in mam.items():
    ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    print(f"mamba {t}: t=1 {a[1]:.4g}, t=50 {a[50]:.4g}, t=100 {a[100]:.4g}, t=299 {a[299]:.4g}")
ri = int(np.argmin(np.abs(ds["radii"] - 0.1)))
a = np.median(ds["dh_norm"][ri, :, -1] / ds["radii"][ri], axis=0)
print(f"ds txt: t=1 {a[1]:.4g}, t=50 {a[50]:.4g}, t=100 {a[100]:.4g}, t=299 {a[299]:.4g}")

print("\n== lambda_10(t) ranges (last layer, r=0.1) ==")
k = 10
for t, z in mam.items():
    ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    tt = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[tt + k], 1e-12) / np.maximum(a[tt - k], 1e-12)) / (2 * k)
    print(f"mamba {t}: lam in [{lam.min():.3f}, {lam.max():.3f}], mean {lam.mean():.3f}, frac>0 {np.mean(lam>0):.2f}")
lam_ds = None
ri = int(np.argmin(np.abs(ds["radii"] - 0.1)))
a_ds = np.median(ds["dh_norm"][ri, :, -1] / ds["radii"][ri], axis=0)
tt = np.arange(k, len(a_ds) - k)
lam_ds = np.log(np.maximum(a_ds[tt + k], 1e-12) / np.maximum(a_ds[tt - k], 1e-12)) / (2 * k)
print(f"ds txt: lam in [{lam_ds.min():.3f}, {lam_ds.max():.3f}], mean {lam_ds.mean():.3f}, frac>0 {np.mean(lam_ds>0):.2f}")
print(f"ds txt: lam on t<100: mean {lam_ds[tt<100].mean():.4f}; mamba txt same: ", end="")
ri = int(np.argmin(np.abs(mam['txt']['radii'] - 0.1)))
a_m = np.median(mam['txt']['dh_norm'][ri, :, -1] / mam['txt']['radii'][ri], axis=0)
lam_m = np.log(np.maximum(a_m[tt + k], 1e-12) / np.maximum(a_m[tt - k], 1e-12)) / (2 * k)
print(f"{lam_m[tt<100].mean():.4f}")

print("\n== T1000 plateau ==")
z = np.load(os.path.join(CTRL, "a1i_txt_T1000.npz"))
for r in z["radii"]:
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    print(f"r={r:g}: a/r at t=100 {a[100]:.4g}, t=500 {a[500]:.4g}, t=999 {a[999]:.4g}")

print("\n== flips: obs steps ==")
rows = []
for f in sorted(glob.glob(os.path.join(CTRL, "fr_*.npz"))):
    name = os.path.basename(f)[:-4]
    tag = name.split("_")[1]
    r = float(name.split("_r")[1])
    z = np.load(f, allow_pickle=True)
    obs = z["obs"][0]
    flips = obs[obs >= 0]
    if len(flips):
        print(f"{name}: flip steps {sorted(int(x) for x in flips)}")
    rows.append((name, r, obs))

# delayed flips: response level at flip
print("\n== response level at flip (delayed vs immediate) ==")
for name, r, obs in rows:
    tag = name.split("_")[1]
    if tag not in mam:
        continue
    a1 = mam[tag]
    ri = int(np.argmin(np.abs(a1["radii"] - r)))
    a = np.median(a1["dh_norm"][ri, :, -1] / a1["radii"][ri], axis=0)
    for di in range(len(obs)):
        if obs[di] >= 0:
            t = int(obs[di])
            lev = a[t] / max(a[:t + 1].max(), 1e-30)
            m_t = a1["margins"][t]
            print(f"{name} dir{di}: flip t={t}, margin={m_t:.3g}, response level={lev:.3f}")

# re-sync check
resync = nflip = 0
for f in sorted(glob.glob(os.path.join(CTRL, "fr_*.npz"))):
    z = np.load(f, allow_pickle=True)
    obs = z["obs"][0]
    gen_rows = list(z["gen_ids"])
    nb = len(gen_rows)
    for b in range(nb):
        for j in range(6):
            di = b * 6 + j
            if di >= len(obs) or obs[di] < 0:
                continue
            nflip += 1
            ref = gen_rows[b][0]; row = gen_rows[b][j + 1]
            T0 = int(obs[di])
            if np.array_equal(row[T0 + 1:], ref[T0 + 1:]):
                resync += 1
print(f"\nre-synchronised after flip: {resync}/{nflip}")

# no-flip bit identity
ok = bad = 0
for f in sorted(glob.glob(os.path.join(CTRL, "fr_*.npz"))):
    z = np.load(f, allow_pickle=True)
    obs = z["obs"][0]
    gen_rows = list(z["gen_ids"])
    nb = len(gen_rows)
    for b in range(nb):
        ref = gen_rows[b][0]
        for j in range(6):
            di = b * 6 + j
            if di >= len(obs):
                continue
            row = gen_rows[b][j + 1]
            if obs[di] < 0:
                if np.array_equal(row[1:], ref[1:]) if len(row) == len(ref) else False:
                    ok += 1
                else:
                    bad += 1
print(f"no-flip runs bit-identical to reference: {ok} ok, {bad} BAD")
