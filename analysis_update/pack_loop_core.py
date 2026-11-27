"""Closed loop vs network-only replay, done properly.

Definitions used everywhere in the report
    a_c(d,t) = ||delta h_t|| / ||delta h_1||   for direction d, closed loop
    a_r(d,t) = same for the replay of the SAME recorded orbit
    L(d,t)   = log( a_c(d,t) / a_r(d,t) )      per direction, then median/IQR
The ratio is always formed per direction and only then summarised, so the band is
an honest spread over probe directions.  L(1) = 0 exactly (both curves are
normalised at t=1), which is why the fit is L(t) = Linf (1 - exp(-t/tau)) with
NO free offset.

Data: every (loop, replay) pair in the same directory, at the same T, all chunks
pooled (a chunk = one run with `--dirs K`; the reference orbit is identical across
chunks because row 0 is unperturbed), so 8-direction legacy runs and 256-direction
v3 runs are handled by the same code.
"""
import os
import sys
import glob
import json
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

TAGS = {
    "DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25", "Qwen3-4B": "qwen34",
    "Qwen3.5-2B": "qwen35", "Qwen3.5-9B": "qwen35_9b", "Phi-4-mini": "phi",
    "DeepSeek-14B": "ds14", "Qwen3-14B": "qwen314", "Phi-4": "phi4",
    "Mamba-130m": "mamba", "Pythia-410m": "pythia", "randQwen": "randqwen",
}
LEGACY = {"Qwen2.5-3B": "qwen25_3b", "Qwen3-4B": "qwen3_4b", "Qwen3.5-2B": "qwen3_5",
          "Phi-4-mini": "phi4mini", "DeepSeek-1.5B": "ds15", "Mamba-130m": "mamba130",
          "Pythia-410m": "pythia410", "randQwen": "randqwen", "Qwen3.5-9B": "qwen35_9b"}
ROOTS = ["results/longloop_v3", "results/longloop", "results/softloop"]


def find_dirs(tag_list, T="T1.0"):
    """all directories for this model/T that contain a loop file"""
    out = []
    for root in ROOTS:
        for tag in tag_list:
            for d in glob.glob(os.path.join(HERE, root, tag, T)):
                if os.path.exists(os.path.join(d, "loop", "softloop.npz")):
                    out.append(d)
                elif os.path.exists(os.path.join(d, "softloop.npz")):
                    out.append(d)
    return out


def loop_path(d):
    p = os.path.join(d, "loop", "softloop.npz")
    return p if os.path.exists(p) else os.path.join(d, "softloop.npz")


def replay_path(d):
    p = os.path.join(d, "replay", "softloop.npz")
    return p if os.path.exists(p) else None


def load_curve_pair(d, layer=-1):
    """per-direction a_c and a_r arrays, normalised at t=1, on the common length"""
    zc = np.load(loop_path(d), allow_pickle=True)
    pr = replay_path(d)
    if pr is None:
        return None
    zr = np.load(pr, allow_pickle=True)
    ac = zc["dh"][:, layer, :].astype(float)
    ar = zr["dh"][:, layer, :].astype(float)
    n = min(ac.shape[1], ar.shape[1])
    ac = ac[:, :n]; ar = ar[:, :n]
    ac = ac / np.maximum(ac[:, [1]], 1e-30)
    ar = ar / np.maximum(ar[:, [1]], 1e-30)
    # a direction is only usable if both curves are finite and above the floor
    ok = np.all(np.isfinite(ac), axis=1) & np.all(np.isfinite(ar), axis=1)
    return ac[ok], ar[ok]


def gather(name, T="T1.0"):
    tags = list(dict.fromkeys([TAGS[name]] + ([LEGACY[name]] if name in LEGACY else [])))
    dirs = sorted(set(find_dirs(tags, T)))
    # prefer v3 (256 dirs); fall back to everything available
    by_root = {}
    for d in dirs:
        parts = d.replace("\\", "/").split("/")
        r = parts[parts.index("results") + 1] if "results" in parts else "?"
        by_root.setdefault(r, []).append(d)
    for root in ROOTS:
        key = root.split("/")[1]
        if key in by_root:
            ds = sorted(by_root[key])
            acs, ars = [], []
            for d in ds:
                got = load_curve_pair(d)
                if got is None:
                    continue
                acs.append(got[0]); ars.append(got[1])
            if acs:
                L = min(a.shape[1] for a in acs)
                ac = np.concatenate([a[:, :L] for a in acs], axis=0)
                ar = np.concatenate([a[:, :L] for a in ars], axis=0)
                return ac, ar, ds
    return None, None, []


def band(x, axis=0):
    return np.median(x, axis=axis), np.percentile(x, [25, 75], axis=axis)


def sat(tt, L, tau):
    return L * (1.0 - np.exp(-tt / tau))


summary = {}
MODELS = [m for m in TAGS]
fig, axes = plt.subplots(3, 4, figsize=(16.6, 10.6))
for ax, name in zip(axes.flat, MODELS):
    ac, ar, ds = gather(name)
    if ac is None:
        ax.axis("off"); continue
    mc, (clo, chi) = band(ac, 0)
    mr, (rlo, rhi) = band(ar, 0)
    t = np.arange(len(mc))
    ax.fill_between(t[1:], np.maximum(clo[1:], 1e-8), chi[1:], color="#d62728", alpha=0.12, lw=0)
    ax.fill_between(t[1:], np.maximum(rlo[1:], 1e-8), rhi[1:], color="#1f77b4", alpha=0.12, lw=0)
    ax.plot(t[1:], mc[1:], color="#d62728", lw=1.2, label=f"closed ({ac.shape[0]} dirs)")
    ax.plot(t[1:], mr[1:], color="#1f77b4", lw=1.0, ls="--", label="replay")
    ax.axhline(1, color="k", ls=":", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{name}  ({len(ds)} run{'s' if len(ds) > 1 else ''})", fontsize=8.8)
    ax.set_xlabel("step $t$", fontsize=7.2)
    ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$", fontsize=7.2)
    ax.legend(fontsize=5.6); ax.grid(True, which="both", alpha=0.24)
    summary[name] = dict(n_dirs=int(ac.shape[0]), steps=int(len(mc)),
                         growth_closed=float(mc[-1]), growth_replay=float(mr[-1]))
for j in range(len(MODELS), 12):
    axes.flat[j].axis("off")
fig.suptitle("Closed loop vs replay at $T=1$: median over directions, band = IQR over directions", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_loop_all.png")

# ---------------------------------------------------------------- per model: response + ratio + fit
for name in MODELS:
    ac, ar, ds = gather(name)
    if ac is None:
        continue
    t = np.arange(ac.shape[1])
    Ld = np.log(np.maximum(ac, 1e-30)) - np.log(np.maximum(ar, 1e-30))   # (dir, t)
    Lm, (Llo, Lhi) = band(Ld, 0)
    fig, axes2 = plt.subplots(1, 2, figsize=(13.2, 4.5))
    ax = axes2[0]
    for arr, c, ls, lab in [(ac, "#d62728", "-", "closed loop"),
                            (ar, "#1f77b4", "--", "replay (network only)")]:
        m, (lo, hi) = band(arr, 0)
        ax.fill_between(t[1:], np.maximum(lo[1:], 1e-8), hi[1:], color=c, alpha=0.12, lw=0)
        ax.plot(t[1:], m[1:], color=c, lw=1.4, ls=ls, label=lab)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$")
    ax.set_title(f"{name}: response ($T=1$, {ac.shape[0]} directions)", loc="left", fontsize=9.6)
    ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.28)

    ax = axes2[1]
    ax.fill_between(t[1:], Llo[1:], Lhi[1:], color="#2ca02c", alpha=0.14, lw=0)
    ax.plot(t[1:], Lm[1:], color="#2ca02c", lw=1.6,
            label=r"$\log(\|\delta h_t^{\rm closed}\|/\|\delta h_t^{\rm replay}\|)$")
    tau = float("nan"); Linf = float("nan"); r2 = float("nan")
    sel = t >= 1
    try:
        popt, _ = curve_fit(sat, t[sel], Lm[sel], p0=[max(Lm[sel][-1], 0.5), max(len(t) / 4, 5)],
                            bounds=([0.0, 0.5], [60.0, 5e4]), maxfev=40000)
        Linf, tau = float(popt[0]), float(popt[1])
        pred = sat(t[sel], *popt)
        r2 = 1 - np.sum((Lm[sel] - pred) ** 2) / np.sum((Lm[sel] - Lm[sel].mean()) ** 2)
        tt = np.linspace(1, t[-1], 300)
        ax.plot(tt, sat(tt, *popt), "k--", lw=1.1,
                label=rf"$L_\infty$={Linf:.1f} nats, $\tau$={tau:.0f} steps ($R^2$={r2:.2f})")
    except Exception as exc:
        print("fit failed", name, exc)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("step $t$"); ax.set_ylabel("log ratio (nats)")
    ax.set_title("feedback gain, saturating fit", loc="left", fontsize=9.6)
    ax.legend(fontsize=6.8); ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"pack_loop_{TAGS[name]}.png")

    summary[name].update(L_inf=Linf, tau=tau, sat_r2=r2,
                         L_end=float(np.median(Ld[:, int(0.8 * len(t)):])),
                         params=[p.replace(HERE, "") for p in ds])
    print("%-14s dirs=%-4d steps=%-5d L_inf=%5.2f tau=%7.1f r2=%5.2f  L_end=%5.2f" % (
        name, ac.shape[0], len(t), Linf, tau, r2, summary[name]["L_end"]))

json.dump(summary, open(os.path.join(HERE, "analysis_update", "pack_loop_core.json"), "w"), indent=1)
print("done")
