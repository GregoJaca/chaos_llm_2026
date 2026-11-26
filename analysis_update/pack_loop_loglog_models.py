"""Log-log closed-loop vs network-only-replay, per model (the F0b family).

For each model we plot
  (a) the response ||dh_t||/||dh_1|| on log-log axes, closed loop vs replay,
      raw (thin) and a 5-step rolling median (thick), plus the IQR over directions;
  (b) the log ratio log(a_closed/a_replay) with its fitted feedback exponent.

Reads whichever orbit is longest: results/longloop/<tag>/T*/loop/softloop.npz
or the older 600-step results/softloop/<tag>/T*/softloop.npz.
"""
import os
import sys
import glob
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

TAGS = {
    "DeepSeek-1.5B": "ds15",
    "Qwen2.5-3B": "qwen25_3b",
    "Qwen3-4B": "qwen3_4b",
    "Qwen3.5-2B": "qwen3_5",
    "Qwen3.5-9B": "qwen35_9b",
    "Phi-4-mini": "phi4mini",
    "DeepSeek-14B": "ds14",
    "Qwen3-14B": "qwen314",
    "Phi-4": "phi4",
    "Mamba-130m": "mamba130",
    "Pythia-410m": "pythia410",
    "randQwen": "randqwen",
}
LONGTAG = dict(TAGS)
LONGTAG.update({"DeepSeek-1.5B": "ds15v2", "Qwen3-4B": "qwen34",
                "Qwen2.5-3B": "qwen25", "Phi-4-mini": "phi"})


def candidates(tag, mode, name=None):
    """All npz paths for one model, best first.

    Preference: the long campaign (`results/longloop`, 3000/1600 steps) over the
    older 600-step `results/softloop`, T=1.0 over other temperatures, and the
    longest run over shorter ones.
    """
    tags = {tag}
    if name is not None and name in LONGTAG:
        tags.add(LONGTAG[name])
    found = []
    for root in ("results/longloop", "results/softloop"):
        for tg in tags:
            for sub in glob.glob(os.path.join(HERE, root, tg, "T*")):
                if mode == "loop":
                    p = os.path.join(sub, "loop", "softloop.npz") if root.endswith("longloop") \
                        else os.path.join(sub, "softloop.npz")
                else:
                    p = os.path.join(sub, "replay", "softloop.npz")
                if not os.path.exists(p):
                    continue
                try:
                    n = np.load(p, allow_pickle=True)["dh"].shape[-1]
                except Exception:
                    n = 0
                t1 = "T1.0" in sub.replace("\\", "/")
                found.append(((root.endswith("longloop"), t1, n), p))
    found.sort(key=lambda r: r[0], reverse=True)
    return [p for _, p in found]


def pick_t1(paths):
    return paths[0] if paths else None


def raw(path, layer=-1):
    z = np.load(path, allow_pickle=True)
    a = z["dh"][:, layer, :].astype(float)
    return a / np.maximum(a[:, [1]], 1e-30)


def roll(x, w=5):
    if len(x) <= w:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


rows = {}
for name, tag in TAGS.items():
    tag_l = LONGTAG.get(name, tag)
    pl = pick_t1(candidates(tag, "loop", name))
    pr = pick_t1(candidates(tag, "replay", name))
    if pl is None and pr is None:
        continue
    rows[name] = dict(loop=pl, replay=pr)

MODELS = list(rows)
if not MODELS:
    raise SystemExit("no loop data found")
print("models:", MODELS)

summary = {}
for name in MODELS:
    r = rows[name]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    ax = axes[0]
    med = {}
    for key, c, ls, lab in [("loop", "#d62728", "-", "closed loop"),
                            ("replay", "#1f77b4", "--", "replay (network only)")]:
        if not r[key]:
            continue
        a = raw(r[key])
        m = np.median(a, axis=0)
        lo, hi = np.percentile(a, [25, 75], axis=0)
        t = np.arange(len(m))
        sel = t >= 1
        ax.plot(t[sel], m[sel], color=c, lw=0.5, alpha=0.35)
        ax.plot(t[sel], roll(m)[sel], color=c, lw=1.5, ls=ls, label=f"{lab} ({len(m)} steps)")
        ax.fill_between(t[sel], np.maximum(lo[sel], 1e-8), hi[sel], color=c, alpha=0.10, lw=0)
        med[key] = m
        summary.setdefault(name, {})[f"n_steps_{key}"] = int(len(m))
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$")
    ax.set_title(f"{name}: closed loop vs replay, $T=1$", loc="left", fontsize=9.4)
    ax.legend(fontsize=7.4, loc="upper left"); ax.grid(True, which="both", alpha=0.28)

    ax = axes[1]
    if "loop" in med and "replay" in med:
        T = min(len(med["loop"]), len(med["replay"]))
        lr = np.log(np.maximum(med["loop"][:T], 1e-30)) - np.log(np.maximum(med["replay"][:T], 1e-30))
        lrs = roll(lr, 9)
        t = np.arange(T)
        ax.plot(t[1:], lr[1:], color="#7f7f7f", lw=0.5, alpha=0.5)
        ax.plot(t[1:], lrs[1:], color="#2ca02c", lw=1.5, label=r"$\log(a_{\rm closed}/a_{\rm replay})$")
        sel = t >= 10
        if sel.sum() > 8:
            b = np.polyfit(t[sel], lr[sel], 1)[0]
            summary[name]["lambda_fb"] = float(b)
            # non-parametric saturation level: median of the last 20% of the run
            k0 = int(0.8 * (T - 1))
            L_end = float(np.median(lrs[max(k0, 1):]))
            summary[name]["L_end"] = L_end
            # time to reach 63% of L_end (rise time)
            t63 = float("nan")
            if L_end > 0.2:
                idx = np.where(lrs >= 0.63 * L_end)[0]
                t63 = float(idx[0]) if len(idx) else float("nan")
            summary[name]["t63"] = t63
            # two-parameter saturating fit with the offset fixed at the early median
            def sat(tt, L, tau):
                return y0 + (L - y0) * (1.0 - np.exp(-tt / tau))
            y0 = float(np.median(lrs[:5]))
            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(sat, t[1:], lrs[1:], p0=[max(L_end, 0.5), max(T / 4.0, 5.0)],
                                    bounds=([0.0, 0.5], [60.0, 5e4]), maxfev=40000)
                resid = lrs[1:] - sat(t[1:], *popt)
                r2 = 1 - np.sum(resid ** 2) / np.sum((lrs[1:] - lrs[1:].mean()) ** 2)
                tt = np.linspace(1, T - 1, 400)
                ax.plot(tt, sat(tt, *popt), "k--", lw=1.1,
                        label=rf"fit $L_\infty$={popt[0]:.1f}, $\tau$={popt[1]:.0f} ($R^2$={r2:.2f})")
                summary[name].update(L_inf=float(popt[0]), tau=float(popt[1]), sat_r2=float(r2))
            except Exception as exc:
                print("sat fit failed", name, exc)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("step $t$")
        ax.set_ylabel("log ratio")
        ax.set_title("feedback gain: saturating, not exponential", loc="left", fontsize=9.4)
        ax.legend(fontsize=6.8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"pack_loop_loglog_{TAGS[name]}.png")

    # a compact number for the text: growth factor at the end, closed vs replay
    if "loop" in med:
        summary[name]["growth_closed"] = float(med["loop"][-1])
    if "replay" in med:
        summary[name]["growth_replay"] = float(med["replay"][-1])
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[name].items()})

# ---------------- the all-models grid (smoothed, same axes style)
fig, axes = plt.subplots(3, 4, figsize=(16.0, 10.8))
for ax, n in zip(axes.flat, MODELS):
    r = rows[n]
    for key, c, ls, lab in [("loop", "#d62728", "-", "closed loop"),
                            ("replay", "#1f77b4", "--", "replay (network only)")]:
        if not r[key]:
            continue
        a = raw(r[key]); m = np.median(a, axis=0); t = np.arange(len(m))
        ax.plot(t[1:], roll(m)[1:], color=c, lw=1.3, ls=ls, label=f"{lab} ({len(m)})")
    ax.axhline(1, color="k", ls=":", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(n, fontsize=8.8)
    ax.set_xlabel("step $t$", fontsize=7.4)
    ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$", fontsize=7.4)
    ax.legend(fontsize=5.8, loc="upper left"); ax.grid(True, which="both", alpha=0.25)
for j in range(len(MODELS), 12):
    axes.flat[j].axis("off")
fig.suptitle("Closed loop vs replay, log--log, every model, $T=1$ (5-step rolling median)", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_loop_loglog_all.png")

json.dump(summary, open(os.path.join(HERE, "analysis_update", "pack_loop_loglog_models.json"), "w"), indent=1)
print("done", len(MODELS), "models")



