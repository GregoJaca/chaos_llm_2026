"""Identify the functional form of every curve we care about, and plot the fit."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from chaos_llm.curvefit import fit_curve

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
TAB = os.path.join(ROOT, "results", "fits")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 10,
                     "legend.fontsize": 8})

RECORDS = []


def add(key, model, x, y, xlabel, ylabel, logx=False, logy=False, note=""):
    RECORDS.append(dict(key=key, model=model, x=np.asarray(x, float), y=np.asarray(y, float),
                        xlabel=xlabel, ylabel=ylabel, logx=logx, logy=logy, note=note))


# ---------------------------------------------------------------- depth map (the requested one)
def build_depth():
    f = os.path.join(ROOT, "results", "depth_w100", "depth.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    lam, tokens, layers = d["lam"], d["tokens"], d["layers"]
    med = np.median(lam[:, :, 0], axis=0)
    add("depth_lam1_median", "DeepSeek-1.5B", layers, med, r"layer $\ell$",
        r"top depth exponent $\lambda_1(\ell)$", logx=False, logy=True,
        note="median over 8 tokens of the finite-time depth exponent")
    for ti, t in enumerate(tokens):
        add(f"depth_lam1_t{int(t)}", "DeepSeek-1.5B", layers, lam[ti, :, 0], r"layer $\ell$",
            r"$\lambda_1(\ell)$", logy=True, note=f"token t={int(t)}")
    for i in (1, 2):
        add(f"depth_lam{i+1}_median", "DeepSeek-1.5B", layers, np.median(lam[:, :, i], axis=0),
            r"layer $\ell$", rf"$\lambda_{i+1}(\ell)$", logy=True,
            note=f"exponent {i+1}, median over tokens")


# ---------------------------------------------------------------- depth profiles per model
MODELS = [("DeepSeek-1.5B", os.path.join(LD, "big", "all_layers", "a1.npz"), 0.1),
          ("Qwen3-4B", os.path.join(LD, "models", "qwen3_4b", "a1.npz"), 0.1),
          ("Phi-4-mini", os.path.join(LD, "models", "phi", "a1.npz"), 0.3),
          ("Qwen3.5-2B", os.path.join(LD, "models", "qwen35", "a1.npz"), 0.1)]


def build_profiles():
    for name, path, r in MODELS:
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        A = d["dh_norm"].astype(float); L = np.asarray(d["layers"])
        h = d["norm_h0"].astype(float)
        ri = int(np.argmin(np.abs(d["radii"] - r)))
        for t in (5, 50):
            a = np.median(A[ri, :, :, t], axis=0) / r
            hL = h[np.arange(len(L)), t]
            ok = a > 0
            add(f"depthgain_{name}_t{t}", name, L[ok], a[ok], r"layer $\ell$",
                r"$\|\delta h_{t,\ell}\|/r$", logy=True, note=f"depth gain, t={t}")
            q = a[ok] / np.maximum(hL[ok], 1e-9)
            add(f"relresp_{name}_t{t}", name, L[ok], q, r"layer $\ell$",
                r"$\|\delta h_{t,\ell}\|/\|h_{t,\ell}\|$", logy=True,
                note=f"relative response, t={t}")
            add(f"hprofile_{name}_t{t}", name, L, hL, r"layer $\ell$", r"$\|h_{t,\ell}\|$",
                logy=True, note=f"state norm, t={t}")


# ---------------------------------------------------------------- time response and rate
def build_time():
    f = os.path.join(LD, "w100_layers", "a1.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    A = d["dh_norm"].astype(float); layers = list(d["layers"]); radii = d["radii"]
    ri = int(np.argmin(np.abs(radii - 0.1)))
    li = len(layers) - 1
    a = np.maximum(np.median(A[ri, :, li, :], axis=0), 1e-12)
    t = np.arange(len(a))
    add("a_t_L27_r0.1", "DeepSeek-1.5B", t, a, "generated token $t$",
        r"$\|\delta h_{t,27}\|$", logy=True, note="response, r=0.1")
    k = 10
    tt = t[k:-k]
    lam = np.log(a[tt + k] / a[tt - k]) / (2 * k)
    add("lam10_t_L27", "DeepSeek-1.5B", tt, lam, "generated token $t$",
        r"$\lambda_{10}(t)$", note="finite-time rate")


# ---------------------------------------------------------------- FSLE
def build_fsle():
    f1 = os.path.join(ROOT, "results", "a1_ds_w100", "a1.npz")
    f2 = os.path.join(ROOT, "results", "a1_ds_w100_large", "a1.npz")
    if not (os.path.exists(f1) and os.path.exists(f2)):
        return
    s = np.load(f1); l = np.load(f2)
    radii = np.concatenate([s["radii"], l["radii"]])
    d = np.concatenate([s["dh_norm"], l["dh_norm"]], axis=0)[:, :, -1, :]
    _, idx = np.unique(np.round(radii, 8), return_index=True)
    radii, d = radii[idx], d[idx]
    T = d.shape[2]
    for rho in (1.5, 2.0, 4.0):
        lam, frac = [], []
        for ri in range(len(radii)):
            vals = []
            for di in range(d.shape[1]):
                x = d[ri, di]
                if x[0] <= 0:
                    continue
                hit = np.where(x >= rho * x[0])[0]
                if len(hit):
                    vals.append(np.log(rho) / max(int(hit[0]), 1))
            lam.append(np.median(vals) if vals else np.nan)
            frac.append(1 - len(vals) / d.shape[1])
        lam = np.asarray(lam)
        ok = np.isfinite(lam)
        add(f"fsle_rho{rho}", "DeepSeek-1.5B", radii[ok], lam[ok], r"radius $r$",
            r"FSLE $\lambda(\varrho)$", logx=True, logy=False,
            note=f"growth threshold rho={rho}")
        add(f"fsle_cens_rho{rho}", "DeepSeek-1.5B", radii, frac, r"radius $r$",
            "fraction censored", logx=True, logy=False, note=f"rho={rho}")


# ---------------------------------------------------------------- flux-tube radius
def build_rstar():
    f = os.path.join(LD, "w100_layers", "a1.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    rc = d["r_c"].astype(float); radii = d["radii"]
    ri = int(np.argmin(np.abs(radii - 0.1)))
    rstar = rc[ri].min(axis=1)
    rstar = rstar[np.isfinite(rstar) & (rstar > 0)]
    xs = np.sort(rstar)
    ccdf = 1 - np.arange(len(xs)) / len(xs)
    add("rstar_ccdf", "DeepSeek-1.5B", xs, ccdf, r"$r_*$ (min over $t$)", "CCDF",
        logx=True, logy=True, note="flux-tube radius, r=0.1, 20 directions")
    add("rstar_hist_x", "DeepSeek-1.5B", xs, np.arange(1, len(xs) + 1) / len(xs),
        r"$r_*$", "ECDF", logx=True, note="ECDF")


def main():
    build_depth()
    build_profiles()
    build_time()
    build_fsle()
    build_rstar()
    print("curves:", len(RECORDS))

    summary = ["# Functional-form fits", "",
               "Fitted with `chaos_llm.curvefit`. BIC comparison; a model within dBIC<2 of the",
               "best is listed as a tie. `domain` says where the fit was made (log domain for",
               "pure exponential/power when the curve spans decades).", ""]
    rows = ["| curve | model | best form | parameters | R2 | dBIC(2nd) | note |",
            "|---|---|---|---|---|---|---|"]
    for rec in RECORDS:
        try:
            res = fit_curve(rec["x"], rec["y"])
        except Exception as e:
            print("FIT FAILED", rec["key"], e)
            continue
        b = res.best
        second = res.fits[1] if len(res.fits) > 1 else None
        dbic = (second.bic - b.bic) if second else np.nan
        name = f"{rec['key']}__{rec['model']}"
        rows.append(f"| {rec['key']} | {rec['model']} | `{b.formula}` | {b.line(3)} | "
                    f"{b.r2:.4f} | {dbic:.1f} | {rec['note']} |")
        json.dump(dict(key=rec["key"], model=rec["model"], best=b.name, formula=b.formula,
                       params=dict(zip(b.pnames, b.params.tolist())),
                       se=dict(zip(b.pnames, b.se.tolist())), r2=b.r2, r2_fit=b.r2_fit,
                       bic=b.bic, aic=b.aic, domain=res.log_mode,
                       table=[dict(name=f.name, formula=f.formula, r2=f.r2, bic=f.bic,
                                   params=dict(zip(f.pnames, f.params.tolist())))
                              for f in res.fits[:8]]),
                  open(os.path.join(TAB, name + ".json"), "w"), indent=1)

        # ---- figure
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(rec["x"], rec["y"], "o", ms=4, color="0.25", label="data", zorder=3)
        xx = np.linspace(rec["x"].min(), rec["x"].max(), 400)
        ax.plot(xx, b.predict(xx), "-", lw=1.8, color="tab:red",
                label=f"best: {b.name}  $R^2$={b.r2:.4f}")
        if second is not None:
            ax.plot(xx, second.predict(xx), "--", lw=1.2, color="tab:blue",
                    label=f"2nd: {second.name}  $\\Delta$BIC={dbic:.0f}")
        if rec["logx"]:
            ax.set_xscale("log")
        if rec["logy"]:
            ax.set_yscale("log")
        ax.set_xlabel(rec["xlabel"]); ax.set_ylabel(rec["ylabel"])
        ax.set_title(f"{rec['model']}: {rec['note']}")
        ax.grid(alpha=0.25, which="both"); ax.legend(loc="best")
        txt = b.formula + "\n" + b.line(3)
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6.5, va="bottom",
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fit_{name}.png"), dpi=180)
        plt.close(fig)
        print("ok", rec["key"], "->", b.name, round(b.r2, 4))

    summary += rows
    open(os.path.join(TAB, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(summary))
    print("wrote", os.path.join(TAB, "SUMMARY.md"))


if __name__ == "__main__":
    main()
