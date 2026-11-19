"""s4b_radial.py -- corrected radial/tangential analysis (D3).

The a1 'cos' key stores cos(h + dh, h) (verified: values ~1), NOT
cos(dh, h).  For small rho = ||dh||/||h||:

    1 - cos(h+dh, h)  ~=  (1 - c^2) rho^2 / 2,   c = cos(dh, h)

so the squared RADIAL fraction of the response is
    c^2 = 1 - (1-cos)*2/rho^2,
recoverable where float precision resolves 1-cos (rho^2/2 >> eps).
DeepSeek's a1 files store cos in float32 with enough precision (and a
12-radius sweep in w100); the other three models' cos arrays are on a
float16 grid (granularity ~1e-3) and saturate at 1.0, so only a vacuous
upper bound is available there.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from dimlib import (A1_BY_TAG, COLOR, DISPLAY, section_numbers, savefig)

RHO_MIN = 3e-4          # need rho^2/2 >> float32 eps (~1.2e-7)
NQ = (10, 90)


def radial_c2(cos, dh, h0):
    """c^2 = 1 - 2(1-cos)/rho^2 with precision mask (per r, dir, l, t)."""
    rho = dh / np.maximum(h0[None, None, :, :], 1e-30)
    om = 1.0 - cos
    c2 = 1.0 - np.where(rho > RHO_MIN, 2 * om / np.maximum(rho**2, 1e-30),
                        np.nan)
    return np.clip(c2, 0, 1), rho


def main():
    out = {"interpretation": (
        "a1 'cos' = cos(h+dh, h) (values ~1, verified); radial energy "
        "fraction c^2 = cos^2(dh, h) recovered via 1-cos ~ (1-c^2) rho^2/2. "
        "Precision: deepseek a1 cos is float32-fine; qwen3_4b/phi4mini/"
        "qwen3_5 cos arrays are on a float16 grid (step ~1e-3) and saturate "
        "-- no radial estimate possible for them from this data."),
        "rho_min": RHO_MIN}

    # ---- deepseek main sweep (3 radii, 28 layers)
    d = np.load(A1_BY_TAG["deepseek"])
    cos = d["cos"].astype(np.float64)
    dh = d["dh_norm"].astype(np.float64)
    h0 = d["norm_h0"].astype(np.float64)
    layers = d["layers"]
    L = 28
    c2, rho = radial_c2(cos, dh, h0)
    ok = np.isfinite(c2)
    per_layer = np.nanmedian(np.where(ok, c2, np.nan), axis=(0, 1, 3))
    out["deepseek_c2_per_layer_median"] = [float(v) for v in per_layer]
    out["deepseek_c2_all_median"] = float(np.nanmedian(c2[ok]))
    out["deepseek_rho_median_lastlayer"] = float(np.nanmedian(
        rho[:, :, -1, :]))
    print("deepseek c^2 all-layers median:", out["deepseek_c2_all_median"])
    print("  per-layer c^2 (first 4, last 4):",
          np.round(per_layer[1:5], 4), np.round(per_layer[-4:], 4))

    f, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(layers[1:] / L, per_layer[1:], "o-", color=COLOR["deepseek"],
            ms=4, lw=1.4, label="median over 3 radii x 20 dirs x 300 tok")
    band_lo = np.nanpercentile(np.where(ok, c2, np.nan), 10, axis=(0, 1, 3))
    band_hi = np.nanpercentile(np.where(ok, c2, np.nan), 90, axis=(0, 1, 3))
    ax.fill_between(layers[1:] / L, band_lo[1:], band_hi[1:],
                    color=COLOR["deepseek"], alpha=0.2, lw=0,
                    label="10-90 pct")
    ax.axhline(0.01, ls=":", lw=0.8, color="k")
    ax.annotate("1% radial energy", (0.02, 0.012), fontsize=7)
    ax.set_xlabel("relative depth $\\ell/L$")
    ax.set_ylabel("radial energy fraction $c^2=\\cos^2(\\delta h, h)$")
    ax.set_title(f"DeepSeek: the response is $\\geq$99% TANGENTIAL at every "
                 f"depth (median $c^2$="
                 f"{out['deepseek_c2_all_median']:.3f})", fontsize=8.8)
    ax.legend(fontsize=7)
    savefig(f, "fig_D3_radial_c2_deepseek.png")
    del d, cos, dh

    # ---- w100 radius sweep (12 radii, 8 layers)
    d = np.load(A1_BY_TAG["deepseek_w100"])
    cos = d["cos"].astype(np.float64)
    dh = d["dh_norm"].astype(np.float64)
    h0 = d["norm_h0"].astype(np.float64)
    radii = d["radii"]
    layers_w = d["layers"]
    c2, rho = radial_c2(cos, dh, h0)
    f, ax = plt.subplots(figsize=(5.8, 3.5))
    sel = [1, 2, 3, 4, 6, 7]          # blocks incl. last
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sel)))
    prof = {}
    for c, li in zip(cmap, sel):
        y = np.nanmedian(c2[:, :, li, :], axis=(1, 2))   # -> per radius
        ax.plot(radii, y, "o-", color=c, ms=4,
                label=f"a1 layer {layers_w[li]}")
        prof[int(layers_w[li])] = [float(v) for v in y]
    ax.set_xscale("log")
    ax.set_xlabel("perturbation radius $r$")
    ax.set_ylabel("radial energy fraction $c^2$")
    ax.set_title("DeepSeek (12 radii, 0.001-2.0): the response stays "
                 "~99% tangential even far outside the linear regime",
                 fontsize=8.6)
    ax.legend(fontsize=6.6)
    savefig(f, "fig_D3c_radial_vs_radius_deepseek.png")
    out["deepseek_w100_c2_vs_radius_per_layer"] = prof
    out["deepseek_w100_radii"] = [float(v) for v in radii]
    med_all = float(np.nanmedian(c2))
    out["deepseek_w100_c2_all_median"] = med_all
    print("w100 c^2 all median:", med_all)

    # ---- saturation check for the other models (granularity)
    gran = {}
    for tag in ["qwen3_4b", "phi4mini", "qwen3_5"]:
        dd = np.load(A1_BY_TAG[tag])
        c = dd["cos"]
        u = np.unique(c)
        step = float(np.min(np.diff(u)[:10])) if len(u) > 1 else 0.0
        frac1 = float((c == 1.0).mean())
        gran[tag] = {"unique_values": int(len(u)),
                     "min_grid_step": step,
                     "frac_exactly_one": frac1,
                     "rho_median_lastlayer": float(np.median(
                         dd["dh_norm"].astype(np.float64)[:, :, -1, :]
                         / dd["norm_h0"].astype(np.float64)[-1, :]))}
        print(f"{tag}: unique cos values={len(u)}, min step={step:.2e}, "
              f"frac==1.0: {frac1:.2f}")
        del dd
    out["saturation_check_other_models"] = gran

    section_numbers("D3_radial", out)


if __name__ == "__main__":
    main()
