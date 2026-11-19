"""
plots.py -- reusable figure factory for transformer perturbation-dynamics data
================================================================================

Self-contained: numpy + matplotlib only (Agg backend, CPU).

Data schemas handled (one directory per run, meta.json optional)
-----------------------------------------------------------------
a1.npz      teacher-forced prompt perturbation:
            dh_norm [n_r, n_dir, n_layer, T]  = ||delta h_{t,l}|| (t index 0 -> t=1)
            dm [n_r,n_dir,T], cos [n_r,n_dir,n_layer,T], margins [T],
            top1/top2 [T], r_c [n_r,n_dir,T] (sentinel ~1e30 = never),
            pred_T [n_r,n_dir] (-1 = never flips), radii [n_r], layers [n_layer],
            norm_h0 [n_layer,T], optional dh_last [n_r, 3, T, D] (last-layer vectors).
state2.npz  single-position state perturbation:
            dh [n_r, n_s, n_dir, n_layer, T], cos/dm same, steps [n_s],
            radii, layers, inject_layer (0-d), margins.
            EMPIRICAL convention (verified on local data): injection at step s
            first affects time index s (0-based) at every recorded layer
            >= inject_layer; values before injection are an fp32 noise floor
            (~1e-3 relative), not exact zeros.
layer_probe.npz  dh [n_inj, n_dir, 28, T], layers (injection layers), pos, r
            (0-d scalars), dirs [n_dir, D].
            EMPIRICAL convention: response is zero at readout layers <= inj
            layer; it first appears at readout layer inj+1, time index pos,
            with value exactly r. Injection at the final layer (27) gives an
            all-zero slab (it only perturbs logits, no recorded state).
free_run.npz  obs [n_r, n_dir] = first flip step (-1 = never), radii [n_r].
h_last npz   h_last [T, D], gen_ids [T].
tf_pred npz  pred [n_r,n_dir,T], ar [n_r,n_dir,T], ref [T], radii [n_r].
depth npz    local_gain [n_tok, 28], lam [n_tok, 28, 8], h_depth [n_tok],
            tokens [n_tok], layers [28].

Quantities (all logs natural, rates per generated step)
------------------------------------------------------
a(t)      = ||delta h_{t,l}|| (dh_norm)                    "amplitude"
g(t)      = a(t)/a(1)                                      growth, normalised by a(1)
rho(t)    = a(t)/a(t-1),  t >= 2                           consecutive-step ratio
lam_k(t)  = (1/k) ln(a(t)/a(t-k))                          local rate, t > k
Lam(t)    = (1/t) ln(a(t)/a(1))                            cumulative rate
departure = first t with median g(t) >= DEPARTURE_THR
latency  (state) = first lag tau with a(s+tau) >= LATENCY_THR * a(s+tau=0)
margin flip    = first t with dm < -DM_FLIP_TOL (tolerance for the
                 +/-1e-5 pre-injection fp noise observed in state2 files)
token agreement(t) = fraction of perturbed rollouts whose first-flip step
                 obs is > t (or obs < 0 = never flipped)

Aggregation: median line + 10-90 percentile band (nan-aware; exact zeros in
dh_norm are treated as "below fp noise" -> NaN and dropped, the valid sample
count n is annotated on every banded figure).

Figures: <= 4 panels, font sizes >= 8, constrained_layout, bbox_inches='tight',
dpi 190, each panel autoscaled (no shared axes).

Usage
-----
    import plots
    a1 = plots.load_a1("results/local_data/w100_layers")
    plots.flux_tube_family(a1, "out/figs/group", "w100_layers")
    ...
Every family function returns the list of PNG paths it wrote.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap  # noqa: F401  (typing only)

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
DPI = 190
PLO, PHI = 10, 90          # percentile band
N_THIN = 3                 # thin per-direction example curves
MAX_PANELS = 4
DEPARTURE_THR = 2.0        # growth "first departure" threshold
LATENCY_THR = 2.0          # state-perturbation latency threshold
DM_FLIP_TOL = 1e-3         # dm < -tol counts as a margin flip
RC_SENTINEL = 1e20          # r_c values above this = "never crosses" -> NaN
KS = (5, 10, 25)            # local-rate windows

WARNINGS: list[str] = []   # collected non-fatal notes, dumped by the runner


def _warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[plots] WARN: {msg}", flush=True)


def _setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })
    # Windows consoles often use cp1252: never die printing unicode tick labels
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:  # noqa: BLE001
            pass


_setup()


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def t_axis(T: int) -> np.ndarray:
    """1-based generation-step axis for a length-T array."""
    return np.arange(1, T + 1)


def pos(x) -> np.ndarray:
    """Positive-only copy (zeros/negatives -> NaN) for log arithmetic."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, np.nan)


def safe_ratio(a, b) -> np.ndarray:
    """a / b where b <= 0 -> NaN (no divide warnings)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)
    return out


def med_band(stack: np.ndarray, axis: int = 0):
    """nan-aware median + PLO-PHI percentile band along `axis`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(stack, axis=axis)
        lo = np.nanpercentile(stack, PLO, axis=axis)
        hi = np.nanpercentile(stack, PHI, axis=axis)
    return med, lo, hi


def n_valid(stack: np.ndarray) -> int:
    """number of sample rows (axis 0) containing at least one finite value."""
    if stack.ndim == 1:
        return int(np.any(np.isfinite(stack)))
    return int(np.sum(np.any(np.isfinite(stack), axis=tuple(range(1, stack.ndim)))))


def select_indices(n: int, k: int) -> list[int]:
    """up to k nicely spread indices out of n."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    return idx.tolist()


def fmt_r(r: float) -> str:
    return f"{r:.3g}"


def _sanitize(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", tag)[:60]


def seq_colors(n: int, cmap: str = "viridis") -> list:
    cm = plt.get_cmap(cmap)
    return [cm(i / max(n - 1, 1)) for i in range(n)]


def _fig_panels(n: int, ncols: int = 2, w: float = 4.5, h: float = 3.3,
                suptitle: str | None = None):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             squeeze=False, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.07, h_pad=0.09)
    flat = [a for row in axes for a in row]
    for a in flat[n:]:
        a.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    return fig, flat[:n]


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for issue in _qa_fig(fig, path)[:6]:
        _warn(f"QA {os.path.basename(path)}: {issue}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _qa_fig(fig, path: str) -> list[str]:
    """detect text artists outside the canvas or overlapping each other.

    Advisory only (bbox_inches='tight' can still rescue out-of-canvas text),
    but text-on-text overlap indicates a real layout problem.
    Ghost tick labels (matplotlib keeps Text objects for out-of-view ticks and
    for hidden axes, but never draws them) are excluded.
    """
    issues: list[str] = []
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        W, H = fig.canvas.get_width_height()
    except Exception:  # noqa: BLE001
        return issues
    ghosts = _ghost_tick_ids(fig)
    texts = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible() or id(t) in ghosts:
            continue
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:  # noqa: BLE001
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        texts.append((s, bb))
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2:
            issues.append(f"outside canvas: {s[:40]!r}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1], texts[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 3 and oy > 3:
                issues.append(f"text overlap: {texts[i][0][:24]!r} vs "
                              f"{texts[j][0][:24]!r}")
    return issues


def _ghost_tick_ids(fig) -> set[int]:
    """ids of tick-label Text objects that will not be drawn."""
    ghosts: set[int] = set()
    for ax in fig.axes:
        all_ids = set()
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks():
                all_ids.add(id(tick.label1))
                all_ids.add(id(tick.label2))
            for tick in axis.get_minor_ticks():
                all_ids.add(id(tick.label1))
                all_ids.add(id(tick.label2))
        if not ax.get_visible():
            ghosts |= all_ids
            continue
        for axis in (ax.xaxis, ax.yaxis):
            try:
                v0, v1 = sorted(axis.get_view_interval())
                lo_t = axis.get_transform().transform([v0, v1])
            except Exception:  # noqa: BLE001
                continue
            for ticks in (axis.get_major_ticks(), axis.get_minor_ticks()):
                for tick in ticks:
                    try:
                        loc_t = float(np.asarray(
                            axis.get_transform().transform([tick.get_loc()]))[0])
                    except Exception:  # noqa: BLE001
                        ghosts.add(id(tick.label1))
                        ghosts.add(id(tick.label2))
                        continue
                    if not (lo_t[0] - 1e-9 <= loc_t <= lo_t[1] + 1e-9):
                        ghosts.add(id(tick.label1))
                        ghosts.add(id(tick.label2))
    return ghosts


def corner_note(ax, text: str, where: str = "tl") -> None:
    """small white-backed note in a corner; chosen corners rarely collide."""
    xy = {"tl": (0.02, 0.98), "tr": (0.98, 0.98), "bl": (0.02, 0.02),
          "br": (0.98, 0.02)}[where]
    ha = {"tl": "left", "tr": "right", "bl": "left", "br": "right"}[where]
    ax.text(*xy, text, transform=ax.transAxes, fontsize=8, va="top", ha=ha,
            bbox=dict(fc="white", ec="none", alpha=0.65), zorder=6)


def leg(ax, ncol: int = 1, loc: str = "best") -> None:
    if not ax.get_legend_handles_labels()[0]:
        return  # nothing labelled; calling legend() would warn
    ax.legend(fontsize=8, framealpha=0.85, handlelength=1.3,
              labelspacing=0.25, borderpad=0.3, ncol=ncol, loc=loc)


def set_log_y(ax) -> None:
    """log y-scale, skipped when no plottable positive data exists.

    matplotlib's log locator raises 'Data cannot be log-scaled' at draw time
    when autoscaled limits are nonpositive (e.g. an all-zero probe slab); in
    that case the panel is left linear (it renders empty instead of crashing).
    """
    vals = []
    for ln in ax.get_lines():
        y = np.asarray(ln.get_ydata(orig=False), dtype=float)
        vals.append(y[np.isfinite(y)])
    for coll in ax.collections:
        try:
            arrs = np.asarray(coll.get_offsets(), dtype=float)
            if arrs.ndim == 2 and arrs.size:
                vals.append(arrs[:, 1])
        except Exception:  # noqa: BLE001
            pass
    v = np.concatenate(vals) if vals else np.zeros(0)
    if v.size == 0 or not np.any(v > 0):
        return
    ax.set_yscale("log")  # noqa: the guarded call (do NOT rewrite to set_log_y)


def plot_med_band(ax, x, stack, color, label=None, lw=1.6, band=True,
                  thin=0, thin_color="0.45", band_alpha=0.16):
    """median line + percentile band (+ `thin` raw sample curves) on ax."""
    med, lo, hi = med_band(np.asarray(stack, dtype=float))
    if band and np.any(np.isfinite(lo)):
        ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, lw=0)
    ax.plot(x, med, color=color, lw=lw, label=label)
    if thin:
        for j in range(min(thin, len(stack))):
            ax.plot(x, pos(stack[j]), color=thin_color, lw=0.55, alpha=0.6)
    return med


def _heatmap(ax, M, x, y, cmap="magma", label=""):
    """pcolormesh of log10(|M|) with NaN holes; M shape = (len(y), len(x))."""
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.88")
    C = np.ma.masked_invalid(np.log10(pos(M)))
    im = ax.pcolormesh(x, y, C, shading="nearest", cmap=cm, rasterized=True)
    return im


def _wnum(label: str):
    m = re.search(r"w(\d+)", label)
    return int(m.group(1)) if m else None


def _scalar(x, default: float = 0.0) -> float:
    """robust extraction of a python scalar from 0-d arrays / lists / scalars."""
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr.item())
        return float(arr.reshape(-1)[0])
    except Exception:  # noqa: BLE001
        return float(default)


# --------------------------------------------------------------------------- #
# derived quantities on a1 stacks
# --------------------------------------------------------------------------- #
def a_stack(a1: dict, r: int, l: int) -> np.ndarray:
    """[n_dir, T] amplitude curves a(t) for radius index r, layer index l."""
    return np.asarray(a1["dh_norm"][r, :, l, :], dtype=float)


def growth(stack: np.ndarray) -> np.ndarray:
    """a(t)/a(1) per row; rows with a(1) <= 0 become all-NaN."""
    return safe_ratio(stack, stack[:, :1])


def stepratio(stack: np.ndarray) -> np.ndarray:
    """a(t)/a(t-1), shape [n_dir, T-1] (t = 2..T)."""
    return safe_ratio(stack[:, 1:], stack[:, :-1])


def lam_local(stack: np.ndarray, k: int) -> np.ndarray:
    """(1/k) ln(a(t)/a(t-k)), shape [n_dir, T-k] (t = k+1..T)."""
    return np.log(pos(safe_ratio(stack[:, k:], stack[:, :-k]))) / k


def lam_cum(stack: np.ndarray) -> np.ndarray:
    """(1/t) ln(a(t)/a(1)), shape [n_dir, T]."""
    g = growth(stack)
    T = g.shape[1]
    return np.log(pos(g)) / t_axis(T)[None, :]


# --------------------------------------------------------------------------- #
# loaders (all tolerant: missing -> None)
# --------------------------------------------------------------------------- #
def _read_meta(run_dir: str) -> dict:
    p = os.path.join(run_dir, "meta.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            _warn(f"meta.json unreadable in {run_dir}: {e}")
    return {}


def _load_npz(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    except Exception as e:  # noqa: BLE001
        _warn(f"np.load failed for {path}: {e}")
        return None


def load_a1(run_dir: str, fname: str = "a1.npz") -> dict | None:
    d = _load_npz(os.path.join(run_dir, fname))
    if d is None or not {"dh_norm", "radii", "layers"} <= set(d):
        return None
    d["_meta"] = _read_meta(run_dir)
    d["_dir"] = run_dir
    return d


def load_state2(run_dir: str, fname: str = "state2.npz") -> dict | None:
    d = _load_npz(os.path.join(run_dir, fname))
    if d is None or not {"dh", "steps", "radii", "layers"} <= set(d):
        return None
    d["_meta"] = _read_meta(run_dir)
    d["_dir"] = run_dir
    return d


def load_layer_probe(path: str) -> dict | None:
    d = _load_npz(path)
    if d is None or not {"dh", "layers", "pos", "r"} <= set(d):
        return None
    d["_path"] = path
    return d


def load_free_run(run_dir: str, fname: str = "free_run.npz") -> dict | None:
    for cand in (os.path.join(run_dir, fname),
                 os.path.join(run_dir, "free_run", fname)):
        d = _load_npz(cand)
        if d is not None and {"obs", "radii"} <= set(d):
            d["_dir"] = run_dir
            return d
    return None


def load_h_last(path: str) -> dict | None:
    d = _load_npz(path)
    if d is None or "h_last" not in d:
        return None
    return d


def load_tf_pred(path: str) -> dict | None:
    d = _load_npz(path)
    if d is None or not {"pred", "ar", "ref", "radii"} <= set(d):
        return None
    return d


def load_depth_npz(path: str) -> dict | None:
    d = _load_npz(path)
    if d is None or "local_gain" not in d:
        return None
    return d


# --------------------------------------------------------------------------- #
# 1. flux-tube family (a1)
# --------------------------------------------------------------------------- #
def flux_tube_family(a1: dict, out_dir: str, tag: str, light: bool = False,
                     radii_sel=None) -> list[str]:
    """a(t)/a(1) and a(t)/a(t-1) variants, per layer + overlays + small mult."""
    dh = np.asarray(a1["dh_norm"], dtype=float)
    radii, layers = np.asarray(a1["radii"]), np.asarray(a1["layers"])
    R, U, L, T = dh.shape
    tag = _sanitize(tag)
    written: list[str] = []
    rsel = radii_sel if radii_sel is not None else select_indices(R, 1 if light else 3)
    rmid = rsel[len(rsel) // 2]
    t1, t2 = t_axis(T), t_axis(T - 1)
    fr = fmt_r(radii[rmid])

    def thin_stack(stack):
        ok = np.where(np.any(stack > 0, axis=1))[0]
        return ok[:N_THIN]

    for li in range(L):
        stack = a_stack(a1, rmid, li)
        if not np.any(stack > 0):
            _warn(f"flux: L={layers[li]} r={fr} all zeros -> skipped")
            continue
        g = growth(stack)
        thin_idx = thin_stack(g if np.any(np.isfinite(g)) else stack)
        Lname = layers[li]

        if np.any(np.isfinite(g)):
            fig, ax = _fig_panels(2, suptitle=f"{tag} | growth, r={fr}")
            for a, scale in zip(ax, ("semilog-y", "loglog")):
                plot_med_band(a, t1, g, "C0", band=True, thin=0)
                for j in thin_idx:
                    a.plot(t1, pos(g[j]), color="0.45", lw=0.55, alpha=0.6)
                set_log_y(a)
                if scale == "loglog":
                    a.set_xscale("log")
                a.set_xlabel("generated step t")
                a.set_title(f"a(t)/a(1) | L={Lname} | {scale}")
            ax[0].set_ylabel("a(t)/a(1)")
            corner_note(ax[0], f"median + {PLO}-{PHI}% band, n={n_valid(g)}/{U} dirs")
            p = os.path.join(out_dir, f"ft_growth__L{Lname}_r{fr}__{tag}.png")
            _save(fig, p)
            written.append(p)

        sr = stepratio(stack)
        if np.any(np.isfinite(sr)):
            fig, ax = _fig_panels(2, suptitle=f"{tag} | step ratio, r={fr}")
            for a, scale in zip(ax, ("semilog-y", "loglog")):
                plot_med_band(a, t2, sr, "C0", band=True)
                for j in thin_stack(sr):
                    a.plot(t2, pos(sr[j]), color="0.45", lw=0.55, alpha=0.6)
                set_log_y(a)
                if scale == "loglog":
                    a.set_xscale("log")
                a.set_xlabel("generated step t")
                a.set_title(f"a(t)/a(t-1) | L={Lname} | {scale}")
            ax[0].set_ylabel("a(t)/a(t-1)")
            corner_note(ax[0], f"median + {PLO}-{PHI}% band, n={n_valid(sr)}/{U} dirs")
            p = os.path.join(out_dir, f"ft_stepratio__L{Lname}_r{fr}__{tag}.png")
            _save(fig, p)
            written.append(p)

        if np.any(np.isfinite(g)) and np.any(np.isfinite(sr)):
            fig, ax = _fig_panels(2, suptitle=f"{tag} | both normalisations, r={fr}")
            for a, scale in zip(ax, ("semilog-y", "loglog")):
                plot_med_band(a, t1, g, "C0", label="a(t)/a(1)")
                plot_med_band(a, t2, sr, "C1", label="a(t)/a(t-1)")
                set_log_y(a)
                if scale == "loglog":
                    a.set_xscale("log")
                a.set_xlabel("generated step t")
                a.set_title(f"normalised by a(1) and by a(t-1) | L={Lname} | {scale}")
                leg(a)
            ax[0].set_ylabel("normalised amplitude")
            p = os.path.join(out_dir, f"ft_bothnorm__L{Lname}_r{fr}__{tag}.png")
            _save(fig, p)
            written.append(p)

    # ---- per-layer small multiples + all-layer overlays, per selected radius
    for ri in rsel:
        fr2 = fmt_r(radii[ri])
        for kind, fn, x in (("growth", growth, t1), ("ratio", stepratio, t2)):
            batches = [layers[i:i + MAX_PANELS] for i in range(0, L, MAX_PANELS)]
            for bi, batch in enumerate(batches):
                idxs = list(range(bi * MAX_PANELS, bi * MAX_PANELS + len(batch)))
                fig, ax = _fig_panels(len(batch), suptitle=f"{tag} | {kind}, r={fr2}")
                for a, li in zip(ax, idxs):
                    st = fn(a_stack(a1, ri, li))
                    plot_med_band(a, x, st, "C0", band=True)
                    set_log_y(a)
                    a.set_xlabel("t")
                    a.set_title(f"L={layers[li]}")
                ax[0].set_ylabel("a(t)/a(1)" if kind == "growth" else "a(t)/a(t-1)")
                p = os.path.join(out_dir,
                                 f"ft_{kind}_sm__r{fr2}_L{batch[0]}-{batch[-1]}__{tag}.png")
                _save(fig, p)
                written.append(p)

            fig, ax = _fig_panels(2, suptitle=f"{tag} | {kind} overlay, r={fr2}")
            cols = seq_colors(L)
            for a, scale in zip(ax, ("semilog-y", "loglog")):
                for li in range(L):
                    st = fn(a_stack(a1, ri, li))
                    med, _, _ = med_band(st)
                    a.plot(x, med, color=cols[li], lw=1.3, label=f"L={layers[li]}")
                set_log_y(a)
                if scale == "loglog":
                    a.set_xscale("log")
                a.set_xlabel("generated step t")
                a.set_title(f"{'a(t)/a(1)' if kind == 'growth' else 'a(t)/a(t-1)'}"
                            f" | median over dirs | {scale}")
                if L <= 10:
                    leg(a, ncol=2)
            ax[0].set_ylabel("a(t)/a(1)" if kind == "growth" else "a(t)/a(t-1)")
            p = os.path.join(out_dir, f"ft_{kind}_overlay__r{fr2}__{tag}.png")
            _save(fig, p)
            written.append(p)

    # ---- r_c(t): critical radius vs time (sentinel-masked)
    if "r_c" in a1:
        rc = np.asarray(a1["r_c"], dtype=float)
        rc = np.where(rc < RC_SENTINEL, rc, np.nan)
        if np.any(np.isfinite(rc)):
            fig, ax = _fig_panels(2, suptitle=f"{tag} | critical radius r_c(t)")
            cols = seq_colors(len(rsel), "plasma")
            for a, scale in zip(ax, ("lin-x", "log-x")):
                for ri, c in zip(rsel, cols):
                    plot_med_band(a, t1, rc[ri], c, label=f"r={fmt_r(radii[ri])}")
                set_log_y(a)
                if scale == "log-x":
                    a.set_xscale("log")
                a.set_xlabel("generated step t")
                a.set_title(f"r_c(t) | median+{PLO}-{PHI}% over dirs | {scale}")
                leg(a)
            ax[0].set_ylabel("r_c(t)")
            p = os.path.join(out_dir, f"ft_rc__{tag}.png")
            _save(fig, p)
            written.append(p)

    # ---- exploratory: direction-alignment of last-layer perturbation vectors
    if "dh_last" in a1:
        dl = np.asarray(a1["dh_last"], dtype=float)  # [R, 3, T, D]
        if dl.shape[1] >= 2:
            fig, ax = _fig_panels(2, suptitle=f"{tag} | perturbation alignment, r={fr}")
            vi = [j for j in range(dl.shape[1])
                  if np.any(dl[rmid, j] > 0)][:3]
            # panel 0: pairwise cos between directions
            a = ax[0]
            pairs = [(vi[i], vi[j]) for i in range(len(vi)) for j in range(i + 1, len(vi))]
            for c, (i, j) in zip(seq_colors(max(len(pairs), 1), "cool"), pairs):
                A, B = pos(dl[rmid, i]), pos(dl[rmid, j])
                cs = np.einsum("td,td->t", A, B) / (
                    np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
                a.plot(t1, cs, color=c, lw=1.2, label=f"cos(d{i},d{j})")
            a.set_xlabel("generated step t")
            a.set_title("cos(delta_i(t), delta_j(t)) | last-layer vectors")
            a.set_ylim(-1.05, 1.05)
            if pairs:
                leg(a)
            # panel 1: temporal alignment
            a = ax[1]
            for c, i in zip(seq_colors(max(len(vi), 1)), vi):
                A = pos(dl[rmid, i])
                cs = np.einsum("td,td->t", A[1:], A[:-1]) / (
                    np.linalg.norm(A[1:], axis=1) * np.linalg.norm(A[:-1], axis=1))
                a.plot(t1[1:], cs, color=c, lw=1.2, label=f"dir {i}")
            a.set_xlabel("generated step t")
            a.set_title("cos(delta(t), delta(t-1)) | last-layer vectors")
            a.set_ylim(-1.05, 1.05)
            leg(a)
            p = os.path.join(out_dir, f"ft_paircos__r{fr}__{tag}.png")
            _save(fig, p)
            written.append(p)

    # ---- exploratory: cos key (alignment of successive perturbations per layer)
    if "cos" in a1:
        cosk = np.asarray(a1["cos"], dtype=float)
        if cosk.ndim == 4:
            fig, ax = _fig_panels(2, suptitle=f"{tag} | cos key, r={fr}")
            cols = seq_colors(L)
            for a, (xlab, ttl) in zip(ax, (("t (all)", "cos(t), all layers"),
                                           ("t (first 60)", "cos(t), zoom t<=60"))):
                sl = slice(None) if "all" in ttl else slice(0, min(60, T))
                for li in range(L):
                    med, _, _ = med_band(cosk[rmid, :, li, sl])
                    a.plot(t1[sl], med, color=cols[li], lw=1.1,
                           label=f"L={layers[li]}")
                a.set_xlabel(xlab)
                a.set_title(ttl)
                a.set_ylim(-1.05, 1.05)
                if L <= 10:
                    leg(a, ncol=2)
            ax[0].set_ylabel("cos (schema key)")
            p = os.path.join(out_dir, f"ft_cos__r{fr}__{tag}.png")
            _save(fig, p)
            written.append(p)

    return written


# --------------------------------------------------------------------------- #
# 2. lambda family (a1)
# --------------------------------------------------------------------------- #
def lambda_family(a1: dict, out_dir: str, tag: str, ks=KS,
                  light: bool = False) -> list[str]:
    dh = np.asarray(a1["dh_norm"], dtype=float)
    radii, layers = np.asarray(a1["radii"]), np.asarray(a1["layers"])
    R, U, L, T = dh.shape
    tag = _sanitize(tag)
    written: list[str] = []
    rsel = select_indices(R, 1 if light else 3)
    rmid = rsel[len(rsel) // 2]
    ldeep = L - 1
    fr = fmt_r(radii[rmid])
    ks = [k for k in ks if k < T - 1] or [min(ks)]

    def panels(stack_fn, per_panel_titles, out_name, supt, colors=None,
               curve_labels=None):
        fig, ax = _fig_panels(len(per_panel_titles), suptitle=supt)
        for a, ttl in zip(ax, per_panel_titles):
            for ci, st in enumerate(stack_fn()):
                c = (colors[ci] if colors is not None else f"C{ci}")
                lab = curve_labels[ci] if curve_labels is not None else None
                plot_med_band(a, st[1], st[0], c, label=lab, band=True)
                a.set_xlabel("generated step t")
                a.set_title(ttl)
            a.set_ylabel("rate per step")
        p = os.path.join(out_dir, out_name)
        _save(fig, p)
        written.append(p)

    # per-layer: local rates + cumulative
    for li in range(L):
        stack = a_stack(a1, rmid, li)
        if not np.any(stack > 0):
            continue

        def stacks_lin():
            out = []
            for k in ks:
                lam = lam_local(stack, k)
                out.append((lam, t_axis(T - k)))
            out.append((lam_cum(stack), t_axis(T)))
            return out

        titles = [f"lam_k, k={k} | L={layers[li]}" for k in ks] + \
                 [f"Lam(t) cumulative | L={layers[li]}"]
        panels(stacks_lin, titles,
               f"lam_local__L{layers[li]}_r{fr}__{tag}.png",
               f"{tag} | local rates, r={fr}")
        if light:
            continue

        def stacks_logx():
            return stacks_lin()

        fig, ax = _fig_panels(len(ks) + 1,
                              suptitle=f"{tag} | local rates (log-x), r={fr}")
        for a, k in zip(ax, ks):
            lam = lam_local(stack, k)
            plot_med_band(a, t_axis(T - k), lam, "C0", band=True)
            a.set_xscale("log")
            a.set_xlabel("t")
            a.set_ylabel("rate per step")
            a.set_title(f"lam_k, k={k} | log-x | L={layers[li]}")
        plot_med_band(ax[-1], t_axis(T), lam_cum(stack), "C0", band=True)
        ax[-1].set_xscale("log")
        ax[-1].set_xlabel("t")
        ax[-1].set_ylabel("rate per step")
        ax[-1].set_title(f"Lam(t) cumulative | log-x | L={layers[li]}")
        corner_note(ax[0], f"median + {PLO}-{PHI}% band, n={U} dirs")
        p = os.path.join(out_dir, f"lam_local_logx__L{layers[li]}_r{fr}__{tag}.png")
        _save(fig, p)
        written.append(p)

    # over layers (at r_mid)
    for k in ks + [0]:
        fig, ax = _fig_panels(2, suptitle=f"{tag} | {'lam_k' if k else 'Lam'} over layers, r={fr}")
        cols = seq_colors(L)
        for a, scale in zip(ax, ("lin-x", "log-x")):
            for li in range(L):
                stack = a_stack(a1, rmid, li)
                if not np.any(stack > 0):
                    continue
                if k:
                    lam = lam_local(stack, k)
                    med, _, _ = med_band(lam)
                    a.plot(t_axis(T - k), med, color=cols[li], lw=1.2,
                           label=f"L={layers[li]}")
                else:
                    lam = lam_cum(stack)
                    med, _, _ = med_band(lam)
                    a.plot(t_axis(T), med, color=cols[li], lw=1.2,
                           label=f"L={layers[li]}")
            if scale == "log-x":
                a.set_xscale("log")
            a.set_xlabel("generated step t")
            a.set_ylabel("rate per step")
            a.set_title(("lam_k" if k else "Lam(t)") +
                        (f", k={k} | " if k else " | ") + f"over layers | {scale}")
            if L <= 10:
                leg(a, ncol=2)
        name = f"lam_overlayers__k{k}_r{fr}__{tag}.png" if k else \
               f"lamcum_overlayers__r{fr}__{tag}.png"
        p = os.path.join(out_dir, name)
        _save(fig, p)
        written.append(p)

    # over radii (at deepest layer)
    for k in ks + [0]:
        fig, ax = _fig_panels(2,
                             suptitle=f"{tag} | {'lam_k' if k else 'Lam'} over radii, L={layers[ldeep]}")
        cols = seq_colors(len(rsel), "plasma")
        for a, scale in zip(ax, ("lin-x", "log-x")):
            for c, ri in zip(cols, rsel):
                stack = a_stack(a1, ri, ldeep)
                if k:
                    lam = lam_local(stack, k)
                    plot_med_band(a, t_axis(T - k), lam, c,
                                   label=f"r={fmt_r(radii[ri])}")
                else:
                    lam = lam_cum(stack)
                    plot_med_band(a, t_axis(T), lam, c,
                                   label=f"r={fmt_r(radii[ri])}")
            if scale == "log-x":
                a.set_xscale("log")
            a.set_xlabel("generated step t")
            a.set_ylabel("rate per step")
            a.set_title(("lam_k" if k else "Lam(t)") +
                        (f", k={k} | " if k else " | ") +
                        f"over radii | {scale}")
            leg(a)
        name = f"lam_overradii__k{k}_L{layers[ldeep]}__{tag}.png" if k else \
               f"lamcum_overradii__L{layers[ldeep]}__{tag}.png"
        p = os.path.join(out_dir, name)
        _save(fig, p)
        written.append(p)

    return written


# --------------------------------------------------------------------------- #
# 3. window comparison family (list of a1 runs sharing a prompt)
# --------------------------------------------------------------------------- #
def _match_runs(runs: list[dict]):
    """common radius (mid of value-intersection), deepest common layer, min T."""
    def rr(a1):
        return np.asarray(a1["radii"], dtype=float)

    def ll(a1):
        return np.asarray(a1["layers"], dtype=int)

    rsets = [set(np.round(rr(r["a1"]), 6).tolist()) for r in runs]
    common = set.intersection(*rsets)
    if not common:
        return None
    radii_common = sorted(common)
    radius = radii_common[len(radii_common) // 2]
    lsets = [set(ll(r["a1"]).tolist()) for r in runs]
    layers_common = set.intersection(*lsets)
    layer = max(layers_common)
    T = min(int(r["a1"]["dh_norm"].shape[-1]) for r in runs)
    return radius, layer, T


def _a1_index(a1, radius: float, layer: int):
    ri = int(np.argmin(np.abs(np.asarray(a1["radii"], float) - radius)))
    li = int(np.argmin(np.abs(np.asarray(a1["layers"], int) - layer)))
    return ri, li


def window_family(runs: list[dict], out_dir: str, tag: str,
                  thr: float = DEPARTURE_THR) -> list[str]:
    """runs: [{'label': 'w55', 'a1': a1_dict, 'free_run': fr_dict|None}, ...]"""
    tag = _sanitize(tag)
    written: list[str] = []
    runs = [r for r in runs if r.get("a1") is not None]
    if len(runs) < 2:
        _warn(f"window_family {tag}: <2 usable runs")
        return written
    m = _match_runs(runs)
    if m is None:
        _warn(f"window_family {tag}: no common radius")
        return written
    radius, layer, T = m
    nums = [(_wnum(r["label"]) if _wnum(r["label"]) is not None else 10 ** 6, r)
            for r in runs]
    nums.sort(key=lambda x: x[0])
    order = [r for _, r in nums]
    wnums = [w for w, _ in nums]
    labels = [r["label"] for r in order]
    t1 = t_axis(T)
    cols = seq_colors(len(order))

    meds, bands = [], []
    for r in order:
        ri, li = _a1_index(r["a1"], radius, layer)
        g = growth(a_stack(r["a1"], ri, li)[:, :T])
        med, lo, hi = med_band(g)
        meds.append(med)
        bands.append((lo, hi))

    # 1) growth curves, semilogy + loglog
    fig, ax = _fig_panels(2, suptitle=f"{tag} | window comparison, r={fmt_r(radius)}, L={layer}")
    for a, scale in zip(ax, ("semilog-y", "loglog")):
        for c, lab, med, (lo, hi) in zip(cols, labels, meds, bands):
            if len(order) <= 6:
                a.fill_between(t1, lo, hi, color=c, alpha=0.12, lw=0)
            a.plot(t1, med, color=c, lw=1.5, label=lab)
        set_log_y(a)
        if scale == "loglog":
            a.set_xscale("log")
        a.set_xlabel("generated step t")
        a.set_title(f"a(t)/a(1) | median over dirs | {scale}")
        leg(a)
    ax[0].set_ylabel("a(t)/a(1)")
    p = os.path.join(out_dir, f"win_growth__{tag}.png")
    _save(fig, p)
    written.append(p)

    # 2) ratio to the widest window
    iw = int(np.argmax(wnums))
    fig, ax = _fig_panels(2, suptitle=f"{tag} | ratio to widest ({labels[iw]})")
    for a, scale in zip(ax, ("semilog-y", "loglog")):
        for c, lab, med in zip(cols, labels, meds):
            if lab == labels[iw]:
                continue
            ra = safe_ratio(med, meds[iw])
            a.plot(t1, ra, color=c, lw=1.4, label=f"{lab}/{labels[iw]}")
        a.axhline(1.0, color="k", lw=0.7, ls=":")
        set_log_y(a)
        if scale == "loglog":
            a.set_xscale("log")
        a.set_xlabel("generated step t")
        a.set_title(f"median growth ratio to widest window | {scale}")
        leg(a)
    ax[0].set_ylabel(f"a_w(t) / a_{labels[iw]}(t)")
    p = os.path.join(out_dir, f"win_ratio_widest__{tag}.png")
    _save(fig, p)
    written.append(p)

    # 3) first departure
    dep = []
    for med in meds:
        hit = np.where(np.isfinite(med) & (med >= thr))[0]
        dep.append(int(hit[0]) + 1 if len(hit) else np.nan)
    fig, ax = _fig_panels(2, suptitle=f"{tag} | first departure (a(t)/a(1)>={thr:g})")
    for c, lab, med, d in zip(cols, labels, meds, dep):
        ax[0].plot(t1, med, color=c, lw=1.3, label=lab)
        if np.isfinite(d):
            ax[0].axvline(d, color=c, lw=0.8, ls=":", alpha=0.7)
    set_log_y(ax[0])
    ax[0].set_xlabel("generated step t")
    ax[0].set_title("growth + departure tick")
    ax[0].set_ylabel("a(t)/a(1)")
    leg(ax[0])
    finite_x = [w for w, d in zip(wnums, dep) if np.isfinite(d)]
    finite_y = [d for d in dep if np.isfinite(d)]
    ax[1].scatter(finite_x, finite_y, marker="o", s=42, color="C0",
                  label=f"n={len(finite_y)}")
    n_none = sum(1 for d in dep if not np.isfinite(d))
    if n_none:
        none_x = [w for w, d in zip(wnums, dep) if not np.isfinite(d)]
        ax[1].scatter(none_x, [T] * len(none_x), marker="x", s=42, color="C3",
                      label=f"never within T (n={n_none})")
    ax[1].set_xlabel("window size (tokens)")
    ax[1].set_ylabel("first departure step")
    ax[1].set_title(f"departure vs window | thr={thr:g}")
    leg(ax[1])
    p = os.path.join(out_dir, f"win_departure__{tag}.png")
    _save(fig, p)
    written.append(p)

    # 4) lambda_k over windows: one panel per window
    k = 10 if 10 < T - 1 else max(1, T // 10)
    lams = []
    for r in order:
        ri, li = _a1_index(r["a1"], radius, layer)
        lams.append(lam_local(a_stack(r["a1"], ri, li)[:, :T], k))
    nb = int(np.ceil(len(order) / MAX_PANELS))
    for b in range(nb):
        sl = order[b * MAX_PANELS:(b + 1) * MAX_PANELS]
        lsl = lams[b * MAX_PANELS:(b + 1) * MAX_PANELS]
        labsl = labels[b * MAX_PANELS:(b + 1) * MAX_PANELS]
        fig, ax = _fig_panels(len(sl), suptitle=f"{tag} | lam k={k} per window")
        for a, lab, lam in zip(ax, labsl, lsl):
            plot_med_band(a, t_axis(T - k), lam, "C0", band=True)
            a.set_xlabel("t")
            a.set_ylabel(f"lam_k (k={k})")
            a.set_title(f"{lab} | r={fmt_r(radius)} L={layer}")
            corner_note(a, f"n={n_valid(lam)} dirs")
        p = os.path.join(out_dir, f"win_lam_k{k}_b{b}__{tag}.png")
        _save(fig, p)
        written.append(p)

    # 5) margin difference dm
    if all("dm" in r["a1"] for r in order):
        fig, ax = _fig_panels(2, suptitle=f"{tag} | logit-margin difference dm(t)")
        for c, lab, r in zip(cols, labels, order):
            ri, _li = _a1_index(r["a1"], radius, layer)
            dms = np.asarray(r["a1"]["dm"], dtype=float)[ri, :, :T]
            med, lo, hi = med_band(dms)
            ax[0].plot(t1, med, color=c, lw=1.4, label=lab)
            ax[0].fill_between(t1, lo, hi, color=c, alpha=0.12, lw=0)
            surv = np.mean(dms >= 0, axis=0)
            ax[1].plot(t1, surv, color=c, lw=1.4, label=lab)
        ax[0].axhline(0, color="k", lw=0.7, ls=":")
        ax[0].set_xlabel("generated step t")
        ax[0].set_title("median dm(t) | band=10-90%")
        ax[0].set_ylabel("dm (logit margin diff)")
        leg(ax[0])
        ax[1].set_xlabel("generated step t")
        ax[1].set_ylabel("fraction with dm>=0")
        ax[1].set_title("margin survival")
        p = os.path.join(out_dir, f"win_dm__{tag}.png")
        _save(fig, p)
        written.append(p)

    # 6) token agreement (only where free-run obs exists at matched radius)
    frs = []
    for r in order:
        fr = r.get("free_run")
        if fr is None:
            continue
        obs = np.asarray(fr["obs"], dtype=int)
        rrad = np.asarray(fr["radii"], dtype=float)
        j = np.argmin(np.abs(rrad - radius)) if len(rrad) else None
        if j is None:
            continue
        frs.append((r["label"], obs[j]))
    if len(frs) >= 2:
        fig, ax = _fig_panels(2, suptitle=f"{tag} | token agreement (free run), r~{fmt_r(radius)}")
        for c, (lab, obs) in zip(cols[:len(frs)], frs):
            agree = np.array([np.mean((obs < 0) | (obs > tt)) for tt in t1])
            ax[0].plot(t1, agree, color=c, lw=1.5, label=lab)
            flips = obs[obs > 0]
            ax[1].scatter([_wnum(lab)] * len(flips), flips, s=14, color=c, alpha=0.6)
        nnone = sum(1 for _, obs in frs for o in obs if o < 0)
        ax[0].set_xlabel("generated step t")
        ax[0].set_ylabel("fraction agreeing with ref")
        ax[0].set_title("token agreement survival")
        leg(ax[0])
        ax[1].set_xlabel("window size")
        ax[1].set_ylabel("first flip step")
        ax[1].set_title(f"per-dir flips (x = no flip: n={nnone})")
        p = os.path.join(out_dir, f"win_tokenagree__{tag}.png")
        _save(fig, p)
        written.append(p)
    else:
        _warn(f"window_family {tag}: free-run obs missing -> token agreement skipped")

    return written


def lambda_window_family(runs: list[dict], out_dir: str, tag: str,
                         ks=KS) -> list[str]:
    """lambda_k panels, one panel per window group (window sweep variant)."""
    tag = _sanitize(tag)
    written = []
    runs = [r for r in runs if r.get("a1") is not None]
    if len(runs) < 2:
        return written
    m = _match_runs(runs)
    if m is None:
        return written
    radius, layer, T = m
    nums = sorted([(_wnum(r["label"]) or 10 ** 6, r) for r in runs], key=lambda x: x[0])
    order = [r for _, r in nums]
    labels = [r["label"] for r in order]
    ks = [k for k in ks if k < T - 1]
    for k in ks:
        nb = int(np.ceil(len(order) / MAX_PANELS))
        for b in range(nb):
            sl = order[b * MAX_PANELS:(b + 1) * MAX_PANELS]
            labsl = labels[b * MAX_PANELS:(b + 1) * MAX_PANELS]
            fig, ax = _fig_panels(len(sl), suptitle=f"{tag} | lam k={k} per window")
            for a, lab, r in zip(ax, labsl, sl):
                ri, li = _a1_index(r["a1"], radius, layer)
                lam = lam_local(a_stack(r["a1"], ri, li)[:, :T], k)
                plot_med_band(a, t_axis(T - k), lam, "C0", band=True)
                a.set_xlabel("t")
                a.set_ylabel(f"lam_k (k={k})")
                a.set_title(f"{lab} | r={fmt_r(radius)} L={layer}")
            p = os.path.join(out_dir, f"lamwin_k{k}_b{b}__{tag}.png")
            _save(fig, p)
            written.append(p)
    return written


# --------------------------------------------------------------------------- #
# 4. depth family (a1)
# --------------------------------------------------------------------------- #
def depth_family(a1: dict, out_dir: str, tag: str,
                 tvals=None, light: bool = False) -> list[str]:
    dh = np.asarray(a1["dh_norm"], dtype=float)
    radii, layers = np.asarray(a1["radii"]), np.asarray(a1["layers"])
    R, U, L, T = dh.shape
    tag = _sanitize(tag)
    written = []
    rsel = select_indices(R, 1 if light else 3)
    tvals = [t for t in (tvals or [1, 2, 5, 10, 25, 50, 100, 200, 300]) if t <= T]
    tvals = tvals[:8]

    for ri in rsel:
        fr = fmt_r(radii[ri])
        stack_all = np.stack([a_stack(a1, ri, li) for li in range(L)])  # [L,U,T]
        resp = np.stack([med_band(stack_all[li, :, :])[0] for li in range(L)])  # [L,T]
        gain = np.stack([med_band(growth(stack_all[li]))[0] for li in range(L)])

        # response vs layer at fixed t
        half = max(1, len(tvals) // 2)
        for bi, tv in enumerate((tvals[:half], tvals[half:])):
            if not tv:
                continue
            fig, ax = _fig_panels(1, suptitle=f"{tag} | response vs layer, r={fr}")
            cols = seq_colors(len(tvals), "cividis")
            for t, c in zip(tv, cols):
                ax[0].plot(layers, pos(resp[:, t - 1]), marker="o", ms=3.5,
                           color=c, lw=1.2, label=f"t={t}")
            set_log_y(ax[0])
            ax[0].set_xlabel("layer")
            ax[0].set_ylabel("||delta h|| (median over dirs)")
            ax[0].set_title(f"a(t) vs layer | t in {tv[0]}..{tv[-1]}")
            leg(ax[0])
            p = os.path.join(out_dir, f"depth_response__r{fr}_t{tv[0]}-{tv[-1]}__{tag}.png")
            _save(fig, p)
            written.append(p)

        # gain vs layer at fixed t
        for bi, tv in enumerate((tvals[:half], tvals[half:])):
            if not tv:
                continue
            fig, ax = _fig_panels(1, suptitle=f"{tag} | gain vs layer, r={fr}")
            cols = seq_colors(len(tvals), "cividis")
            for t, c in zip(tv, cols):
                ax[0].plot(layers, pos(gain[:, t - 1]), marker="o", ms=3.5,
                           color=c, lw=1.2, label=f"t={t}")
            set_log_y(ax[0])
            ax[0].set_xlabel("layer")
            ax[0].set_ylabel("a(t)/a(1) (median over dirs)")
            ax[0].set_title(f"gain a(t)/a(1) vs layer | t in {tv[0]}..{tv[-1]}")
            leg(ax[0])
            p = os.path.join(out_dir, f"depth_gain__r{fr}_t{tv[0]}-{tv[-1]}__{tag}.png")
            _save(fig, p)
            written.append(p)

        # heatmaps layer x t: absolute + gain
        fig, ax = _fig_panels(2, suptitle=f"{tag} | depth heatmaps, r={fr}")
        im = _heatmap(ax[0], resp, t_axis(T), layers)
        fig.colorbar(im, ax=ax[0], shrink=0.85, pad=0.02,
                      label="log10 ||dh||")
        ax[0].set_xlabel("generated step t")
        ax[0].set_ylabel("layer")
        ax[0].set_title("||delta h||, median over dirs")
        im = _heatmap(ax[1], gain, t_axis(T), layers)
        fig.colorbar(im, ax=ax[1], shrink=0.85, pad=0.02, label="log10 gain")
        ax[1].set_xlabel("generated step t")
        ax[1].set_ylabel("layer")
        ax[1].set_title("gain a(t)/a(1), median over dirs (grey = undefined)")
        p = os.path.join(out_dir, f"depth_heatmap__r{fr}__{tag}.png")
        _save(fig, p)
        written.append(p)

    return written


# --------------------------------------------------------------------------- #
# 5. gamma / state family (state2.npz)
# --------------------------------------------------------------------------- #
def _state_stack(st: dict, r: int, s: int, l: int) -> np.ndarray:
    """[n_dir, T] response after injection step index s."""
    return np.asarray(st["dh"][r, s, :, l, :], dtype=float)


def _state_lag(stack: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """amplitude vs lag-0.. and lag axis, given 0-based injection index s."""
    return stack[:, s:], np.arange(0, stack.shape[1] - s)


def state_family(st: dict, out_dir: str, tag: str, light: bool = False) -> list[str]:
    dh = np.asarray(st["dh"], dtype=float)
    steps = np.asarray(st["steps"], dtype=int)
    radii, layers = np.asarray(st["radii"]), np.asarray(st["layers"])
    R, S, U, L, T = dh.shape
    tag = _sanitize(tag)
    written = []
    rsel = select_indices(R, 1 if light else 3)
    rmid = rsel[len(rsel) // 2]
    inj = _scalar(st.get("inject_layer", 0))
    # noise floor: median of pre-injection values (first injection step)
    pre_end = int(steps[0])
    floor = float(np.nanmedian(dh[:, 0, :, :, :max(pre_end, 1)])) if pre_end > 0 else np.nan

    ldeep = L - 1
    for ri in rsel:
        fr = fmt_r(radii[ri])
        # ---- response vs lag for each injection step (deepest layer)
        lsel = range(L) if ri == rmid else [ldeep]
        for li in lsel:
            fig, ax = _fig_panels(2, suptitle=
                                  f"{tag} | response vs lag, r={fr}, L={layers[li]}")
            cols = seq_colors(S, "viridis")
            for a, norm in zip(ax, ("absolute", "normalised by a(0)")):
                for si, c in zip(range(S), cols):
                    stck = _state_stack(st, ri, si, li)
                    resp, lagx = _state_lag(stck, int(steps[si]))
                    med, _, _ = med_band(resp)
                    if norm == "absolute":
                        a.plot(lagx, med, color=c, lw=1.3, label=f"s={steps[si]}")
                    else:
                        a.plot(lagx, safe_ratio(med, med[0]), color=c, lw=1.3,
                               label=f"s={steps[si]}")
                if np.isfinite(floor):
                    if norm == "absolute":
                        a.axhline(floor, color="k", lw=0.8, ls="--", alpha=0.7)
                        corner_note(a, f"pre-injection noise floor ~{floor:.2g}", "bl")
                set_log_y(a)
                a.set_xlabel("lag (steps after injection)")
                a.set_title(f"a(s+tau) | {norm} | inj layer={int(inj)}")
                if S <= 9:
                    leg(a)
            ax[0].set_ylabel("||delta h||")
            ax[1].set_ylabel("a(s+tau)/a(s)")
            p = os.path.join(out_dir, f"gamma_lag__r{fr}_L{layers[li]}__{tag}.png")
            _save(fig, p)
            written.append(p)

        # ---- depth profile at fixed lags
        lags = [lg for lg in (0, 1, 5, 25) if lg < T - int(steps[0])][:4] or [0]
        fig, ax = _fig_panels(len(lags), suptitle=f"{tag} | depth profile, r={fr}")
        for a, lag in zip(ax, lags):
            cols = seq_colors(S, "viridis")
            for si, c in zip(range(S), cols):
                prof = np.stack([med_band(_state_stack(st, ri, si, li))[0]
                                 for li in range(L)])
                a.plot(layers, pos(prof[:, lag]), color=c, lw=1.3,
                       label=f"s={steps[si]}")
            set_log_y(a)
            a.set_xlabel("readout layer")
            a.set_ylabel("||delta h||")
            a.set_title(f"depth profile at lag {lag}")
            if S <= 9:
                leg(a, ncol=2)
        p = os.path.join(out_dir, f"gamma_depth__r{fr}__{tag}.png")
        _save(fig, p)
        written.append(p)

        # ---- latency scatter
        fig, ax = _fig_panels(2, suptitle=f"{tag} | latency, r={fr}, L={layers[ldeep]}")
        for a, kind in zip(ax, ("growth", "margin")):
            for si in range(S):
                s0 = int(steps[si])
                stck = _state_lag(_state_stack(st, ri, si, ldeep), s0)[0]
                base = stck[:, 0]
                if kind == "growth":
                    hit = stck >= LATENCY_THR * np.where(base > 0, base, np.nan)[:, None]
                    hit[:, 0] = False
                    lat = np.where(hit.any(1), np.argmax(hit, 1), np.nan)
                else:
                    if "dm" not in st:
                        continue
                    dms = np.asarray(st["dm"], dtype=float)[ri, si, :, s0:]
                    fl = dms < -DM_FLIP_TOL
                    lat = np.where(fl.any(1), np.argmax(fl, 1), np.nan)
                xs = np.full(len(lat), s0)
                fin = np.isfinite(lat)
                a.scatter(xs[fin], lat[fin], s=13, color="C0", alpha=0.45,
                          label="dirs" if si == 0 else None)
                if np.any(~fin):
                    a.scatter([s0], [T - s0], marker="x", s=40, color="C3",
                              alpha=0.8, label="none within T" if si == 0 else None)
                if np.any(fin):
                    a.scatter([s0], [np.nanmedian(lat)], s=52, color="C1",
                              zorder=5, label="median" if si == 0 else None)
            a.set_xlabel("injection step s")
            a.set_ylabel("latency (lag steps)")
            a.set_title(("a(s+tau) >= %.0f a(s)" % LATENCY_THR) if kind == "growth"
                        else f"dm < -{DM_FLIP_TOL:g} (margin flip)")
            leg(a)
        p = os.path.join(out_dir, f"gamma_latency__r{fr}__{tag}.png")
        _save(fig, p)
        written.append(p)

        # ---- heatmaps (batches of <=4 injection steps) only for the mid radius
        if ri == rmid:
            nb = int(np.ceil(S / MAX_PANELS))
            for b in range(nb):
                sl = list(range(b * MAX_PANELS, min((b + 1) * MAX_PANELS, S)))
                fig, ax = _fig_panels(len(sl), suptitle=f"{tag} | state heatmap, r={fr}")
                for a, si in zip(ax, sl):
                    M = np.stack([med_band(_state_stack(st, ri, si, li))[0]
                                  for li in range(L)])  # [L,T]
                    im = _heatmap(a, M, t_axis(T), layers)
                    fig.colorbar(im, ax=a, shrink=0.8, pad=0.02,
                                  label="log10||dh||")
                    a.axvline(int(steps[si]), color="w", lw=1.0, ls="--")
                    a.set_xlabel("generated step t")
                    a.set_ylabel("readout layer")
                    a.set_title(f"inj step s={steps[si]}")
                p = os.path.join(out_dir, f"state_heatmap__r{fr}_b{b}__{tag}.png")
                _save(fig, p)
                written.append(p)

    return written


# --------------------------------------------------------------------------- #
# 6. layer-probe family + comparison
# --------------------------------------------------------------------------- #
def layer_probe_family(lp: dict, out_dir: str, tag: str) -> list[str]:
    dh = np.asarray(lp["dh"], dtype=float)      # [n_inj, n_dir, 28, T]
    inj_layers = np.asarray(lp["layers"], dtype=int)
    pos0 = int(_scalar(lp["pos"]))
    r = _scalar(lp["r"])
    N, U, L28, T = dh.shape
    tag = _sanitize(tag)
    written = []

    # heatmaps readout-layer x t, one panel per injection layer (batches of 4)
    nb = int(np.ceil(N / MAX_PANELS))
    for b in range(nb):
        sl = list(range(b * MAX_PANELS, min((b + 1) * MAX_PANELS, N)))
        fig, ax = _fig_panels(len(sl), suptitle=f"{tag} | probe heatmaps (pos={pos0}, r={fmt_r(r)})")
        for a, i in zip(ax, sl):
            M = np.stack([med_band(dh[i, :, l, :])[0] for l in range(L28)])
            im = _heatmap(a, M, t_axis(T), np.arange(L28))
            fig.colorbar(im, ax=a, shrink=0.8, pad=0.02, label="log10||dh||")
            a.axhline(inj_layers[i] + 0.5, color="w", lw=1.0, ls="--")
            a.axvline(pos0, color="w", lw=1.0, ls="--")
            a.set_xlabel("generated step t")
            a.set_ylabel("readout layer")
            ttl = f"inj L={inj_layers[i]}"
            if not np.any(M > 0):
                ttl += " (no response)"
            a.set_title(ttl)
        p = os.path.join(out_dir, f"lp_heatmap__b{b}__{tag}.png")
        _save(fig, p)
        written.append(p)

    # depth profile at fixed lags: curves = injection layers
    lags = [lg for lg in (0, 1, 5, 25) if lg < T - pos0][:4] or [0]
    fig, ax = _fig_panels(len(lags), suptitle=f"{tag} | probe depth profiles (pos={pos0})")
    cols = seq_colors(N, "plasma")
    for a, lag in zip(ax, lags):
        for i, c in zip(range(N), cols):
            prof = np.stack([med_band(dh[i, :, l, :])[0] for l in range(L28)])
            a.plot(np.arange(L28), pos(prof[:, pos0 + lag]), color=c, lw=1.3,
                   label=f"inj L={inj_layers[i]}")
        set_log_y(a)
        a.set_xlabel("readout layer")
        a.set_ylabel("||delta h||")
        a.set_title(f"profile at lag {lag}")
        if N <= 10:
            leg(a, ncol=2)
    p = os.path.join(out_dir, f"lp_depth_lag__{tag}.png")
    _save(fig, p)
    written.append(p)

    # growth overlay at the final readout layer
    lread = L28 - 1
    fig, ax = _fig_panels(2, suptitle=f"{tag} | growth at readout L={lread}, r={fmt_r(r)}")
    cols = seq_colors(N, "plasma")
    t1 = t_axis(T)
    for a, scale in zip(ax, ("semilog-y", "loglog")):
        for i, c in zip(range(N), cols):
            g = growth(dh[i, :, lread, :])
            if not np.any(np.isfinite(g)):
                continue
            med, _, _ = med_band(g)
            a.plot(t1, med, color=c, lw=1.3, label=f"inj L={inj_layers[i]}")
        set_log_y(a)
        if scale == "loglog":
            a.set_xscale("log")
        a.set_xlabel("generated step t")
        a.set_title(f"a(t)/a(1) | median over dirs | {scale}")
        if N <= 10:
            leg(a, ncol=2)
    ax[0].set_ylabel("a(t)/a(1)")
    p = os.path.join(out_dir, f"lp_growth_overlay__{tag}.png")
    _save(fig, p)
    written.append(p)
    return written


def layer_probe_compare(files: list[tuple[str, dict]], out_dir: str,
                        tag: str) -> list[str]:
    """files: [(label, lp_dict), ...] -> compare positions / radii."""
    tag = _sanitize(tag)
    written = []
    if len(files) < 2:
        return written
    lread = np.asarray(files[0][1]["dh"]).shape[2] - 1

    fig, ax = _fig_panels(2, suptitle=f"{tag} | probe growth comparison (readout L={lread})")
    cols = seq_colors(len(files), "tab10")
    for a, scale in zip(ax, ("semilog-y", "loglog")):
        for c, (lab, lp) in zip(cols, files):
            dh = np.asarray(lp["dh"], dtype=float)
            N = dh.shape[0]
            imid = N // 2
            g = growth(dh[imid, :, lread, :])
            med, _, _ = med_band(g)
            a.plot(t_axis(dh.shape[-1]), med, color=c, lw=1.5, label=lab)
        set_log_y(a)
        if scale == "loglog":
            a.set_xscale("log")
        a.set_xlabel("generated step t")
        a.set_title(f"a(t)/a(1) | middle injection layer | {scale}")
        leg(a)
    ax[0].set_ylabel("a(t)/a(1)")
    p = os.path.join(out_dir, f"lpc_growth__{tag}.png")
    _save(fig, p)
    written.append(p)

    # immediate depth profile at lag 0 and lag 5
    for lag in (0, 5):
        fig, ax = _fig_panels(1, suptitle=f"{tag} | probe lag-{lag} depth profiles")
        for c, (lab, lp) in zip(cols, files):
            dh = np.asarray(lp["dh"], dtype=float)
            N, U, L28, T = dh.shape
            imid = N // 2
            pos0 = int(_scalar(lp["pos"]))
            prof = np.stack([med_band(dh[imid, :, l, :])[0] for l in range(L28)])
            ax[0].plot(np.arange(L28), pos(prof[:, min(pos0 + lag, T - 1)]),
                       color=c, lw=1.5, label=f"{lab} (inj L={np.asarray(lp['layers'])[imid]})")
        set_log_y(ax[0])
        ax[0].set_xlabel("readout layer")
        ax[0].set_ylabel("||delta h||")
        ax[0].set_title(f"depth profile at lag {lag} | median over dirs")
        leg(ax[0])
        p = os.path.join(out_dir, f"lpc_lag{lag}__{tag}.png")
        _save(fig, p)
        written.append(p)
    return written


# --------------------------------------------------------------------------- #
# 7. aggregate family (pool across directions and prompts)
# --------------------------------------------------------------------------- #
def aggregate_family(runs: list[dict], out_dir: str, tag: str,
                     ks=KS) -> list[str]:
    """runs: [{'label','a1'}]; pools (prompt, dir) samples at matched radius/layer."""
    tag = _sanitize(tag)
    written = []
    runs = [r for r in runs if r.get("a1") is not None]
    if len(runs) < 2:
        _warn(f"aggregate {tag}: <2 runs")
        return written
    m = _match_runs(runs)
    if m is None:
        _warn(f"aggregate {tag}: no common radius")
        return written
    radius, layer, T = m
    t1 = t_axis(T)

    g_parts, sr_parts, lam_parts, cum_parts, dm_parts, per_run_med = \
        [], [], {k: [] for k in ks}, [], [], []
    for r in runs:
        ri, li = _a1_index(r["a1"], radius, layer)
        st = a_stack(r["a1"], ri, li)[:, :T]
        g = growth(st)
        if np.any(np.isfinite(g)):
            g_parts.append(g)
            per_run_med.append((r["label"], med_band(g)[0]))
        sr = stepratio(st)
        if np.any(np.isfinite(sr)):
            sr_parts.append(sr)
        for k in ks:
            lam = lam_local(st, k)
            lam_parts[k].append(lam)
        cum_parts.append(lam_cum(st))
        if "dm" in r["a1"]:
            dm_parts.append(np.asarray(r["a1"]["dm"], dtype=float)[ri, :, :T])
    n = sum(p.shape[0] for p in g_parts)
    nruns = len(per_run_med)
    sub = f"r={fmt_r(radius)},L={layer},n={n}({nruns}x)"
    os.makedirs(out_dir, exist_ok=True)

    def pooled_curve(parts):
        return np.concatenate(parts, axis=0) if parts else None

    # pooled growth
    gp = pooled_curve(g_parts)
    if gp is not None:
        fig, ax = _fig_panels(2, suptitle=f"{tag} | pooled growth | {sub}")
        for a, scale in zip(ax, ("semilog-y", "loglog")):
            med, lo, hi = med_band(gp)
            a.fill_between(t1, lo, hi, color="C0", alpha=0.18, lw=0)
            a.plot(t1, med, color="C0", lw=1.8, label=f"pooled median (n={n})")
            for lab, pm in per_run_med:
                a.plot(t1, pm, color="0.55", lw=0.6, alpha=0.5)
            set_log_y(a)
            if scale == "loglog":
                a.set_xscale("log")
            a.set_xlabel("generated step t")
            a.set_title(f"a(t)/a(1) | pooled median + {PLO}-{PHI}% band | {scale}")
            leg(a)
        ax[0].set_ylabel("a(t)/a(1)")
        corner_note(ax[0], f"grey thin = per-run medians (n_runs={nruns})", "bl")
        p = os.path.join(out_dir, f"agg_growth__r{fmt_r(radius)}_L{layer}__{tag}.png")
        _save(fig, p)
        written.append(p)

    # pooled step ratio
    srp = pooled_curve(sr_parts)
    if srp is not None:
        fig, ax = _fig_panels(2, suptitle=f"{tag} | pooled step ratio | {sub}")
        for a, scale in zip(ax, ("semilog-y", "loglog")):
            med, lo, hi = med_band(srp)
            a.fill_between(t_axis(T - 1), lo, hi, color="C0", alpha=0.18, lw=0)
            a.plot(t_axis(T - 1), med, color="C0", lw=1.8, label=f"pooled (n={n})")
            set_log_y(a)
            if scale == "loglog":
                a.set_xscale("log")
            a.set_xlabel("generated step t")
            a.set_title(f"a(t)/a(t-1) | pooled median + {PLO}-{PHI}% band | {scale}")
            leg(a)
        ax[0].set_ylabel("a(t)/a(t-1)")
        p = os.path.join(out_dir, f"agg_stepratio__r{fmt_r(radius)}_L{layer}__{tag}.png")
        _save(fig, p)
        written.append(p)

    # pooled local rates + cumulative (lin-x and log-x)
    for xscale in ("lin", "log"):
        fig, ax = _fig_panels(len(ks) + 1, suptitle=f"{tag} | pooled rates ({xscale}-x) | {sub}")
        for a, k in zip(ax, ks):
            lp = pooled_curve(lam_parts[k])
            if lp is None:
                continue
            plot_med_band(a, t_axis(T - k), lp, "C0", label=f"pooled (n={n})")
            if xscale == "log":
                a.set_xscale("log")
            a.set_xlabel("generated step t")
            a.set_ylabel(f"lam_k (k={k})")
            a.set_title(f"lam_k, k={k} | pooled median + {PLO}-{PHI}% band")
        cp = pooled_curve(cum_parts)
        if cp is not None:
            plot_med_band(ax[-1], t1, cp, "C0", label=f"pooled (n={n})")
            if xscale == "log":
                ax[-1].set_xscale("log")
            ax[-1].set_xlabel("generated step t")
            ax[-1].set_ylabel("Lam(t)")
            ax[-1].set_title("cumulative Lam(t) | pooled median + band")
        p = os.path.join(out_dir, f"agg_lam_{'logx' if xscale == 'log' else 'linx'}__r{fmt_r(radius)}_L{layer}__{tag}.png")
        _save(fig, p)
        written.append(p)

    # pooled margin difference
    if dm_parts:
        dmp = pooled_curve(dm_parts)
        fig, ax = _fig_panels(2, suptitle=f"{tag} | pooled dm | {sub}")
        med, lo, hi = med_band(dmp)
        ax[0].fill_between(t1, lo, hi, color="C0", alpha=0.18, lw=0)
        ax[0].plot(t1, med, color="C0", lw=1.6)
        ax[0].axhline(0, color="k", lw=0.7, ls=":")
        ax[0].set_xlabel("generated step t")
        ax[0].set_ylabel("dm")
        ax[0].set_title(f"dm(t) pooled median + {PLO}-{PHI}% band")
        surv = np.mean(dmp >= 0, axis=0)
        ax[1].plot(t1, surv, color="C0", lw=1.6)
        ax[1].set_xlabel("generated step t")
        ax[1].set_ylabel("fraction dm>=0")
        ax[1].set_title("margin survival")
        p = os.path.join(out_dir, f"agg_dm__r{fmt_r(radius)}_L{layer}__{tag}.png")
        _save(fig, p)
        written.append(p)
    return written


# --------------------------------------------------------------------------- #
# 8. h_last family
# --------------------------------------------------------------------------- #
def h_last_family(files: list[tuple[str, dict]], out_dir: str,
                  tag: str, max_curves: int = 8) -> list[str]:
    tag = _sanitize(tag)
    written = []
    files = files[:max_curves]
    if not files:
        return written
    cols = seq_colors(len(files), "tab10")

    # norm + relative step size
    fig, ax = _fig_panels(2, suptitle=f"{tag} | last-layer state norm")
    for c, (lab, d) in zip(cols, files):
        h = np.asarray(d["h_last"], dtype=float)
        T = h.shape[0]
        ax[0].plot(t_axis(T), np.linalg.norm(h, axis=1), color=c, lw=1.1, label=lab)
        step = np.linalg.norm(np.diff(h, axis=0), axis=1) / np.maximum(
            np.linalg.norm(h[1:], axis=1), 1e-12)
        ax[1].plot(t_axis(T - 1), step, color=c, lw=0.9, label=lab)
    ax[0].set_xlabel("generated step t")
    ax[0].set_ylabel("||h_last(t)||")
    ax[0].set_title("state norm")
    leg(ax[0], ncol=2)
    ax[1].set_xlabel("generated step t")
    ax[1].set_ylabel("||h(t)-h(t-1)|| / ||h(t)||")
    ax[1].set_title("relative step size")
    p = os.path.join(out_dir, f"hlast_norm__{tag}.png")
    _save(fig, p)
    written.append(p)

    # pairwise divergence + self drift
    fig, ax = _fig_panels(2, suptitle=f"{tag} | trajectory divergence")
    pairs = [(i, i + 1) for i in range(0, min(len(files) - 1, 3))]
    pcols = seq_colors(max(len(pairs), 1), "cool")
    for c, (i, j) in zip(pcols, pairs):
        ha = np.asarray(files[i][1]["h_last"], dtype=float)
        hb = np.asarray(files[j][1]["h_last"], dtype=float)
        T = min(ha.shape[0], hb.shape[0])
        d = np.linalg.norm(ha[:T] - hb[:T], axis=1) / np.maximum(
            np.linalg.norm(ha[:T], axis=1), 1e-12)
        ax[0].plot(t_axis(T), d, color=c, lw=1.2,
                   label=f"{files[i][0]} vs {files[j][0]}")
    for c, (lab, d0) in zip(cols, files):
        h = np.asarray(d0["h_last"], dtype=float)
        T = h.shape[0]
        drift = np.linalg.norm(h - h[0], axis=1) / max(np.linalg.norm(h[0]), 1e-12)
        ax[1].plot(t_axis(T), drift, color=c, lw=1.0, label=lab)
    ax[0].set_xlabel("generated step t")
    ax[0].set_ylabel("||h_a(t)-h_b(t)||/||h_a(t)||")
    ax[0].set_title("cross-prompt divergence (prompt pairs)")
    leg(ax[0])
    ax[1].set_xlabel("generated step t")
    ax[1].set_ylabel("||h(t)-h(1)||/||h(1)||")
    ax[1].set_title("self drift from t=1")
    leg(ax[1], ncol=2)
    p = os.path.join(out_dir, f"hlast_pairdiv__{tag}.png")
    _save(fig, p)
    written.append(p)

    # consecutive-state cosine (exploratory)
    fig, ax = _fig_panels(2, suptitle=f"{tag} | consecutive-state cosine")
    for c, (lab, d0) in zip(cols, files):
        h = np.asarray(d0["h_last"], dtype=float)
        cs = np.einsum("td,td->t", h[1:], h[:-1]) / (
            np.linalg.norm(h[1:], axis=1) * np.linalg.norm(h[:-1], axis=1))
        T = h.shape[0]
        ax[0].plot(t_axis(T - 1)[: min(200, T - 1)], cs[: min(200, T - 1)],
                   color=c, lw=0.9, label=lab)
        ax[1].plot(t_axis(T - 1), cs, color=c, lw=0.8, label=lab)
    ax[0].set_xlabel("generated step t")
    ax[0].set_ylabel("cos(h(t-1), h(t))")
    ax[0].set_title("zoom t<=200")
    ax[1].set_xlabel("generated step t")
    ax[1].set_title("full horizon")
    if len(files) <= 5:
        leg(ax[0], ncol=2)
    p = os.path.join(out_dir, f"hlast_cos__{tag}.png")
    _save(fig, p)
    written.append(p)
    return written


# --------------------------------------------------------------------------- #
# 9. token-agreement families
# --------------------------------------------------------------------------- #
def token_survival_family(frs: list[tuple[str, dict]], out_dir: str,
                          tag: str, T: int = 300) -> list[str]:
    """frs: [(label, free_run_dict with obs [n_r, n_dir], radii)]"""
    tag = _sanitize(tag)
    written = []
    if not frs:
        return written
    rcols = seq_colors(8, "plasma")
    lss = ["-", "--", ":", "-."]
    fig, ax = _fig_panels(2, suptitle=f"{tag} | free-run token agreement")
    n_lab = 0
    rng = np.random.default_rng(0)
    for fi, (lab, fr) in enumerate(frs):
        obs = np.asarray(fr["obs"], dtype=int)
        radii = np.asarray(fr["radii"], dtype=float)
        T = max(int(obs[obs > 0].max()) if np.any(obs > 0) else 300, 50)
        for ri in range(obs.shape[0]):
            agree = np.array([np.mean((obs[ri] < 0) | (obs[ri] > tt))
                              for tt in t_axis(T)])
            ax[0].plot(t_axis(T), agree, color=rcols[ri % len(rcols)],
                       ls=lss[fi % len(lss)], lw=1.4,
                       label=f"{lab} r={fmt_r(radii[ri])}")
            flips = obs[ri][obs[ri] > 0]
            if len(flips):
                ax[1].scatter(np.full(len(flips), ri) + rng.uniform(
                    -.15, .15, len(flips)), flips, s=15, alpha=0.6,
                    color=rcols[ri % len(rcols)], marker=["o", "s", "^", "D"][fi % 4])
            n_lab += 1
    ax[0].set_xlabel("generated step t")
    ax[0].set_ylabel("fraction agreeing with reference")
    ax[0].set_title("token-agreement survival curves")
    if n_lab <= 10:
        leg(ax[0])
    ax[1].set_xlabel("radius index")
    ax[1].set_ylabel("first flip step")
    ax[1].set_title("per-direction first flips (marker = source file)")
    p = os.path.join(out_dir, f"tok_survival__{tag}.png")
    _save(fig, p)
    written.append(p)
    return written


def tf_pred_family(tfs: list[tuple[str, dict]], out_dir: str,
                   tag: str) -> list[str]:
    """tfs: [(label, tf_pred dict with pred/ar [n_r,n_dir,T], ref, radii)]"""
    tag = _sanitize(tag)
    written = []
    if not tfs:
        return written
    rcols = seq_colors(6, "plasma")
    lss = ["-", "--", ":", "-."]
    fig, ax = _fig_panels(2, suptitle=f"{tag} | token agreement: teacher-forced vs free-run")
    for fi, (lab, tf) in enumerate(tfs):
        pred = np.asarray(tf["pred"], dtype=int)
        ar = np.asarray(tf["ar"], dtype=int)
        ref = np.asarray(tf["ref"], dtype=int)
        radii = np.asarray(tf["radii"], dtype=float)
        T = ref.shape[0]
        for ri in range(pred.shape[0]):
            ag_tf = np.mean(pred[ri] == ref[None, :], axis=0)
            ag_ar = np.mean(ar[ri] == ref[None, :], axis=0)
            ax[0].plot(t_axis(T), ag_tf, color=rcols[ri % len(rcols)],
                       ls=lss[fi % len(lss)], lw=1.4,
                       label=f"{lab} r={fmt_r(radii[ri])}")
            ax[1].plot(t_axis(T), ag_ar, color=rcols[ri % len(rcols)],
                       ls=lss[fi % len(lss)], lw=1.4,
                       label=f"{lab} r={fmt_r(radii[ri])}")
    ax[0].set_xlabel("generated step t")
    ax[0].set_ylabel("fraction pred == ref")
    ax[0].set_title("teacher-forced agreement (pred vs ref)")
    if sum(t[1]["pred"].shape[0] for t in tfs) <= 10:
        leg(ax[0])
    ax[1].set_xlabel("generated step t")
    ax[1].set_ylabel("fraction ar == ref")
    ax[1].set_title("free-run agreement (ar vs ref)")
    p = os.path.join(out_dir, f"tok_tfpred__{tag}.png")
    _save(fig, p)
    written.append(p)
    return written


# --------------------------------------------------------------------------- #
# 10. state window comparison
# --------------------------------------------------------------------------- #
def state_window_family(states: list[tuple[str, dict]], out_dir: str,
                        tag: str) -> list[str]:
    """states: [(label, state2_dict)] with matched radii/layers -> latency vs window."""
    tag = _sanitize(tag)
    written = []
    states = [(l, s) for l, s in states if s is not None]
    if len(states) < 2:
        return written
    # match: mid radius, deepest layer
    rsets = [set(np.round(np.asarray(s["radii"], float), 6).tolist())
             for _, s in states]
    common = set.intersection(*rsets)
    if not common:
        _warn(f"state_window {tag}: no common radius")
        return written
    radius = sorted(common)[len(common) // 2]
    lsets = [set(np.asarray(s["layers"], int).tolist()) for _, s in states]
    layer = max(set.intersection(*lsets))
    T = min(int(s["dh"].shape[-1]) for _, s in states)
    order = sorted(states, key=lambda x: _wnum(x[0]) or 10 ** 6)
    cols = seq_colors(len(order))

    # response vs lag for the first <=4 windows
    sl = order[:MAX_PANELS]
    fig, ax = _fig_panels(len(sl), suptitle=f"{tag} | state response vs lag (r={fmt_r(radius)}, L={layer})")
    for a, (lab, s) in zip(ax, sl):
        ri = int(np.argmin(np.abs(np.asarray(s["radii"], float) - radius)))
        li = int(np.argmin(np.abs(np.asarray(s["layers"], int) - layer)))
        steps = np.asarray(s["steps"], int)
        dh = np.asarray(s["dh"], float)
        scol = seq_colors(len(steps), "viridis")
        for si, c in zip(range(len(steps)), scol):
            stck = dh[ri, si, :, li, :]
            lag_arr = stck[:, steps[si]:]
            med, _, _ = med_band(lag_arr)
            a.plot(np.arange(len(med)), med, color=c, lw=1.2, label=f"s={steps[si]}")
        set_log_y(a)
        a.set_xlabel("lag")
        a.set_ylabel("||delta h||")
        a.set_title(f"{lab}")
        if len(steps) <= 9:
            leg(a, ncol=2)
    p = os.path.join(out_dir, f"swin_lag__{tag}.png")
    _save(fig, p)
    written.append(p)

    # latency vs injection step per window
    fig, ax = _fig_panels(2, suptitle=f"{tag} | state latency vs window")
    for a, kind in zip(ax, ("growth", "margin")):
        for c, (lab, s) in zip(cols, order):
            ri = int(np.argmin(np.abs(np.asarray(s["radii"], float) - radius)))
            li = int(np.argmin(np.abs(np.asarray(s["layers"], int) - layer)))
            steps = np.asarray(s["steps"], int)
            dh = np.asarray(s["dh"], float)
            xs, ys = [], []
            for si in range(len(steps)):
                stck = dh[ri, si, :, li, steps[si]:]
                if kind == "growth":
                    base = stck[:, 0]
                    hit = stck >= LATENCY_THR * np.where(base > 0, base, np.nan)[:, None]
                    hit[:, 0] = False
                    lat = np.where(hit.any(1), np.argmax(hit, 1), np.nan)
                else:
                    dms = np.asarray(s.get("dm", np.full((1, 1, 1, T), np.nan)),
                                     float)[ri, si, :, steps[si]:]
                    if dms.ndim == 1:
                        dms = dms[None, :]
                    fl = dms < -DM_FLIP_TOL
                    lat = np.where(fl.any(1), np.argmax(fl, 1), np.nan)
                xs.append(steps[si])
                ys.append(np.nanmedian(lat) if np.any(np.isfinite(lat)) else np.nan)
            a.plot(xs, ys, color=c, lw=1.4, marker="o", ms=4, label=lab)
        a.set_xlabel("injection step s")
        a.set_ylabel("median latency (lag)")
        a.set_title(("a(s+tau)>=%.0f a(s)" % LATENCY_THR) if kind == "growth"
                    else "margin flip dm<0")
        leg(a)
    p = os.path.join(out_dir, f"swin_latency__{tag}.png")
    _save(fig, p)
    written.append(p)
    return written


# --------------------------------------------------------------------------- #
# 11. depth.npz bonus family (local_gain / lam per token)
# --------------------------------------------------------------------------- #
def depth_npz_family(dnp: dict, out_dir: str, tag: str) -> list[str]:
    tag = _sanitize(tag)
    written = []
    lg = np.asarray(dnp["local_gain"], dtype=float)   # [n_tok, 28]
    layers = np.asarray(dnp["layers"], int)
    tokens = np.asarray(dnp.get("tokens", np.arange(lg.shape[0])), int)
    fig, ax = _fig_panels(2, suptitle=f"{tag} | depth.npz per-token profiles")
    cols = seq_colors(len(tokens), "cividis")
    for c, row, tk in zip(cols, lg, tokens):
        ax[0].plot(layers, pos(row), color=c, lw=1.2, label=f"tok {tk}")
    set_log_y(ax[0])
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("local gain")
    ax[0].set_title("local_gain vs layer (per probe token)")
    if len(tokens) <= 10:
        leg(ax[0])
    hd = np.asarray(dnp.get("h_depth", np.zeros(len(tokens))), float)
    ax[1].bar(np.arange(len(tokens)), hd, color="C0", alpha=0.75)
    ax[1].set_xlabel("token index")
    ax[1].set_ylabel("h_depth")
    ax[1].set_title("h_depth per token")
    p = os.path.join(out_dir, f"depthnpz_gain__{tag}.png")
    _save(fig, p)
    written.append(p)

    if "lam" in dnp:
        lam = np.asarray(dnp["lam"], dtype=float)  # [n_tok, 28, m]
        m = lam.shape[-1]
        js = [0, m - 1] if m > 1 else [0]
        fig, ax = _fig_panels(len(js), suptitle=f"{tag} | depth.npz lam heatmaps")
        for a, j in zip(ax, js):
            im = _heatmap(a, lam[:, :, j], layers, np.arange(lam.shape[0]))
            fig.colorbar(im, ax=a, shrink=0.85, pad=0.02, label="log10 lam")
            a.set_xlabel("layer")
            a.set_ylabel("token index")
            a.set_title(f"lam[:, :, {j}] (schema lam[n_tok, 28, {m}])")
        p = os.path.join(out_dir, f"depthnpz_lam__{tag}.png")
        _save(fig, p)
        written.append(p)
    return written


# --------------------------------------------------------------------------- #
# 12. margins npz bonus: baseline hidden-state norms per layer
# --------------------------------------------------------------------------- #
def margins_norms_family(mnp: dict, out_dir: str, tag: str) -> list[str]:
    tag = _sanitize(tag)
    written = []
    hkeys = sorted([k for k in mnp if re.fullmatch(r"h\d+", k)],
                   key=lambda s: int(s[1:]))
    if not hkeys:
        return written
    T = mnp[hkeys[0]].shape[0]
    tv = select_indices(T, 5)
    fig, ax = _fig_panels(2, suptitle=f"{tag} | baseline hidden-state norms")
    cols = seq_colors(len(tv), "cividis")
    lay = [int(k[1:]) for k in hkeys]
    for t, c in zip(tv, cols):
        norms = [np.linalg.norm(mnp[k][t - 1]) for k in hkeys]
        ax[0].plot(lay, norms, color=c, lw=1.3, marker="o", ms=3, label=f"t={t}")
        drift = [np.linalg.norm(mnp[k][t - 1] - mnp[k][0]) /
                 max(np.linalg.norm(mnp[k][0]), 1e-12) for k in hkeys]
        ax[1].plot(lay, drift, color=c, lw=1.3, marker="o", ms=3, label=f"t={t}")
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("||h_l(t)||")
    ax[0].set_title("baseline norms vs layer")
    leg(ax[0])
    ax[1].set_xlabel("layer")
    ax[1].set_ylabel("||h_l(t)-h_l(1)||/||h_l(1)||")
    ax[1].set_title("drift from t=1 vs layer")
    leg(ax[1])
    p = os.path.join(out_dir, f"margins_norms__{tag}.png")
    _save(fig, p)
    written.append(p)
    return written
