"""
make_all.py -- produce the full grid of figures from run directories.

Usage
-----
    python make_all.py                       # demo run over results/local_data
    python make_all.py RUN_DIR [RUN_DIR ...] # explicit run directories
    python make_all.py --out DIR --data DIR --light --only SUBSTR

Each RUN_DIR is auto-detected by schema:
    a1.npz       -> flux-tube + lambda + depth families (+ r_c, cos, dh_last)
    state2.npz   -> gamma/state family
    layer-probe npz (keys dh/layers/pos/r) -> layer-probe family
    free_run.npz / fr npz (obs+radii) -> token survival
    h_last npz   -> h_last family

Outputs: <out>/figs/<group>/<figure>.png, <out>/manifest.csv, <out>/warnings.log
Self-contained: numpy/matplotlib only (plots.py next to this file).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plots  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _group_name(s: str) -> str:
    g = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s).strip("_")
    return g[:60] or "run"


def run_family(manifest, group, family, fn, *args, **kw):
    t0 = time.time()
    try:
        paths = fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        msg = f"{group}/{family} FAILED: {e}\n{traceback.format_exc(limit=3)}"
        plots._warn(msg)
        print(f"[make_all] {group}/{family} FAILED ({e})", flush=True)
        return
    dt = time.time() - t0
    for p in paths:
        manifest.append((os.path.abspath(p), group, family))
    print(f"[make_all] {group}/{family}: {len(paths)} figs in {dt:.1f}s",
          flush=True)


# --------------------------------------------------------------------------- #
# standalone processing of an arbitrary run directory
# --------------------------------------------------------------------------- #
def process_run_dir(rd: str, out_root: str, light: bool, manifest) -> None:
    group = _group_name(os.path.basename(os.path.normpath(rd)))
    a1 = plots.load_a1(rd)
    if a1 is not None:
        run_family(manifest, group, "flux_tube", plots.flux_tube_family,
                   a1, os.path.join(out_root, "figs", group), group, light=light)
        run_family(manifest, group, "lambda", plots.lambda_family,
                   a1, os.path.join(out_root, "figs", group), group, light=light)
        run_family(manifest, group, "depth", plots.depth_family,
                   a1, os.path.join(out_root, "figs", group), group, light=light)
    st = plots.load_state2(rd)
    if st is not None:
        run_family(manifest, group, "state", plots.state_family,
                   st, os.path.join(out_root, "figs", group), group, light=light)
    fr = plots.load_free_run(rd)
    if fr is not None:
        run_family(manifest, group, "token", plots.token_survival_family,
                   [(group, fr)], os.path.join(out_root, "figs", group), group)
    # layer-probe style npz files lying in the directory
    if os.path.isdir(rd):
        for f in sorted(os.listdir(rd)):
            if not f.endswith(".npz"):
                continue
            lp = plots.load_layer_probe(os.path.join(rd, f))
            if lp is not None:
                sub = _group_name(f"{group}_{os.path.splitext(f)[0]}")
                run_family(manifest, sub, "layer_probe", plots.layer_probe_family,
                           lp, os.path.join(out_root, "figs", sub), sub)
    hl = plots.load_h_last(os.path.join(rd, "h_last.npz"))
    if hl is not None:
        run_family(manifest, group, "h_last", plots.h_last_family,
                   [(group, hl)], os.path.join(out_root, "figs", group), group)


# --------------------------------------------------------------------------- #
# demo: the curated local_data tour
# --------------------------------------------------------------------------- #
def demo(data_root: str, out_root: str, only: str | None, manifest) -> None:
    L = lambda *p: os.path.join(data_root, *p)  # noqa: E731
    F = lambda *p: os.path.join(out_root, "figs", *p)  # noqa: E731

    def a1_runs(prefix, labels, sub=""):
        out = []
        for lab in labels:
            rd = L(*( [prefix] + ([sub] if sub else []) + [lab] ))
            if os.path.isdir(rd):
                a1 = plots.load_a1(rd)
                if a1 is not None:
                    out.append({"label": lab, "a1": a1,
                                "free_run": plots.load_free_run(rd)})
        return out

    # ---------------- 1. primary single-run families ----------------
    primary = L("w100_layers")
    if os.path.isdir(primary):
        a1 = plots.load_a1(primary)
        if a1 is not None:
            g = "w100_layers"
            run_family(manifest, g, "flux_tube", plots.flux_tube_family,
                       a1, F(g), g)
            run_family(manifest, g, "lambda", plots.lambda_family, a1, F(g), g)
            run_family(manifest, g, "depth", plots.depth_family, a1, F(g), g)

    light_a1 = [("window_sweep_w100", L("window_sweep", "w100")),
                ("scan_prompts_w100", L("big", "scan", "prompts_w100"))]
    for g, rd in light_a1:
        if os.path.isdir(rd):
            a1 = plots.load_a1(rd)
            if a1 is not None:
                run_family(manifest, g, "flux_tube", plots.flux_tube_family,
                           a1, F(g), g, light=True)
                run_family(manifest, g, "lambda", plots.lambda_family,
                           a1, F(g), g, light=True)
                run_family(manifest, g, "depth", plots.depth_family,
                           a1, F(g), g, light=True)

    # ---------------- 2. state runs ----------------
    state_runs = [("state_deep_emb", L("state_deep_emb")),
                  ("state_deep_L14", L("state_deep_L14")),
                  ("big_state_w100", L("big", "state", "w100")),
                  ("state_window_w50", L("state_window", "w50"))]
    for g, rd in state_runs:
        if os.path.isdir(rd):
            st = plots.load_state2(rd)
            if st is not None:
                run_family(manifest, g, "state", plots.state_family,
                           st, F(g), g,
                           light=(g != "state_deep_emb"))

    # ---------------- 3. layer probes ----------------
    probe_files = [("lp_pos50", L("layer_probe", "pos50.npz")),
                   ("lp_pos16", L("layer_probe", "pos16.npz")),
                   ("lp_r02", L("layer_probe", "r02.npz")),
                   ("lpm_p16", L("layer_probe_multi", "p16.npz")),
                   ("lpm_p75", L("layer_probe_multi", "p75.npz")),
                   ("lpm_p150", L("layer_probe_multi", "p150.npz")),
                   ("biglp_p16", L("big", "probe", "p16.npz")),
                   ("biglp_p50", L("big", "probe", "p50.npz")),
                   ("biglp_p100", L("big", "probe", "p100.npz")),
                   ("biglp_p200", L("big", "probe", "p200.npz"))]
    for g, p in probe_files:
        if only and only not in g:
            continue
        lp = plots.load_layer_probe(p)
        if lp is not None:
            run_family(manifest, g, "layer_probe", plots.layer_probe_family,
                       lp, F(g), g)

    # probe comparisons: position sweep + radius sweep
    pos_files = [("pos16", L("big", "probe", "p16.npz")),
                 ("pos50", L("big", "probe", "p50.npz")),
                 ("pos100", L("big", "probe", "p100.npz")),
                 ("pos200", L("big", "probe", "p200.npz"))]
    lps = [(lab, plots.load_layer_probe(p)) for lab, p in pos_files]
    lps = [(l, d) for l, d in lps if d is not None]
    if len(lps) >= 2:
        run_family(manifest, "lpc_pos", "layer_probe_compare",
                   plots.layer_probe_compare, lps, F("lpc_pos"), "big_probe_pos")
    rad_files = [("r0.05", L("layer_probe", "pos50.npz")),
                 ("r0.2", L("layer_probe", "r02.npz"))]
    lps = [(lab, plots.load_layer_probe(p)) for lab, p in rad_files]
    lps = [(l, d) for l, d in lps if d is not None]
    if len(lps) >= 2:
        run_family(manifest, "lpc_radius", "layer_probe_compare",
                   plots.layer_probe_compare, lps, F("lpc_radius"),
                   "layer_probe_radius")

    # ---------------- 4. window comparisons ----------------
    wgroups = [("win_prompts", "big", "scan", "prompts_w"),
               ("win_cyber", "big", "scan", "prompts_cyber_w"),
               ("win_long2", "big", "scan", "prompts_long2_w"),
               ("win_scanB1", "big", "scan", "scanB_1_w")]
    for g, *pref in wgroups:
        if only and only not in g:
            continue
        root = L(*pref[:-1])
        pfx = pref[-1]
        if not os.path.isdir(root):
            continue
        labs = sorted([d for d in os.listdir(root) if d.startswith(pfx)],
                      key=lambda s: _wn_or_inf(s))
        runs = a1_runs(os.path.join(*pref[:-1]), labs)
        if len(runs) >= 2:
            run_family(manifest, g, "window", plots.window_family, runs, F(g), g)
            run_family(manifest, g, "lambda_window", plots.lambda_window_family,
                       runs, F(g), g)

    # ---------------- 5. aggregates across prompts ----------------
    for g, base in [("agg_rate_prompts", ("rate_prompts",)),
                    ("agg_rate_w50", ("rate_w50",))]:
        if only and only not in g:
            continue
        root = L(*base)
        if not os.path.isdir(root):
            continue
        labs = sorted([d for d in os.listdir(root)
                       if os.path.isfile(os.path.join(root, d, "a1.npz"))])
        runs = a1_runs(*base, labs)
        if len(runs) >= 2:
            run_family(manifest, g, "aggregate", plots.aggregate_family,
                       runs, F(g), g)

    # ---------------- 6. state window comparison ----------------
    swins = []
    for sub in ["w50", "w70", "w90", "w100", "w150", "w300", "w4096"]:
        rd = L("big", "state", sub)
        if os.path.isdir(rd):
            st = plots.load_state2(rd)
            if st is not None:
                swins.append((sub, st))
    for sub in ["w50", "w200", "w4096"]:
        rd = L("state_window", sub)
        if os.path.isdir(rd):
            st = plots.load_state2(rd)
            if st is not None:
                swins.append(("swin_" + sub, st))
    if len(swins) >= 2:
        run_family(manifest, "swin_state", "state_window",
                   plots.state_window_family, swins, F("swin_state"), "state_windows")

    # ---------------- 7. h_last trajectories ----------------
    ll_dir = L("long_last")
    if os.path.isdir(ll_dir):
        files = []
        for f in sorted(os.listdir(ll_dir)):
            d = plots.load_h_last(os.path.join(ll_dir, f))
            if d is not None:
                files.append((os.path.splitext(f)[0][:18], d))
        if files:
            run_family(manifest, "h_last", "h_last",
                       plots.h_last_family, files[:8], F("h_last"), "long_last")

    # ---------------- 8. token agreement ----------------
    frs = []
    for lab, f in [("mid", "fr_w100_mid.npz"), ("extra", "fr_w100_extra.npz"),
                   ("cyber", "fr_cyber.npz"), ("childhood", "fr_childhood.npz")]:
        d = plots._load_npz(L(f))
        if d is not None and {"obs", "radii"} <= set(d):
            frs.append((lab, d))
    if frs:
        run_family(manifest, "tok_free", "token_survival",
                   plots.token_survival_family, frs, F("tok_free"), "fr_runs")

    tfs = []
    for lab, f in [("childhood", L("tf_pred", "childhood.npz")),
                   ("cyber", L("tf_pred", "cyber.npz")),
                   ("mid", L("tf_pred", "mid.npz")),
                   ("large", L("tf_pred", "large.npz"))]:
        d = plots.load_tf_pred(f)
        if d is not None:
            tfs.append((lab, d))
    if tfs:
        run_family(manifest, "tok_tf", "tf_pred", plots.tf_pred_family,
                   tfs, F("tok_tf"), "tf_pred_runs")

    # ---------------- 9. bonus: depth.npz + margins norms ----------------
    dnp = plots.load_depth_npz(L("depth", "depth.npz"))
    if dnp is not None:
        run_family(manifest, "depthnpz", "depth_npz", plots.depth_npz_family,
                   dnp, F("depthnpz"), "depth_npz")
    mnp = plots._load_npz(L("margins", "w100.npz"))
    if mnp is not None and "h1" in mnp:
        run_family(manifest, "margins_norms", "margins_norms",
                   plots.margins_norms_family, mnp, F("margins_norms"), "margins_w100")


def _wn_or_inf(s: str):
    m = re.search(r"w(\d+)", s)
    return int(m.group(1)) if m else 10 ** 6


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories to process")
    ap.add_argument("--out", default=os.path.join(HERE),
                    help="output root (default: alongside this script)")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "local_data"),
                    help="local_data root for the demo tour")
    ap.add_argument("--light", action="store_true",
                    help="trim per-layer variants and radii for big inputs")
    ap.add_argument("--only", default=None,
                    help="demo: only groups whose name contains SUBSTR")
    ap.add_argument("--max-figs", type=int, default=0,
                    help="abort after N figures (0 = no limit; smoke test)")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    data_root = os.path.abspath(args.data)
    os.makedirs(os.path.join(out_root, "figs"), exist_ok=True)
    manifest: list[tuple] = []
    plots.WARNINGS.clear()
    t0 = time.time()

    if args.runs:
        for rd in args.runs:
            if not os.path.isdir(rd):
                print(f"[make_all] not a directory: {rd}", flush=True)
                continue
            process_run_dir(os.path.abspath(rd), out_root, args.light, manifest)
    else:
        if not os.path.isdir(data_root):
            print(f"[make_all] data root not found: {data_root}", flush=True)
            return 2
        demo(data_root, out_root, args.only, manifest)

    # manifest + warnings
    with open(os.path.join(out_root, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["figure", "group", "family"])
        w.writerows(manifest)
    with open(os.path.join(out_root, "warnings.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(plots.WARNINGS))

    n = len(manifest)
    print(f"\n[make_all] wrote {n} figures to {os.path.join(out_root, 'figs')} "
          f"in {time.time() - t0:.1f}s", flush=True)
    print(f"[make_all] manifest.csv + warnings.log written", flush=True)
    if args.max_figs and n > args.max_figs:
        print("[make_all] MAX-FIGS exceeded (smoke-test guard)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
