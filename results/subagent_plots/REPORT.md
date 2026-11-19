# REPORT — subagent plotting module (flux-tube / rate / window / depth / state figures)

**Deliverables** (this folder): `plots.py` (reusable figure factory), `make_all.py`
(runner over run directories), `figs/<group>/*.png` (all generated figures),
`manifest.csv` (figure -> group -> family), `warnings.log` (QA + skipped panels).

## How to call the module

```bash
# full curated tour over results/local_data (what produced figs/):
python make_all.py

# arbitrary run directories (auto-detected by schema: a1.npz / state2.npz /
# layer-probe npz / free_run.npz / h_last npz):
python make_all.py path/to/runA path/to/runB ...
# options: --out DIR --data DIR --light --only SUBSTR
```

Programmatic use:

```python
import plots
a1 = plots.load_a1("results/local_data/w100_layers")           # a1.npz + meta.json
st  = plots.load_state2("results/local_data/state_deep_emb")    # state2.npz
lp  = plots.load_layer_probe("results/local_data/layer_probe/pos50.npz")

plots.flux_tube_family(a1, "figs/mygroup", "mytag")             # growth/ratio variants
plots.lambda_family(a1, "figs/mygroup", "mytag")                # lam_k + cumulative
plots.depth_family(a1, "figs/mygroup", "mytag")                 # response/gain/heatmap
plots.state_family(st, "figs/mygroup", "mytag")                 # lag/depth/latency
plots.layer_probe_family(lp, "figs/mygroup", "mytag")           # 28-layer probe
plots.window_family([{"label": "w55", "a1": a1_55, "free_run": fr_or_None},
                     {"label": "w130", "a1": a1_130}], "figs/g", "g")   # window compare
plots.aggregate_family([{"label": "p1", "a1": a1_p1}, ...], "figs/g", "g")  # pooled
```

Every family returns the list of PNG paths it wrote. `light=True` trims
per-layer variants and uses a single (middle) radius for large runs.

## Quantities and conventions

| symbol | definition | notes |
|---|---|---|
| `a(t)` | `||delta h_{t,l}||` = `dh_norm` | t = array index + 1 (generated step) |
| `g(t)` | `a(t)/a(1)` | growth, normalised by a(1) |
| `rho(t)` | `a(t)/a(t-1)` | consecutive-step ratio, t >= 2 |
| `lam_k(t)` | `(1/k) ln(a(t)/a(t-k))` | local rate, k in {5, 10, 25} |
| `Lam(t)` | `(1/t) ln(a(t)/a(1))` | cumulative rate |
| departure | first t with median `g(t) >= 2` | `DEPARTURE_THR` |
| latency | first lag `tau` with `a(s+tau) >= 2 a(s)` | state perturbations |
| margin flip | first t with `dm < -1e-3` | tol. for ~1e-5 fp noise pre-injection |
| token agreement(t) | fraction of rollouts with `obs > t` or `obs < 0` | free-run obs |

All aggregations: median line + **10–90 percentile band** (nan-aware); exact
zeros in `dh_norm` are treated as below-fp-noise and become NaN; the valid
sample count `n` is annotated. Figures: <= 4 panels, font sizes >= 8 pt,
`constrained_layout`, `bbox_inches='tight'`, dpi 190, each panel autoscaled
("free axes"), descriptive filenames `<family>__<variant>__<tag>.png`.

A built-in QA pass (`_qa_fig`) inspects every figure before saving and logs
clipped/overlapping text to `warnings.log` (ghost tick labels of out-of-view
ticks are excluded, since matplotlib keeps but never draws them).

## Schema assumptions (verified against local data)

1. **a1.npz**: `dh_norm[r,u,l,t]` with t index 0 = first generated step.
   Exact zeros (26% of entries at tiny radii) = sub-fp32 response -> NaN.
   `r_c` sentinel ~1e30 ("never crosses") is masked. `pred_T == -1` = no flip.
   `dh_last[r,3,T,D]` = last-layer perturbation vectors for 3 directions.
2. **state2.npz**: injection at step `s` (1-based label) first affects
   **time index s (0-based)** at every recorded layer >= `inject_layer`,
   where the response is ~ r at the injected layer itself. Values before
   injection are an fp32 noise floor (~1e-3 relative, not exact zero) — drawn
   as a dashed "noise floor" line on absolute-response panels. `dm` is
   exactly 0 pre-injection (flip test uses dm < -1e-3).
3. **layer_probe.npz** (`dh[n_inj, n_dir, 28, T]`): response is zero at
   readout layers <= injection layer; first appears at readout `inj+1`,
   time index `pos`, with value exactly r. Injection at the final layer (27)
   gives an **all-zero slab** (it only perturbs logits; no recorded state
   downstream) — such panels are labelled "(no response)".
4. **free_run.npz**: `obs[r,u]` = first flip step (-1 = never); older `fr_*.npz`
   files with the same keys + object `gen_ids` are handled (only `obs`/`radii`
   read).
5. **tf_pred npz**: `pred` = teacher-forced tokens under perturbation,
   `ar` = free-run tokens, `ref` = reference sequence; agreement curves are
   fraction matching `ref` over directions.
6. Pooling across runs requires a **common radius value** (matched to 6
   decimals, middle of the intersection) and uses the deepest common layer
   and the shortest T. Runs with differing direction counts are pooled
   raggedly (nan-aware).
