# REPORT — subagent plotting module (flux-tube / rate / window / depth / state figures)

**Deliverables** (this folder):

| file | what |
|---|---|
| `plots.py` | reusable figure factory (self-contained: numpy + matplotlib, Agg, CPU) |
| `make_all.py` | runner over run directories / curated demo tour |
| `figs/<group>/*.png` | **300 generated figures** |
| `manifest.csv` | figure → group → family mapping |
| `warnings.log` | skipped panels + QA text-overlap/clipping log (currently: 1 structural skip + 4 expected "no free-run data" notes; **zero QA overlap/clipping issues**) |

## 1. How to call the module

```bash
# reproduce everything in figs/ (~40 min on one CPU core):
python make_all.py

# arbitrary run directories (schema auto-detected per directory:
# a1.npz -> flux/lambda/depth; state2.npz -> state; probe npz -> layer_probe;
# free_run.npz -> token survival; h_last npz -> h_last):
python make_all.py path/to/runA path/to/runB ...
# options: --out DIR --data DIR --light --only SUBSTR
# --light = one (middle) radius and trimmed per-layer grids for big inputs
```

Programmatic use — every family returns the list of PNG paths written:

```python
import plots
a1 = plots.load_a1("results/local_data/w100_layers")            # a1.npz + meta.json
st = plots.load_state2("results/local_data/state_deep_emb")      # state2.npz
lp = plots.load_layer_probe("results/local_data/layer_probe/pos50.npz")
fr = plots.load_free_run("some_run")                             # obs [n_r, n_dir]

plots.flux_tube_family(a1, "figs/g", "mytag")                     # growth / ratio / bothnorm / sm / overlay / r_c / paircos
plots.lambda_family(a1, "figs/g", "mytag", ks=(5, 10, 25))        # lam_k + cumulative, lin-x and log-x
plots.depth_family(a1, "figs/g", "mytag")                         # response & gain vs layer, layer x t heatmaps
plots.state_family(st, "figs/g", "mytag")                         # gamma(lag), depth profiles, latency, state heatmaps
plots.layer_probe_family(lp, "figs/g", "mytag")                   # 28-layer probe heatmaps + profiles
plots.window_family([{"label": "w55", "a1": a1_55, "free_run": fr_or_None},
                     ...], "figs/g", "g")                         # growth, ratio-to-widest, departure, dm, token agreement
plots.aggregate_family([{"label": "p1", "a1": a1_p1}, ...], "figs/g", "g")   # pooled median + 10-90% band, n annotated
plots.h_last_family([(label, hlast_dict), ...], "figs/g", "g")
plots.token_survival_family([(label, fr_dict), ...], "figs/g", "g")
```

All figure families share these guarantees: ≤ 4 panels per figure, font
sizes ≥ 8 pt, `constrained_layout`, `bbox_inches='tight'`, dpi 190, each
panel autoscaled ("free axes"), descriptive filenames
`<family>__<variant>__<tag>.png`, and a built-in QA pass (`_qa_fig`) that
logs any clipped or mutually-overlapping text to `warnings.log`
(matplotlib's ghost tick labels of out-of-view ticks are excluded, since
they are kept but never drawn).

## 2. Quantities and conventions

| symbol | definition | notes |
|---|---|---|
| `a(t)` | `||delta h_{t,l}||` = `dh_norm` | t = array index + 1 (generated step) |
| `g(t)` | `a(t)/a(1)` | growth, normalised by a(1) |
| `rho(t)` | `a(t)/a(t-1)` | consecutive-step ratio, t ≥ 2 |
| `lam_k(t)` | `(1/k) ln(a(t)/a(t-k))` | local rate, k ∈ {5, 10, 25} |
| `Lam(t)` | `(1/t) ln(a(t)/a(1))` | cumulative rate |
| departure | first t with median `g(t) ≥ 2` | `DEPARTURE_THR` |
| latency | first lag `tau` with `a(s+tau) ≥ 2 a(s)` | state perturbations |
| margin flip | first t with `dm < -1e-3` | tolerance for ±1e-5 pre-injection fp noise |
| token agreement(t) | fraction of rollouts with `obs > t` or `obs < 0` | free-run `obs` |

Aggregation is always median + **10–90 percentile band** (nan-aware), with
the valid sample count `n` annotated (grey thin lines = per-run medians in
aggregate figures). Titles state the normalisation.

## 3. Produced figures (300 total, manifest.csv has the full list)

| group | family | figs | content |
|---|---|---|---|
| w100_layers | flux_tube / lambda / depth | 42 / 22 / 15 | primary a1 run (12 radii, 20 dirs, 8 layers): per-layer growth (semilog-y + loglog, 3 thin dirs + median/IQR band), step ratio, both-normalisations panel, per-radius small multiples, all-layer overlays, `r_c(t)`, last-layer pairwise-cos, per-layer λ_k + cumulative (lin-x, log-x), λ over layers/radii, depth response/gain vs layer at fixed t, layer×t heatmaps |
| window_sweep_w100, scan_prompts_w100 | flux/lambda/depth (light) | 36 | same families, trimmed radius set |
| state_deep_emb, state_deep_L14, big_state_w100, state_window_w50 | state | 34 | response vs lag per injection step (absolute + normalised by a(0), noise-floor line), depth profiles at fixed lag, growth/margin latency scatter, layer×t state heatmaps with injection marker |
| lp_pos50, lp_pos16, lp_r02, lpm_*, biglp_* (9 probe files) | layer_probe | 44 | 28×T heatmaps per injection layer (zero-slab panels labelled "(no response)"), depth profiles at lag 0/1/5/25, growth overlay at readout L27 |
| lpc_pos, lpc_radius | probe_compare | 6 | position sweep (pos16…pos200) and radius sweep (r=0.05 vs 0.2) |
| win_prompts, win_cyber, win_long2, win_scanB1 (w55…w130) | window + lambda_window | 70 | growth overlay, ratio-to-widest-window, first-departure marker + scatter, dm(t) + margin survival, λ_10 one-panel-per-window, Λcum |
| agg_rate_prompts, agg_rate_w50 | aggregate | 10 | pooled (5 prompts × dirs) growth / step ratio / λ_k / Λ / dm with 10–90% bands and n annotated |
| swin_state | state_window | 2 | response-vs-lag per window; growth & margin latency vs injection step across windows |
| h_last | h_last | 3 | state norm + relative step size; cross-prompt divergence + self drift; consecutive-state cosine |
| tok_free, tok_tf | token | 2 | free-run agreement survival + per-dir flip scatter; teacher-forced vs free-run agreement (tf_pred) |
| depthnpz, margins_norms | bonus | 3 | `depth.npz` per-token local-gain profiles + lam heatmaps; baseline hidden-state norms & drift vs layer |

## 4. What looked good vs uninformative (from the verification pass)

**Good — rich structure, worth reviewing first:**
- `ft_growth__L*` (growth per layer, thin dirs + band): clean exponential-type
  separation across layers; the 10–90% band shows real direction spread.
- `ft_growth_overlay__*` / `ft_stepratio_overlay__*`: layer ordering at a glance.
- `depth_heatmap__*`: layer×t log-amplitude maps are the densest, most
  readable summaries (0.74–0.76 ink coverage; clear depth-time structure).
- `state_heatmap__*` and `lp_heatmap__*`: the injection front is directly
  visible (white dashed marker), including the exact `r`-valued response at
  readout `inj+1` and the propagation wedge.
- `gamma_lag__*`: embedding-layer injections spike at lag 0 and decay —
  the absolute panel with the noise-floor line is the honest view.
- `win_growth__*` + `win_departure__*`: **w55 never reaches 2× growth
  within T=300 while every window ≥ 70 departs at t≈4** (all four prompt
  families, r=0.1, L27, pooled dirs) — the clearest single result of the tour.
- `agg_*`: pooling 5 prompts × 6–8 dirs gives tight bands; per-run medians
  (grey) stay inside the band — low prompt-to-prompt dispersion at w100.
- `tok_survival__*`, `tok_tfpred__*`: agreement survival curves decay
  smoothly with t and separate cleanly by radius.
- `hlast_pairdiv__*`: cross-prompt divergence and self-drift both grow
  monotonically; consecutive-state cosine (median ≈ 0.69) is *not* flat —
  the last-layer state rotates substantially each step.
- `ft_paircos__*`: pairwise cosine between perturbation directions at the
  last layer collapses from ±1 toward ≈ 0.19 median — directions decorrelate
  (one panel per figure; the temporal-cos panel sits near 1).
- `margins_norms`, `depthnpz_*`: informative depth context.

**Uninformative / negative-result figures (kept, but low value):**
- `ft_cos__*` (the a1 `cos` schema key): ≈ 1.0 everywhere (median 1.0,
  p1 = 0.987 even at r=2.0). It documents that amplification is almost purely
  amplitude-along-a-fixed-direction, but the figure itself is visually flat.
- `lam_local__L0_*` and any L0 panel: teacher-forced embeddings are identical
  by construction, so layer 0 is *exactly* zero at every radius — the module
  skips per-layer L0 figures with a logged warning; small-multiple panels
  for L0 render (correctly) empty.
- `lp_growth_overlay__*` (growth at readout L27): sparse — only shallow
  injection layers produce sustained last-layer growth, so most curves sit
  near 1; the heatmaps carry the same information better.
- Tiny radii (r ≤ 0.003) curves in λ/ratio figures: dominated by the fp32
  noise floor (gaps from NaN-guarding); harmless but don't over-read them.
- `lp_r02` and other single-injection-layer probe files with injection at
  deep layers: several all-zero panels, correctly labelled.

## 5. Schema assumptions (verified against local data)

1. **a1.npz**: `dh_norm[r,u,l,t]`, t index 0 = first generated step.
   Layer 0 (embedding) is exactly zero under teacher forcing (structural).
   Exact zeros at tiny radii = sub-fp32 response → treated as NaN (dropped
   from medians, valid-n annotated). `r_c` sentinel ≈ 1e30 ("never crosses")
   is masked. `pred_T == -1` = no flip. `dh_last[r,3,T,D]` = last-layer
   perturbation vectors; zero *coordinates* within a vector are legitimate
   and are NOT masked (only whole-row zero vectors are).
2. **state2.npz**: injection at step `s` (1-based label) first affects
   **time index s (0-based)** at every recorded layer ≥ `inject_layer`,
   where the response ≈ r at the injected layer itself. Values before
   injection are an fp32 noise floor (~1e-3 relative, *not* exact zero);
   drawn as a dashed "noise floor" line. `dm` is exactly 0 pre-injection,
   hence the flip tolerance `dm < -1e-3`.
3. **layer_probe.npz** (`dh[n_inj, n_dir, 28, T]`): response is zero at
   readout layers ≤ injection layer; first appears at readout `inj+1`, time
   index `pos`, with value exactly `r`. Injection at the final layer (27)
   gives an **all-zero slab** (it only perturbs logits; nothing recorded
   downstream) — panels are labelled "(no response)".
4. **free_run.npz**: `obs[r,u]` = first flip step (−1 = never). The older
   `fr_*.npz` files with object `gen_ids` are handled (only `obs`/`radii`
   read, `allow_pickle`).
5. **tf_pred npz**: `pred` = teacher-forced perturbed tokens, `ar` =
   free-run tokens, `ref` = reference; agreement = fraction matching `ref`.
6. **Pooling** requires a common radius value (matched to 6 decimals, middle
   of the intersection), uses the deepest common layer and the shortest T,
   and pools ragged direction counts nan-aware.
7. The big/scan window groups are the odd-valued windows **w55…w95 and
   w105…w130** (no w100 exists) — labels are parsed from directory names.

## 6. Reproducibility notes

- Full tour: `python make_all.py` → 300 PNGs in ≈ 41 min on this CPU
  (the heavy parts are the 28-layer probe heatmaps and the 72 MB w100_layers
  a1 file). Single family on one run: ~10–160 s.
- Everything is deterministic (jitter in token scatter uses a seeded
  `default_rng(0)`).
- `warnings.log` is the QA trail: any future figure with overlapping or
  clipped text will be logged there; the current full run has **zero** such
  entries (ghost tick labels excluded by design).
