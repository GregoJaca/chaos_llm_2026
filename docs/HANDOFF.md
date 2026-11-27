# HANDOFF — chaos_llm_2026

Updated 2026‑11‑28 00:00 CET. Read this first, then `notebook/update.md` (the report)
and `writing/explanation2.md` (the PI's own summary, annotated with `//` comments).

---

## 0. The project in one paragraph

Do LLMs amplify perturbations? Under **teacher forcing** the answer is no: the
response $\|\delta h_t\|/\|\delta h_1\|$ decays or plateaus. Under a **closed soft
loop** — feed the model its own expected embedding $e_t=\sum_v\mathrm{softmax}(z_t/T)_vE_v$
as the next input — the same perturbation grows by $10^1$–$10^4$ and then
**saturates** at $L_\infty=\log(a_{\rm closed}/a_{\rm replay})=1$–$11$ nats with a
rise time $\tau=5$–$1500$ steps. The network alone (replay control) is neutral or
contracting; the **readout→embedding feedback** is the amplifier. Depth-wise, the
*relative* perturbation $\|\delta h_\ell\|/\|h_\ell\|$ is roughly conserved (and
decays in the 14B models); the *absolute* growth is mostly residual-stream norm
growth. The sliding-window flush law
$t^*_\ell=W-P+p_{\rm last}+(\ell-1)(W-1)$ is exact and explains the "100-token
staircase" that appears in every $W=100$ run.

---

## 1. What is running (check this first)

**x11 (T4, 15 GB)** — `scripts/queue_v3b_x11.sh`, log `results/v3b_x11.log`.

Rounds: (A) $T=1$, 4 chunks × 32 directions = **128 directions**, loop+replay, all
8 small models; (B) $T=2$, 64 dirs, loop+replay; (C) $T=0.5$, 64 dirs, loop; (D)
prompts 2 and 3 for ds15/qwen34/phi, 64 dirs; (E) 3000-step runs with 32 dirs for
ds15/qwen34/phi; (F) **layer-0 runs** (32 dirs, `--save-layers 0,...`); (G) attention
for all layers and $W\in\{50,100,200,4096\}$. ~13 h.

**IMPORTANT — the first v3 attempt OOMed.** 256 directions × 1200 steps does *not*
fit on the T4 (the KV cache grows with rows × steps; it dies around $t=1000$). The
safe envelope measured here is **32 directions × 800 steps** for all models
(KV 0.8–4 GB). Do not raise the direction count and the step count together.
`results/v3_x11.failed.log` has the traceback. Spark has enough memory for
64 dirs × 1200 steps.

Then, automatically, two waiters:

* `scripts/queue_topk_x11.sh` (log `results/topk_x11.log`) — the top-k rotation and
  perturbation-alignment probe (`run_topk.py`) for ds15, qwen25, qwen34, qwen35, phi,
  pythia, randqwen.
* `scripts/queue_extra2_x11.sh` (log `results/extra2_x11.log`) — Qwen3.5-2B in
  **fp32** (its fp16 path dies in cuDNN: `CUDNN_STATUS_NOT_INITIALIZED`; do not use
  `--dtype float16` for Qwen3.5), plus A1 runs with all layers at $W=100$ and
  $W=4096$ for the staircase figure.

**Spark (GB10)** — `scripts/queue_v3_spark.sh`, log `results/v3_spark.log`.
$T=1$: 8 chunks × 64 directions = **512 directions** per 14B model, loop+replay;
then prompts 2/3 for qwen314 and ds14; $T=2$ and $T=0.5$; 3000-step runs; attention.
~25–40 h. 64 dirs × 1200 steps is measured to fit (≈12–20 min per run).
`scripts/queue_topk_spark.sh` waits for it and runs the probe on the 4 big models.

Nothing retries: a failure prints `FAILED` and the queue advances. Everything is
resume-safe (a finished `.npz` is skipped).

**Monitoring**

```
ssh x11   "grep -c saved ~/LLM/chaos_llm_2026/results/v3b_x11.log; tail -2 ~/LLM/chaos_llm_2026/results/v3b_x11.log"
ssh spark "grep -c saved /home/grego/LLM/chaos_llm_2026/results/v3_spark.log; tail -2 /home/grego/LLM/chaos_llm_2026/results/v3_spark.log"
```

---

## 2. Refreshing the report (one command)

```
powershell -ExecutionPolicy Bypass -File scripts\refresh_report.ps1
```

It syncs both hosts (`scripts/sync_pack.sh`, uncompressed tar — gzip was the
bottleneck), regenerates the loop figures, copies them into `notebook/update_figures`,
rebuilds `notebook/update.pdf` and commits. Use `-NoCommit` to skip the commit and
`-SkipSync` to re-render only.

Individual pieces:

```
python analysis_update/pack_loop_core.py          # F2, F2b + per-model pack_loop_<tag>.png
python analysis_update/pack_loop_layers.py        # F2c, F2d (layer-resolved)
python analysis_update/pack_loop_repetition.py    # F4 (period, cycle, tokens)
python analysis_update/pack_loop_sampling.py      # F5, F5b (soft vs CRN vs ind)
python analysis_update/pack_loop_tau_T.py         # F6 (L_inf/tau vs T, W)
python analysis_update/pack_depth_story.py        # B0, B0b, B5b, B7
python analysis_update/pack_window_staircase.py   # E5 (the 100-token staircase)
python analysis_update/pack_attention3.py         # attention per model, t/W, fits
python analysis_update/pack_topk.py               # rotation/alignment (needs the probe)
python analysis_update/run_all.py                 # everything
```

`analysis_update/looputil.py` holds the shared discovery logic: a **pair** is a loop
run and a replay run in the *same directory*; ratios are only ever formed inside a
pair; chunks of the same model/T are pooled (the reference row is unperturbed, so all
chunks share the orbit). Layers are read from `meta.json` (`layers` is not stored in
the npz) and matched by *layer number*, never by array index.

The PDF is served statically at <http://100.101.197.82:8898/update.pdf>; there is no
auto-rebuild.

---

## 3. What was fixed in this revision (and why it mattered)

* **The log-ratio band was wrong.** The old code smoothed the median but not the
  percentiles and divided by $a_1$ *before* smoothing, so the first point was not 1
  and the median could fall outside its own band. Now: no smoothing, the ratio is
  formed **per direction** and only then summarised, and $L(1)=0$ by construction.
* **The fit had a free offset.** $y_0$ was a free parameter; it is 0 by construction
  (both curves are normalised at $t=1$), so the fit is now $L_\infty(1-e^{-t/\tau})$
  with two parameters.
* **Missing log-ratio panels** (Qwen3-4B, Qwen3.5-2B) were a directory-matching bug;
  fixed by pairing inside one directory and matching layer numbers from `meta.json`.
* **fp32 OOM on the T4** for ≥3B at 3000 steps → those runs are fp16 (safe for the
  loop, whose responses are $O(10^2)$), except Qwen3.5 which needs fp32 for cuDNN.
* **Qwen3.5 fp16 crashes** in `causal_conv1d_fn` (`CUDNN_STATUS_NOT_INITIALIZED`).
* **The three depth metrics are now separated**: absolute $A_\ell$, state norm
  $S_\ell$, relative $R_\ell=A_\ell/S_\ell$, plus the linearised $\mathrm{cum}$ and
  $g$. The identity $R_\ell/R_1=(A_\ell/A_1)/(S_\ell/S_1)$ is the key to reading them.

---

## 4. Findings (updated numbers, section F of the report)

| model | directions | steps | $L_\infty$ | $\tau$ | $R^2$ | end $\times$ |
|---|---|---|---|---|---|---|
| Mamba-130m | 8 | 3000 | 11.3 | 32 | 0.41 | 12 400 |
| DeepSeek-1.5B | 16 | 3000 | 7.8 | 56 | 0.92 | 2 400 |
| Qwen3-14B | 8 | 1600 | 6.5 | 228 | 0.77 | 600 |
| Phi-4 | 8 | 1600 | 6.7 | 255 | 0.77 | 530 |
| DeepSeek-14B | 8 | 1600 | 6.0 | 209 | 0.77 | 220 |
| Phi-4-mini | 8 | 3000 | 5.3 | 60 | 0.69 | 173 |
| Pythia-410m | 8 | 3000 | 6.4 | 357 | 0.48 | 85 |
| Qwen3-4B | 16 | 3000 | 4.8 | 17 | 0.46 | 76 |
| Qwen3.5-9B | 40 | 1600 | 2.4 | 2 | 0.02 | 35 |
| randQwen | 8 | 3000 | 3.8 | 1480 | 1.00 | 13 |
| Qwen2.5-3B | 16 | 3000 | 1.3 | 5 | 0.06 | 15 |

* **Sampling is not the cause.** Soft (mean) grows $10^1$–$10^4$; CRN (identical hard
  token for every row) stays at 0.03–1.3; independent sampling at 0.8–6.
* **The gain is depth-independent** (F2c): every saved layer shows the same $L(t)$.
  The loop acts on the *token*, not on a layer.
* **What the loop generates at saturation** (F4, decoded with each tokenizer):
  Qwen2.5-3B repeats `" propulsion"` forever (period 1), randQwen `"orse"`,
  Mamba `"\nA\n"` (period 3), Phi-4 `"acja uw"` and Qwen3.5-9B `":\n\n"` (period 2);
  Phi-4-mini repeats a 40-token paragraph, DeepSeek-1.5B re-writes the prompt,
  Qwen3-4B keeps writing **coherent** text, DeepSeek-14B/Qwen3-14B write
  semi-coherent markdown. So saturation is not caused by literal repetition.
* **Relative depth response** ($R_L/R_1$ at $t=50$): 43 (Phi-4-mini), 4.0 (Qwen2.5),
  2.3 (DeepSeek-1.5B), 1.5 (Qwen3-4B, Qwen3.5-2B), 1.55 (randQwen), 1.17 (Mamba),
  0.61 (Pythia), **0.33–0.43 for the three 14B models**. Absolute growth
  (8–1880×) is the residual-stream norm growth (2.5–266×).
* **The 100-token staircase is the window law**, not a new phenomenon (E5).

Rejected / corrected claims (do not resurrect):
* "any soft feedback gives the same gain" — true for DeepSeek-1.5B, false in general
  ($T=2$ kills the gain for Phi-4-mini, Mamba, Qwen3-4B, randQwen).
* "the gate-law identity breaks down for the random control" — false; slope −1.01
  for $m/T>3$; it is the *amplitude* that does not match.
* "sparse directions saturate" — false; same exponent, constant offset.
* "the response grows with depth in every trained model" — true for the *absolute*
  response, misleading for significance; use $R_\ell$.
* "one-shot ratchet", "self-regulation" as a mechanism — replaced by "the loop
  switches to a different regime and stays there"; the saturation is a property of
  the attracting orbit.

---

## 5. Next steps, in priority order

1. **Re-run the report after the campaigns** (`refresh_report.ps1`) — the loop figures
   will move from 8–16 to 128 directions (x11) and 512 (Spark), which tightens every
   percentile band and the $L_\infty$/$\tau$ table.
2. **Insert the top-k figure** once `pack_topk.png` exists: the report currently
   discusses k=1,3,10 in §C but the figure is not yet embedded (add
   `![Fig. C3b](update_figures/pack_topk.png)` with the caption from
   `analysis_update/pack_topk.py`).
3. **Multi-prompt replication** ($L_\infty$ per prompt) — runs are in the x11 round D
   and the Spark round B. Average $L_\infty$ over prompts to get a real error bar on
   the headline number.
4. **Exact $\Sigma_p$** (all-vocab instead of top-64 mass) — never run, ~1 h.
5. **Mamba rotation control** — `run_topk.py` handles Mamba's block interface, but it
   was not queued (small model, only on x11); add it to the topk list when time allows.
6. **Dense 14B $r_*$** for Qwen3-14B and DeepSeek-14B — still missing (Phi-4 is done,
   $r_*=1.21$–$1.28$).
7. **A second prompt family for the layer-0 measurement** — the layer-0 runs answer
   "does the perturbation re-enter at the input in the loop?" (yes) versus teacher
   forcing (exactly zero); currently 3 models.

---

## 6. Gotchas

- **KV cache ≠ weights.** Direction count × steps is the memory knob. 32×800 is safe
  on the T4 for every model; 256×1200 is not.
- **Qwen3.5-2B must run in fp32** on x11 (`venv311`); fp16 hits
  `CUDNN_STATUS_NOT_INITIALIZED` in its conv path.
- **`layers` lives in `meta.json`**, not in the npz; different campaigns saved
  different layer sets (some include layer 0, which is identically zero).
- **Direction pairing**: the same `--seed` and `--dirs` in loop and replay gives the
  same probe directions; ratios must be formed per direction before averaging.
- **PowerShell quoting**: use a temp `.py` file or `bash -s` with `\r` stripped;
  `python -c "..."` with nested quotes gets mangled.
- **CRLF**: `sed -i 's/\r$//'` after every `scp` of a `.sh`.
- **Killing a queue**: kill the bash parent *and* the `run_softloop` child, or the
  child keeps the GPU.
- **Do not kill other people's jobs on Spark** (~54 GB is theirs).
- The old `pack_loop_loglog_models.py` is superseded by `pack_loop_core.py`; the old
  figure files are kept in Appendix N3 as requested.

---

## 7. Commit hygiene

Small, frequent commits; never delete a figure that was in a previous PDF without
being told — supersede it in Appendix N (as with N3).
