# HANDOFF — chaos_llm_2026

Written 2026‑11‑26 ~23:05 CET, at the moment two GPU campaigns were launched.
Read this first, then `notebook/update.md` (the report the PI is reading).

---

## 0. One-paragraph summary of the project

We study whether LLMs amplify perturbations. Under **teacher forcing** (hard
tokens from a fixed autoregressive stream) they do **not**: the response
$\|\delta h_t\|/\|\delta h_1\|$ decays or plateaus over 300–4000 tokens, across
9 architectures. Under a **closed soft loop** — feed the model its own expected
embedding $e_t=\sum_v \mathrm{softmax}(z_t/T)_v E_v$ instead of a token — the
same perturbation grows by $10^2$–$10^4$ and then **saturates** at a finite
amplification $L_\infty=\log(a_{\text{closed}}/a_{\text{replay}})$ of 1–10 nats.
The network alone is neutral/contracting; the **readout→embedding feedback** is
what makes the loop unstable. The companion window result is an exact mask
identity: the response leaves layer $\ell$ at generated step
$t^*_\ell = W-P+p_{\text{last}}+(\ell-1)(W-1)$.

---

## 1. What is running right now (check this first)

Two long campaigns were started at **2026‑11‑26 22:50 CET**. They are
**resume-safe** (each step is skipped when its `.npz` exists), **never retry**
(a failure prints `FAILED` and the queue advances), and they only ever kill
their own processes.

| host | script | log | what |
|---|---|---|---|
| x11 (T4, 15 GB) | `scripts/queue_long_x11.sh` | `results/longloop_x11.log` | 8 small models, **3000 steps**, 4 saved layers, 8 dirs |
| Spark (GB10, ~121 GB unified) | `scripts/queue_long_spark.sh` | `results/longloop_spark.log` | Qwen3-14B, DeepSeek-14B, Qwen3.5-9B, Phi-4, fp16, **1600 steps** (`≈3.4 s/step`) |

Round order (both hosts) — designed so that partial completion still covers the
most important comparison:

| round | x11 | Spark | why |
|---|---|---|---|
| A | $T=1$ loop + replay, every model | same | the closed-vs-replay log-log figures (top priority) |
| B | $T=2$ loop + replay | $T=2$ loop | temperature dependence |
| C | CRN + independent sampling, $T=1$ | — | sampling-noise control |
| D | teacher-forced, 1000 tokens | 800 tokens | long-horizon control |
| E | $T=0.5$ loop + replay | $T=0.5$ loop | low-temperature limit |

Runtime: x11 ≈ 15 min per 3000-step 1.5B run (0.28 s/step), 2–3× that for the
3–4B models, so the whole x11 queue is ~30 h. Spark is ~2 h per 1600-step run,
~30 h for the queue. Both were given 36–48 h of headroom.

**How to check progress**

```
ssh x11   "grep '^===' ~/LLM/chaos_llm_2026/results/longloop_x11.log | tail -3"
ssh x11   "tail -3 ~/LLM/chaos_llm_2026/results/longloop_x11.log"
ssh spark "grep '^===' /home/grego/LLM/chaos_llm_2026/results/longloop_spark.log | tail -3"
```

`grep 'saved'` lists finished runs. If a queue died, just re-run the same script
with nohup; it picks up where it stopped.

**Do not kill other people's jobs on Spark.** The machine is shared
(`free -g` showed 54 GB used by others when we started). Our models are fp16 and
need ~30 GB.

---

## 2. When the campaigns finish: the exact sequence

```
# 1. pull results from both clusters (merges into results/, same layout)
powershell -ExecutionPolicy Bypass -File analysis_update\sync_from_cluster.ps1

# 2. regenerate every figure (writes update_figures/)
python analysis_update\run_all.py

# 3. rebuild the pdf (copy is automatic inside run_all.py)
cd notebook
python .preview\build.py

# 4. check and commit
git add -A; git commit -m "Long-campaign results"
```

`pack_loop_loglog_models.py` always picks the **longest** orbit available per
model, from `results/longloop/` or the older `results/softloop/`, so figures
improve automatically as data lands. It also writes
`analysis_update/pack_loop_loglog_models.json` (fields: `L_end`, `L_inf`, `tau`,
`sat_r2`, `lambda_fb`, `growth_closed`, `growth_replay`), which
`pack_loop_saturation.py` turns into the summary figure F0m.

The PDF is served at <http://100.101.197.82:8898/update.pdf> (static server;
there is no auto-rebuild — rebuild manually with `build.py`).

---

## 3. Where the data and the code live

Repo layout (identical on x11, Spark and the Windows checkout; `~` is
`/u0/grego` on x11, `/home/grego` on Spark):

```
src/chaos_llm/
  run_softloop.py      loop / replay / CRN / ind  <- the central experiment
  run_a1.py            teacher-forced prompt perturbation (A1)
  run_attention_frac.py attention mass on prompt tokens (needs the 4-D mask)
  run_depth2.py        Jacobian-vector products along depth
  run_angle.py         response-vs-state angle
  run_nonnorm.py       block non-normality / rotation
  run_window_tokens.py sliding-window token agreement
  protocol.py          load_model, get_layers, band_mask4d (mask = the window law)

analysis_update/       one script per figure, all read .npz, none fabricate data
  run_all.py           runs every pack_*.py then copies to notebook/update_figures
  cache.py             path table (SOFT, BIG, COL), soft_runs()
  pack_loop_*.py       the looping figures F0c-F0m, F2, F13-F18
  pack_window_*.py     the window figures E1-E3
  pack_gate*.py        gate law (log-log, lin-log, per model)
  pack_depth*.py       depth figures B1-B6
  pack_angle*/rotation*/relative*/time_*/layer_gain  geometry and time
  sync_from_cluster.ps1  pulls results/longloop + results/longtf

notebook/
  update.md            THE report (edit this, then build)
  .preview/build.py    md -> html -> pdf (DOCS list at the top)
  update_figures/      copies of update_figures/ used by the md

results/
  longloop/<tag>/T<T>/{loop,replay,crn,ind}/softloop.npz   new 3000/1600-step runs
  longtf/<tag>/a1.npz                                      new long teacher-forced
  softloop/<tag>/T<T>/...                                  old 600-step campaign
  update/...                                               fp32, dense, angle, attention
  softloop_sample/<tag>/{crn,ind}_T1.0                     sampled feedback
```

Tags: `ds15` DeepSeek-R1-Distill-Qwen-1.5B, `qwen25` Qwen2.5-3B-Instruct,
`qwen34` Qwen3-4B-Instruct-2507, `qwen35` Qwen3.5-2B, `qwen35_9b` Qwen3.5-9B,
`phi`/`phi4` Phi-4-mini / Phi-4, `ds14` DeepSeek-R1-Distill-Qwen-14B,
`qwen314` Qwen3-14B, `mamba` mamba-130m, `pythia` pythia-410m,
`randqwen` random-weight Qwen.

Python environments (this matters — not every model loads in every venv):
`python3` (x11 system) → ds15, mamba, randqwen; `~/LLM/venv_tf` → qwen25,
qwen34, phi, pythia; `~/LLM/venv311` → qwen35 (needs newer transformers);
`/home/grego/LLM/venv5` on Spark → everything.

---

## 4. The findings, in the order they are worth writing up

1. **No amplification under teacher forcing / replay.** 9 architectures, 300–4000
   tokens, finite-time exponent $\le 0$. Figures D1, D1b, D5.
2. **The closed soft loop amplifies and then saturates.** $L_\infty$ = 1.0 (Qwen2.5)
   to 9.6 nats (Mamba); $\tau$ = 3–670 steps; the saturating fit is good
   ($R^2>0.6$) for 6 of 7 models. This is the headline. Figures F0c–F0m.
   The growth is a **staircase** of gate events, not an exponential: FSLE,
   finite-time exponent, increment autocorrelation, power spectrum (F14–F18).
3. **The instability is the readout, not the network.** Replay (frozen soft
   teacher orbit) and CRN sampling are flat; only mean feedback and independent
   sampling destabilise (F1, F2, F9). Mamba (no attention) amplifies as much as
   any transformer; the random transformer amplifies $\times4$ instead of
   $\times10^3$ (F0j, F0l, F0m).
4. **What the replay teacher is:** the closed soft loop of the *unperturbed*
   row, recorded step by step — not greedy autoregressive generation. At
   $T\to0$ it reduces to greedy (the script checks this). Consequence: a replay
   run must use the loop file of the same $T$. §F.0b of the report.
5. **Exact window clock.** $t^*_1=p_{\text{last}}+W-P$,
   $t^*_\ell=t^*_1+(\ell-1)(W-1)$, residual 0 over 106 windows
   ($p_{\text{last}}=43$, $P=65$; 55/64 for the other two models). E1, E1b.
   Attention mass drops at $t\approx W$ (last *prompt* token leaves the window),
   the response flush is earlier because only *perturbed* positions matter. E4.
6. **Window token agreement** starts to decay exactly at $t_{\text{onset}}=W-P$
   and then follows a stretched exponential ($\beta\approx1.7$–4). Fit must
   exclude the plateau. E3.
7. **Gate law.** The slope of $\sigma_{\max}(\Sigma_p)$ vs $m_t/T$ is the
   softmax identity ($-1$) for every trained model and for randQwen
   ($-1.01$ for $m/T>3$, $R^2=0.99$); the *amplitude* is not universal — the
   data sit below $G/4\cosh^2$ for most models. Do not present it as a new law.
   F4, F4b, F5, F5b.
8. **Depth.** Block 0 dominates (2–85% of the log gain depending on model); after
   block 0 the local gain is nearly flat with a small late rise (1.00–1.11).
   The response grows with depth by 3.5–14× (last/middle) in every trained model
   and by 1.2–1.7× in Pythia/randQwen. B1, B4b–B6.
9. **Geometry.** Median response-vs-state angle is 87–91° for LLMs *and*
   controls → not discriminative. The top-singular-vector rotation between
   consecutive blocks is 78–87° in trained models vs 89° for randQwen → real but
   modest. C1–C3.
10. **Fragility.** Sparse and dense directions have the **same** exponent
    ($\beta\approx1$) over $r=0.03$–0.3; dense is a constant 3–10× higher, i.e. a
    vertical shift in log–log. Sparse $r_*$ depends on the probe radius; the
    dense $r_*$ is the comparable number (DeepSeek 0.37, Qwen3.5 0.36,
    Qwen2.5 0.58, Phi-4 1.2). C4, C5.

Rejected claims (do not resurrect): five-regime taxonomy, "perpendicular
response" as a discovery, "sparse directions saturate", the spike+two-exponential
depth fit, the log-logistic $r_*$ law (KS rejects it), "entropy rises with $T$"
(`ent_ref` is only at $T=1$), any edge-of-chaos claim.

---

## 5. Obvious next steps, in priority order

1. **Re-run the loop analysis on the long data.** `pack_loop_loglog_models.py`
   handles it, but the report text (F0c–F0m, F13–F18) needs the numbers updated
   — search for `600 steps` in `notebook/update.md`.
2. **Fill the missing replays.** `Qwen3.5-2B` had no replay orbit, and `ds15v2`
   is a second seed; both now exist in the x11 queue at $T=1,2,0.5$. Also add
   `Qwen3.5-9B`/14B replays once Spark finishes.
3. **Multi-prompt replication of the loop.** Everything here is one prompt and
   8 directions per orbit. The single most valuable new run is loop+replay for
   3–5 prompts per model, so $L_\infty$ gets an error bar across prompts. The
   code already supports `--prompt`; the queue does not loop over prompts yet.
4. **Seed replication** at $T=1$ for 2 models to get a seed-to-seed error bar.
5. **Exact $\Sigma_p$** (all-vocab instead of top-64 mass) to confirm the gate
   $\sigma_{\max}$; ~1 h on x11, never run.
6. **Pythia/Mamba rotation controls**: Pythia needs `gpt_neox.layers` in
   `protocol.get_layers` (the entry was added; `run_nonnorm.py` still uses
   `model.model.rotary_emb` and fails — patch that or use `run_nonnorm_big.py`).
   Mamba rotation was never run.
7. **Dense 14B $r_*$** for Qwen3-14B and DeepSeek-14B: blocked by colleagues'
   memory; `queue_spark_wait.sh` retries when ≥66 GB are free. Phi-4 succeeded
   ($r_*=1.21$–$1.28$).
8. **A "loop horizon" metric**: the step where $a_t$ first exceeds
   $0.25\,\mathrm{median}\|h\|$ is noisy; the saturating $(\tau, L_\infty)$ pair
   is a better summary and is already extracted.

---

## 6. Gotchas that cost me time

- **fp16 floor.** Qwen3-4B and Phi-4-mini absolute responses at small radii are
  floor-dominated in fp16; only discrete flips and *shapes* are meaningful.
  fp32 for those two is still not done (a 4B fp32 will not fit on the T4).
- **Sparse ≠ dense scale.** Never mix absolute response numbers from the sparse
  simplex ensemble with the dense Gaussian one; they differ by a constant factor
  in log–log.
- **mathtext has no `\ge`** (use `\geq`) and `\frac12` needs braces.
- **PowerShell quoting.** `python -c "..."` with nested quotes mangles; write a
  temp `.py` or pipe a here-string to `ssh host "bash -s"` — and strip `\r`
  (`-replace "\`r",""`), otherwise remote paths get a literal CR and redirects
  create files with `\r` in the name.
- **CRLF shell scripts.** Always `sed -i 's/\r$//'` after `scp` of a `.sh`.
- **Don't `kill` the queue parent only** — the python child keeps running and
  keeps the GPU; kill both patterns (`queue_long_x11.sh` and `run_softloop`).
- **`save-layers` accepts a comma list**; saving 4 layers instead of 1 costs
  almost nothing and makes the layer-resolved loop figure possible.
- **Replay must be paired with the loop file of the same $T$** (`--replay-file`),
  otherwise the teacher orbit and the temperature disagree.
- The random control's margins are 0.11 nats vs 1.5–6 for trained models; that
  is why its gates are "open" at $T=0.13$ and why it looks different in F3b.

---

## 7. Commit hygiene

The PI asked for many small commits. Recent history is a good model:
data-fix, figures, report, then the next analysis. Never delete a figure that
was in a previous PDF without being told; supersede it in Appendix N instead.
