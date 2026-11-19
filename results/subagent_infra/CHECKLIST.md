# CHECKLIST — running the protocol on a new model

Order matters: never launch a queue before steps 1–5 pass for that model.
All commands assume the repo root (`~/LLM/chaos_llm_2026` on the GPU host) and
`PYTHONPATH=src`.

## 0. Install the audited files (one-time)

```bash
git add -A && git commit -m "pre-multimodel snapshot"   # so anything is revertible
cp results/subagent_infra/protocol_general.py results/subagent_infra/model_plan.py src/chaos_llm/
cp results/subagent_infra/patched/*.py                  src/chaos_llm/   # replaces the 8 scripts
# scripts NOT replaced (run_state.py, run_state_mid.py, run_tf_predict.py, ...)
# still import chaos_llm.protocol and keep working exactly as before.
```

## 1. CPU self test (any machine, ~1 min, no GPU, no weights)

```bash
PYTHONPATH=src python results/subagent_infra/selftest_cpu.py            # all 4 defaults
PYTHONPATH=src python results/subagent_infra/selftest_cpu.py --offline --models \
    Qwen/Qwen3-4B-Instruct-2507 microsoft/Phi-4-mini-instruct
```

Read the printed **PLAN** per model — it is exactly what the runs will use:
dtype (auto rule: fp32 ≤2.5B params else bf16), `layer_path_candidates` (first entry
is what `model_layers()` will resolve), `window_key` + support notes, default
measurement layers, vocab size. **Exit code must be 0.**
Expected: DeepSeek→float32, Qwen3-4B→bfloat16, Phi-4-mini→bfloat16,
Qwen3.5-2B→float32 (vision/mtp weights excluded), window unsupported for Qwen3.5.

## 2. Environments (two, do not merge)

| env | transformers | models | why |
|---|---|---|---|
| **env4** | `4.51.3` (≥4.51 required) | DeepSeek (repro), Qwen3-4B-Instruct-2507, Phi-4-mini | Qwen3 needs 4.51; Phi-4-mini native needs 4.49; patched run_a1_long needs 4.50 (`logits_to_keep`) |
| **env5** | latest **5.x** (must expose `Qwen3_5ForCausalLM`; card says install from git) | Qwen3.5-2B only | Qwen3.5 architecture exists only there |

Verify imports before downloading weights:

```bash
python -c "from transformers import Qwen3ForCausalLM, Phi3ForCausalLM; print('env4 ok')"
python -c "from transformers import Qwen3_5ForCausalLM; print('env5 ok')"
```

Optional (Qwen3.5 speed/memory): `pip install causal-conv1d fla` — on T4 there is no
build, transformers silently uses the slower PyTorch fallback; budget extra activation
memory and time.

Keep the OLD host environment untouched for DeepSeek (REPORT §24: masking internals
changed across versions; upgrading in place risks drifting the published flush-law
numbers).

## 3. Download the models (GPU host)

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
huggingface-cli download microsoft/Phi-4-mini-instruct
huggingface-cli download Qwen/Qwen3.5-2B
```

## 4. Smoke test per model (GPU, 2–5 min; REQUIRED before any queue)

```bash
# env4:
PYTHONPATH=src python results/subagent_infra/smoke_gpu.py \
  --model Qwen/Qwen3-4B-Instruct-2507 --window 100 --prompt @prompts.txt \
  --out results/smoke/qwen3_4b
PYTHONPATH=src python results/subagent_infra/smoke_gpu.py \
  --model microsoft/Phi-4-mini-instruct --window 100 --prompt @prompts.txt \
  --radii 1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1 --out results/smoke/phi4mini
# env5 (window NOT applicable -> 0):
PYTHONPATH=src python results/subagent_infra/smoke_gpu.py \
  --model Qwen/Qwen3.5-2B --window 0 --prompt @prompts.txt \
  --out results/smoke/qwen35_2b
```

Acceptance gates, in printed order:
1. `[1] loaded` — check dtype (bf16 for 4B/3.8B, fp32 for Qwen3.5-2B), num_layers,
   vocab, and that `params` looks right (≈4.0B / 3.8B / ≈2.2B text).
2. `[2] forward OK` — logits finite, hidden_states == num_layers+1.
3. `[3] r=0 duplicate rows identical: True` and full generation length — if False,
   STOP (see HANDOFF: never mix precisions; do not "fix" by re-running until it passes).
4. `[4] window check` —
   - Qwen3-4B-2507, Phi-4-mini: expect `effective=True`. If False on Qwen3-4B, the
     installed transformers builds masks differently → run window studies with
     `run_a1 --hard-window` (dtype-correct now) and note it in meta.
   - Qwen3.5: skipped by design (no window mechanism).
5. `[5] r-linearity` — pick the radii ladder from the printed **linear band**.
   - fp32 models: expect a band from the small radii up.
   - bf16 models: expect **zero response below r≈1e-2** (bf16 rounding floor,
     REPORT Part C). If all radii give zero, rerun the smoke with
     `--radii 1e-2,3e-2,1e-1,3e-1,1,3`.
6. Phi-4-mini extra (once): repeat step 4–3 with
   `load_model(..., trust_remote_code=True)` and confirm identical greedy ids to the
   native path before trusting native for paper numbers.

Also do one **DeepSeek repro** on env4 right after upgrading anything:
re-run one stored A1 configuration (same prompt/window/radii/seed/layers) and compare
`margins` with the old `a1.npz` (`np.array_equal` ideally). Identical → env4 is safe
for cross-model tables; not identical → keep model-specific claims within
environments and record the drift.

## 5. Choose radii and layers

- Radii: 12-point geometric ladder covering the smoke's linear band — bf16 models
  typically land around `1e-2` up to `0.3–1`; fp32 models usually reproduce the
  legacy `1e-6…1e-2` shape. Use the smoke's printed
  `SUGGESTED run_a1 --radii:` line verbatim; extend one decade above the band if you
  want flip-saturation coverage.
- Layers: leave `--layers` empty (auto = `[0, n/4, n/2, 3n/4, n-1]`, the same
  convention as DeepSeek's `0,7,14,21,27`).
- Direction count: 20 for the main A1 (as before), ≥4 always (k=1 is refused).

## 6. Full runs, per model (in this order; one job at a time on the T4)

```bash
M=Qwen/Qwen3-4B-Instruct-2507        # or microsoft/Phi-4-mini-instruct (env4)
R=<chosen radii>

# a) main A1 (teacher-forced, margins, thresholds)
PYTHONPATH=src python -m chaos_llm.run_a1 --model $M --window 100 \
  --max-new-tokens 300 --num-directions 20 --radii $R --out results/<modeltag>/a1_w100

# b) free-run validation of predicted flips
PYTHONPATH=src python -m chaos_llm.run_thresholds --model $M \
  --run results/<modeltag>/a1_w100 --radii <3 radii from $R> --out results/<modeltag>/thr_w100

# c) state kicks (embedding + mid-layer injection)
PYTHONPATH=src python -m chaos_llm.run_state2 --model $M --run results/<modeltag>/a1_w100 \
  --inject-layer 0 --out results/<modeltag>/state2_emb
PYTHONPATH=src python -m chaos_llm.run_state2 --model $M --run results/<modeltag>/a1_w100 \
  --inject-layer $((NL/2)) --out results/<modeltag>/state2_mid   # NL from model_info

# d) autoregressive kick validation (r=0 self-check first!)
PYTHONPATH=src python -m chaos_llm.run_state_freerun --model $M \
  --run results/<modeltag>/a1_w100 --check-r0 --out results/<modeltag>/fr_w100

# e) layer-resolved amplification (all blocks, all hidden states)
PYTHONPATH=src python -m chaos_llm.run_layer_probe --model $M \
  --run results/<modeltag>/a1_w100 --out results/<modeltag>/probe_w100

# f) long trajectories (>600 tokens: collect + extend)
PYTHONPATH=src python -m chaos_llm.collect_trajectories --model $M \
  --prompts prompts_long2.txt --window 4096 --max-new-tokens 4000 --out results/<modeltag>/traj4k
PYTHONPATH=src python -m chaos_llm.extend_trajectories --model $M \
  --src results/<modeltag>/traj4k/<name>.npz --out results/<modeltag>/traj4k/<name>_ext.npz

# g) long teacher-forced response (uses logits_to_keep; needs transformers>=4.50)
PYTHONPATH=src python -m chaos_llm.run_a1_long --model $M \
  --prompt @prompts_long2.txt --fixed-tokens-from results/<modeltag>/traj4k/<name>.npz \
  --window 100 --max-new-tokens 4000 --num-directions 4 --radii 0.03,0.1,0.3 \
  --out results/<modeltag>/a1_long/<name>
```

For **Qwen3.5-2B** use the same commands in **env5**, with:
- `--window` present but expected to log `set_window ... NO-OP` — fine; **skip all
  window-sweep studies** for this model (the flush law is not defined for a hybrid
  linear-attention stack; `--hard-window` is NOT equivalent either).
- `--batch-dirs 2` (248k-vocab logits in fp32 are ~400 MB per row at 400 tokens).
- inject-layer studies: allowed mechanically, but remember blocks alternate
  linear/full (only 6 attention layers); label them by block index, not "attention
  depth".
- watch for harmless `unexpected keys` warnings (vision tower + MTP weights).

## 7. Per-model ops limits (T4, 15 GB, one job at a time)

| model | weights | run_a1 full-logits cap | batch-dirs @300 tok | notes |
|---|---|---|---|---|
| DeepSeek fp32 | 7.1 GB | ≤500–600 tok (use run_a1_long beyond) | 1–4 (as before) | unchanged protocol |
| Qwen3-4B bf16 | 8.0 GB | ≤600 tok (bf16 halves logits) | ≤4 | bf16 on T4 has no tensor cores → slower |
| Phi-4-mini bf16 | 7.6 GB | ≤600 tok | ≤4 | keep prompt+gen < 4096 (longrope re-prefill); sdpa fine on T4 (the model card's "eager on ≤V100" note concerns flash-attention defaults, not sdpa) |
| Qwen3.5-2B fp32 | ~8.8 GB | ≤500 tok | ≤2 | GDN PyTorch fallback is memory hungry; check nvidia-smi after the smoke |

## 8. After each run

- `meta.json` now records `model_class`, `params`, `dtype`, `num_layers`,
  `vocab_size`, `hidden_size` — glance at it before analyzing; a wrong dtype in meta
  means someone overrode it.
- free-run outputs: confirm `summary.json` `prompt_len` matches the a1 run
  (the patched scripts re-tokenize the prompt and compare).
- keep raw-completion prompts off chat formats for Phi-4-mini (it will emit
  `<|end|>` early on instruction-like text; the protocol continues past it, which is
  fine, but a mid-response EOS makes the trajectory less interesting).

## 9. If something fails

| symptom | meaning | action |
|---|---|---|
| `r=0 duplicate rows differ` | nondeterminism / precision mixing | do not run; report model+dtype+env (HANDOFF rule: bf16 r=0 previously diverged) |
| `generate returned N != max tokens` | early stop leaked through a version change | inspect `model.generation_config.eos_token_id`; keep `eos_token_id=None` path |
| `set_window ... NO-OP` | model has no window mechanism (Qwen3.5) | expected; skip window studies |
| `could not locate the decoder-layer list` | wrapper layout changed | read the printed attribute layout; add the path to `model_plan._ARCH_LAYER_PATHS` / candidates |
| `token ids beyond this model's vocab` | cross-model reuse of a1 npz | regenerate the a1 run on this model |
| smoke shows zero response at all radii | below the bf16 floor | rerun smoke with `--radii 1e-2,3e-2,1e-1,3e-1,1,3` |
| OOM at 300 tokens | batch-dirs too high / vocab too big | halve `--batch-dirs`, or use `run_a1_long` |
