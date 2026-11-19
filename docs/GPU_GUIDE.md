# GPU operations manual (T4, 15 GB)

Everything here is from measured failures. Read before launching anything.

## 1. Hardware and budget

- One Tesla T4, **15 360 MiB (14.56 GiB)**, shared with other users. Always check first:
  `ssh x11 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`
- DeepSeek-1.5B in FP32 = **7.1 GB** of weights. That leaves ~7.4 GB for activations.
- If another process holds >1 GB, **do not launch**; wait.

## 2. Golden rules

1. **One job at a time.** Two jobs → 14.4 GB ceiling and a cryptic OOM in the MLP layer.
2. **`torch.inference_mode()` around every forward.** Without it autograd retains the 28-layer
   graph and 800+ tokens OOM even at batch 1. All scripts now do this; keep it.
3. **Batch = 1 for `run_a1`.** Batch 2 fails sporadically at 300–400 tokens (allocator
   fragmentation, allocation of 20–800 MB fails with ~15 GB "used").
4. **For >600 tokens use `run_a1_long`**, which calls `model.model` + `lm_head` only at the
   generated positions and never materialises `[L,V]` logits. 4000 tokens works (13 s for 2
   directions).
5. **Never pass `--num-directions 1`** to `run_a1`: the simplex basis is degenerate and produces a
   zero perturbation (NaN downstream). Minimum 2, use 4–20.
6. **Detach everything:** `nohup setsid bash script.sh > log 2>&1 < /dev/null & disown`.
   Never rely on the ssh session staying open.
7. **Resume-safe scripts:** every queue checks `[ -f out/a1.npz ]` and skips. Re-running is free.
8. **Check the log after 30 s**; a Python traceback is printed immediately, the process then moves
   on to the next job (the queue does not stop).

## 3. Measured memory and speed

| task | tokens | batch | dirs | memory | time |
|---|---|---|---|---|---|
| `run_a1` (fp32, full logits) | 300 | 1 | 12 | ~9 GB | ~40 s |
| `run_a1` | 500 | 1 | 8 | ~10 GB | ~70 s |
| `run_a1` | 800 | 1 | 8 | **OOM** | – |
| `run_a1_long` (lm_head only) | 2000 | 1 | 2 | ~9 GB | 6 s |
| `run_a1_long` | 4000 | 1 | 2 | ~10 GB | 13 s |
| `collect_trajectories` (KV cache) | 4000 | 1 | – | ~8 GB | ~2–4 min |
| `run_state2` | 300 | 2 | 16 | ~10 GB | ~4 min |
| `run_layer_probe` | 300 | 1 | 6 | ~9 GB | ~1 min |
| `run_thresholds` (generation) | 300 | 6 rows | – | ~9 GB | ~40 s |

Practical limits: teacher-forced **4000 tokens OK**; generation **4000+ OK**; `run_a1` (full
logits) **≤600 tokens**.

## 4. Failure log (do not repeat)

| symptom | cause | fix |
|---|---|---|
| OOM "tried to allocate 20–800 MB", 14.4 GB used | missing `inference_mode` (graph kept) | wrap forwards |
| OOM at 300 tokens with batch 2 | allocator fragmentation | batch 1 |
| `KeyError: 'prompt_ids is not a file'` | source npz made by `extract_last.py` (only h_last, gen_ids) | use the full trajectory npz |
| `IndexError: list index out of range` in `run_a1` | prompt file has a trailing empty line and the job asked for it | `sed -n "Np" file` and check non-empty |
| NaN response | `--num-directions 1` | use ≥2 |
| scp "No such file" | log/npz created after the copy attempt | sleep, then copy |
| OOM while another job runs | concurrent jobs | one at a time |
| 800-token `run_a1` fails | full `[L,V]` logits + hidden states | use `run_a1_long` |

## 5. Recipes

**Long teacher-forced response (up to 4000 tokens):**
```bash
PYTHONPATH=src python3 -m chaos_llm.run_a1_long \
  --prompt @prompts_long2.txt --window 100 --max-new-tokens 4000 \
  --fixed-tokens-from results/long_traj_4k/history_of_computing.npz \
  --num-directions 4 --radii 0.03,0.1,0.3 --layers 7,14,21,27 \
  --out results/big4/a1_4k/foo
```

**Long trajectory:**
```bash
python3 -m chaos_llm.collect_trajectories --prompts p.txt --window 4096 \
  --max-new-tokens 4000 --layers 7,14,21,27 --out results/big4/traj4k_tmp
```

**Extend an existing trajectory** (needs `prompt_ids`, `gen_ids`, `h_last`):
```bash
python3 -m chaos_llm.extend_trajectories --src results/long_traj_2k/ocean_currents.npz \
  --out results/big4/traj4k/ocean.npz --extra 2000 --window 4096
```

## 6. Monitoring

```bash
ssh x11 "watch -n 5 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'"
ssh x11 "cd ~/LLM/chaos_llm_2026 && tail -f results/QUEUE.log"
ssh x11 "pgrep -af chaos_llm"
```

A healthy job shows **>90 % GPU utilisation** and a stable memory value. Utilisation near 0 % with
memory > 3 GB means a hung/loading process; kill it (`pkill -f chaos_llm`) and rerun.

## 7. Throughput achieved

- 1000-token trajectory: ~1 min.
- 4000-token trajectory: ~3 min.
- Teacher-forced 4000-token response, 4 dirs × 3 radii: ~90 s.
- 300-token A1, 12 dirs: ~40 s.
So a 4 h block does ~200 long trajectories or ~1000 A1 configurations. Plan queues in units of
"jobs × 1 min".
