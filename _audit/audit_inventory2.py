import os, glob, json, re, collections
import numpy as np
try:
    import torch
except Exception:
    torch = None

ROOT = os.path.expanduser('~/LLM')
print('# Data inventory (generated)\n')

def safe_json(p):
    try: return json.load(open(p))
    except Exception: return None

def pt_shape(p):
    if torch is None: return None
    try: t = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    except Exception:
        try: t = torch.load(p, map_location='cpu', weights_only=True)
        except Exception as e: return f'ERR {repr(e)[:60]}'
    return tuple(t.shape) if hasattr(t, 'shape') else type(t).__name__

def fmt(x): return json.dumps(x) if isinstance(x,(dict,list)) else str(x)

# ---------------- chaos token outputs ----------------
print('## chaos_llm_2026/outputs_* (token sequences only)\n')
base = ROOT + '/chaos_llm_2026'
for od in sorted(glob.glob(base + '/outputs*')):
    if not os.path.isdir(od): continue
    rows = []
    for f in glob.glob(od + '/*/config.json'):
        c = safe_json(f)
        if not c: continue
        rt = c.get('runtime', {})
        rows.append(dict(model=os.path.basename(c.get('paths', {}).get('model_path', '?')),
            dtype=c.get('project', {}).get('dtype'), nc=c.get('perturbation', {}).get('num_conditions'),
            ss=c.get('perturbation', {}).get('subspace_seed'), batch=c.get('generation', {}).get('batch_size'),
            win=rt.get('sliding_window'), mag=rt.get('perturbation_magnitude'),
            maxnew=c.get('generation', {}).get('max_new_tokens'), prompt=c.get('prompt', {}).get('name')))
    if not rows: continue
    u = collections.Counter((r['model'], r['dtype'], r['nc'], r['ss'], r['batch'], r['maxnew']) for r in rows)
    wins = sorted({r['win'] for r in rows if r['win'] is not None})
    mags = sorted({r['mag'] for r in rows if r['mag'] is not None})
    r0 = any(r['mag'] == 0 for r in rows)
    print(f"- `{os.path.basename(od)}`: {len(rows)} runs; windows={wins}; mags={mags}; r=0 control={r0}")
    for k, n in u.most_common():
        print(f"    - n={n}: model={k[0]}, dtype={k[1]}, n_cond={k[2]}, seed={k[3]}, batch={k[4]}, max_new={k[5]}")
    zf = glob.glob(od + '/*/tokens.npz')
    if zf:
        z = np.load(zf[0])
        print(f"    - sample: baseline={z['baseline_ids'].shape}, perturbed={z['perturbed_ids'].shape}, prompt_len={int(z['prompt_len'])}")
    print()

# ---------------- TDK layer-wise hidden states ----------------
print('## TDK_LLM/runs_aug/results_perturbations* (layer-wise hidden states, single-token prompts)\n')
tdk = ROOT + '/TDK_LLM/runs_aug'
for ds in sorted(glob.glob(tdk + '/results_perturbations*')):
    if not os.path.isdir(ds): continue
    run_dirs = [d for d in glob.glob(ds + '/*/') if os.path.isdir(d) and 'aggregated' not in d]
    if not run_dirs:
        print(f"- `{os.path.basename(ds)}`: no run dirs\n"); continue
    cfgs = []
    for d in run_dirs:
        c = safe_json(d + 'config.json')
        if c: cfgs.append((d, c))
    radii = sorted({float(c.get('radius')) for _, c in cfgs if c.get('radius') is not None})
    seeds = sorted({int(c.get('seed')) for _, c in cfgs if c.get('seed') is not None})
    nconds = sorted({int(c.get('n_conditions')) for _, c in cfgs if c.get('n_conditions') is not None})
    pts = glob.glob(run_dirs[0] + '*layer_0.pt') or glob.glob(run_dirs[0] + '*.pt')
    shape = pt_shape(pts[0]) if pts else None
    example = cfgs[0][1] if cfgs else {}
    print(f"- `{os.path.basename(ds)}`: {len(run_dirs)} run dirs; radii={radii}; seeds={seeds}; n_conditions={nconds}; r=0 present={0.0 in radii}")
    print(f"    config keys: {list(example.keys())}")
    print(f"    example: radius={example.get('radius')}, n_conditions={example.get('n_conditions')}, prompt_text={str(example.get('prompt_text'))[:60]!r}, max_new={example.get('max_new_tokens')}, seed={example.get('seed')}")
    print(f"    sample layer_0 shape={shape}; hidden files/run={len(pts)}")
    print("    model/dtype NOT recorded per run; dataset name is the only clue (see notes).")
    print()

# ---------------- launch_sep ----------------
print('## /home/grego/LLM/launch_sep (last-layer token trajectories, 100 conditions)\n')
for d in sorted(glob.glob(ROOT + '/launch_sep/*/')):
    if not os.path.isdir(d): continue
    name = os.path.basename(d.rstrip('/'))
    hs = glob.glob(d + 'hidden_states_cond_*_layer_-1.pt')
    if not hs:
        print(f"- `{name}`: no hidden_states_cond files\n"); continue
    shape = pt_shape(hs[0])
    res = safe_json(d + 'results.json')
    mc = None
    if isinstance(res, list) and res and isinstance(res[0], dict):
        mc = res[0].get('model_configuration')
    print(f"- `{name}`: {len(hs)} conditions; sample shape={shape}")
    if mc: print(f"    model_configuration={fmt(mc)}")
    # check time length vs generated tokens
    if isinstance(res, list) and res and isinstance(res[0], dict):
        print(f"    results.json: list[{len(res)}], item keys={list(res[0].keys())}")
    print("    no r=0 condition; all 100 conditions are perturbed at the directory radius.")
    print("    NOTE: repetition_penalty is 1.1 (not pure greedy argmax).")
    print()

# ---------------- results_pecs ----------------
print('## /home/grego/LLM/results_pecs (single-trajectory hidden states + distance matrices)\n')
for d in sorted(glob.glob(ROOT + '/results_pecs/*/')):
    if not os.path.isdir(d): continue
    name = os.path.basename(d.rstrip('/'))
    files = os.listdir(d)
    if not files: continue
    res = safe_json(d + 'result.json')
    mc = res.get('model_configuration') if isinstance(res, dict) else None
    hs = glob.glob(d + 'hidden_states_*.pt')
    shape = pt_shape(hs[0]) if hs else None
    print(f"- `{name}`: {len(files)} files; hidden shapes sample={shape}; model_configuration={fmt(mc)}")
    print(f"    files: {sorted(files)}")
    print()

# ---------------- results_10_april ----------------
print('## /home/grego/LLM/results_10_april (temperature/radius runs, recurrence plots)\n')
for d in sorted(glob.glob(ROOT + '/results_10_april/launch_pentek_*/')):
    name = os.path.basename(d.rstrip('/'))
    files = os.listdir(d)
    hs = glob.glob(d + 'hidden_states_cond_*_layer_*.pt')
    shape = pt_shape(hs[0]) if hs else None
    res = safe_json(d + 'result.json')
    mc = res.get('model_configuration') if isinstance(res, dict) else None
    print(f"- `{name}`: {len(files)} files; hidden={len(hs)}; sample shape={shape}; model_configuration={fmt(mc)}")
    print(f"    files sample: {sorted(files)[:12]}")
print(f"- `recurrence_plots_doc`: {len(glob.glob(ROOT + '/results_10_april/recurrence_plots_doc/*'))} files")
print()

# ---------------- launch_aug ----------------
print('## /home/grego/LLM/launch_aug (temperature/radius runs)\n')
for d in sorted(glob.glob(ROOT + '/launch_aug/*/')):
    name = os.path.basename(d.rstrip('/'))
    files = os.listdir(d)
    hs = glob.glob(d + 'hidden_states_cond_*_layer_*.pt')
    shape = pt_shape(hs[0]) if hs else None
    res = safe_json(d + 'result.json')
    mc = res.get('model_configuration') if isinstance(res, dict) else None
    print(f"- `{name}`: {len(files)} files; hidden={len(hs)}; sample shape={shape}; model_configuration={fmt(mc)}")
    print(f"    files sample: {sorted(files)[:12]}")
print()

# ---------------- data_analysis ----------------
print('## /home/grego/LLM/data_analysis (previous analysis code/results)\n')
for d in sorted(glob.glob(ROOT + '/data_analysis/*')):
    if os.path.isdir(d):
        print(f"- `{os.path.basename(d)}/`: {len(os.listdir(d))} entries")
    elif d.endswith(('.py', '.md', '.json', '.txt', '.yaml')):
        print(f"- `{os.path.basename(d)}`")
