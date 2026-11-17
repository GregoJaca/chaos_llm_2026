import os, glob, json, re, collections
import numpy as np
try:
    import torch
except Exception:
    torch = None

ROOT = os.path.expanduser('~/LLM')
print('# Data inventory (generated)\n')

def safe_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def pt_shape(p):
    if torch is None:
        return None
    try:
        t = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    except Exception:
        try:
            t = torch.load(p, map_location='cpu', weights_only=True)
        except Exception as e:
            return f'ERR {repr(e)[:60]}'
    return tuple(t.shape) if hasattr(t, 'shape') else type(t).__name__

# ---------------- chaos_llm_2026 token outputs ----------------
print('## chaos_llm_2026/outputs_* (token sequences only)\n')
base = ROOT + '/chaos_llm_2026'
for od in sorted(glob.glob(base + '/outputs*')):
    if not os.path.isdir(od): continue
    rows = []
    for f in glob.glob(od + '/*/config.json'):
        c = safe_json(f)
        if not c: continue
        rt = c.get('runtime', {})
        rows.append(dict(
            model=os.path.basename(c.get('paths', {}).get('model_path', '?')),
            dtype=c.get('project', {}).get('dtype'),
            nc=c.get('perturbation', {}).get('num_conditions'),
            ss=c.get('perturbation', {}).get('subspace_seed'),
            batch=c.get('generation', {}).get('batch_size'),
            win=rt.get('sliding_window'),
            mag=rt.get('perturbation_magnitude'),
            maxnew=c.get('generation', {}).get('max_new_tokens'),
            prompt=c.get('prompt', {}).get('name'),
        ))
    if not rows: continue
    u = collections.Counter((r['model'], r['dtype'], r['nc'], r['ss'], r['batch'], r['maxnew']) for r in rows)
    wins = sorted({r['win'] for r in rows if r['win'] is not None})
    mags = sorted({r['mag'] for r in rows if r['mag'] is not None})
    r0 = any((r['mag'] == 0) for r in rows)
    print(f"- `{os.path.basename(od)}`: {len(rows)} runs; windows={wins}; mags={mags}; r=0 control={r0}")
    for k, n in u.most_common():
        print(f"    - n={n}: model={k[0]}, dtype={k[1]}, n_cond={k[2]}, subspace_seed={k[3]}, batch={k[4]}, max_new={k[5]}")
    # sample baseline length
    zf = glob.glob(od + '/*/tokens.npz')
    if zf:
        z = np.load(zf[0])
        print(f"    - sample tokens.npz: baseline={z['baseline_ids'].shape}, perturbed={z['perturbed_ids'].shape}, prompt_len={int(z['prompt_len'])}")
    print()

# ---------------- TDK results_perturbations ----------------
print('## TDK_LLM/runs_aug/results_perturbations* (layer-wise hidden states, single-token prompts)\n')
tdk = ROOT + '/TDK_LLM/runs_aug'
for ds in sorted(glob.glob(tdk + '/results_perturbations*')):
    if not os.path.isdir(ds): continue
    run_dirs = [d for d in glob.glob(ds + '/*/') if os.path.isdir(d) and os.path.basename(d.rstrip('/')).startswith('single_token')]
    if not run_dirs:
        print(f"- `{os.path.basename(ds)}`: no single_token run dirs (maybe aggregated/excluded)\n"); continue
    cfg = safe_json(run_dirs[0] + 'config.json') or {}
    radii = sorted({float(m.group(1)) for d in run_dirs for m in [re.search(r'_r([0-9.eE+-]+)_s\d+', d)] if m})
    seeds = sorted({int(m.group(1)) for d in run_dirs for m in [re.search(r'_s(\d+)$', d.rstrip('/'))] if m})
    files = glob.glob(run_dirs[0] + 'hidden_states_cond_*_layer_*.pt')
    shape = pt_shape(files[0]) if files else None
    print(f"- `{os.path.basename(ds)}`: {len(run_dirs)} run dirs; radii={radii}; seeds={seeds}; r=0 present={0.0 in radii}")
    print(f"    config: model={cfg.get('model', {}).get('name')}, dtype={cfg.get('model', {}).get('dtype')}, n_conditions={cfg.get('experiment', {}).get('n_conditions')}, batch={cfg.get('experiment', {}).get('batch_size')}")
    print(f"    prompts={len(cfg.get('experiment', {}).get('prompts', []))}; sample shape={shape}; hidden files/run={len(files)}")
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
    gt = safe_json(d + 'generated_text.json')
    cfgp = [p for p in glob.glob(d + '*config*') if os.path.isfile(p)]
    print(f"- `{name}`: {len(hs)} conditions; sample shape={shape}")
    if res: print(f"    results.json keys: {list(res.keys())[:12]}")
    if gt: print(f"    generated_text.json keys: {list(gt.keys())[:12]}")
    print(f"    config files: {[os.path.basename(p) for p in cfgp]}")
    print()

# ---------------- results_pecs ----------------
print('## /home/grego/LLM/results_pecs (multi-prompt, temperatures/radii)\n')
for d in sorted(glob.glob(ROOT + '/results_pecs/*/')):
    if not os.path.isdir(d): continue
    name = os.path.basename(d.rstrip('/'))
    files = os.listdir(d)
    pats = collections.Counter(re.sub(r'[0-9]+', 'N', f) for f in files if f.endswith('.pt'))
    hs = glob.glob(d + 'hidden_states_cond_*_layer_*.pt')
    shape = pt_shape(hs[0]) if hs else None
    cfgp = [p for p in glob.glob(d + '*config*') if os.path.isfile(p)]
    print(f"- `{name}`: {len(files)} files; hidden={len(hs)}; sample shape={shape}; configs={[os.path.basename(p) for p in cfgp]}")
    print(f"    patterns: {dict(list(pats.items())[:5])}")
    print()

# ---------------- results_10_april ----------------
print('## /home/grego/LLM/results_10_april (temperature/radius runs, recurrence plots)\n')
for d in sorted(glob.glob(ROOT + '/results_10_april/launch_pentek_*/')):
    name = os.path.basename(d.rstrip('/'))
    files = os.listdir(d)
    hs = glob.glob(d + 'hidden_states_cond_*_layer_*.pt')
    shape = pt_shape(hs[0]) if hs else None
    cfgp = [p for p in glob.glob(d + '*config*') if os.path.isfile(p)]
    print(f"- `{name}`: {len(files)} files; hidden={len(hs)}; sample shape={shape}; configs={[os.path.basename(p) for p in cfgp]}")
print(f"- recurrence_plots_doc: {len(glob.glob(ROOT + '/results_10_april/recurrence_plots_doc/*'))} files")
print()

# ---------------- launch_aug ----------------
print('## /home/grego/LLM/launch_aug (temperature/radius runs)\n')
for d in sorted(glob.glob(ROOT + '/launch_aug/*/')):
    name = os.path.basename(d.rstrip('/'))
    files = os.listdir(d)
    hs = glob.glob(d + 'hidden_states_cond_*_layer_*.pt')
    shape = pt_shape(hs[0]) if hs else None
    cfgp = [p for p in glob.glob(d + '*config*') if os.path.isfile(p)]
    print(f"- `{name}`: {len(files)} files; hidden={len(hs)}; sample shape={shape}; configs={[os.path.basename(p) for p in cfgp]}")
print()

# ---------------- data_analysis ----------------
print('## /home/grego/LLM/data_analysis (previous analysis code/results)\n')
for d in sorted(glob.glob(ROOT + '/data_analysis/*')):
    if os.path.isdir(d):
        n = len(os.listdir(d))
        print(f"- `{os.path.basename(d)}/`: {n} entries")
    elif d.endswith(('.py', '.md', '.json', '.txt', '.yaml')):
        print(f"- `{os.path.basename(d)}`")
