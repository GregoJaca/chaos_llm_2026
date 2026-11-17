import os, glob, json, numpy as np, torch
from transformers import AutoModelForCausalLM

ROOT = os.path.expanduser('~/LLM/chaos_llm_2026')
OUT = os.path.join(ROOT, 'analysis_audit'); os.makedirs(OUT, exist_ok=True)
snap = glob.glob(os.path.expanduser('~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/'))[0]
model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32, local_files_only=True,
                                             attn_implementation='sdpa').cuda().eval()
model.config.use_sliding_window = True

cands = {
    10:   ('outputs_0.0001',  'run_10_0.0001_interstellar_travel_seed_69'),
    50:   ('outputs_OLD',     'run_50_0.0004_interstellar_travel_seed_69'),
    100:  ('outputs_OLD',     'run_100_0.0004_interstellar_travel_seed_69'),
    200:  ('outputs_OLD',     'run_200_0.0004_interstellar_travel_seed_69'),
    500:  ('outputs_OLD',     'run_500_0.0004_interstellar_travel_seed_69'),
    1000: ('outputs_0.0001',  'run_1000_0.0001_interstellar_travel_seed_69'),
    4096: ('outputs_OLD',     'run_4096_0.0004_interstellar_travel_seed_69'),
}
out = {}
for W, (od, rn) in cands.items():
    d = os.path.join(ROOT, od, rn)
    if not os.path.isdir(d):
        print('missing', W, d, flush=True); continue
    z = np.load(os.path.join(d, 'tokens.npz'))
    b = z['baseline_ids']; P = int(z['prompt_len'])
    # dtype from config
    c = json.load(open(os.path.join(d, 'config.json')))
    dt = c['project']['dtype']
    if len(b) > 3000:
        print('skip long', W, len(b), flush=True); continue
    model.config.sliding_window = W
    ids = torch.tensor(b)[None].cuda()
    with torch.no_grad():
        lo = model(ids, use_cache=False).logits[0]
    t2 = torch.topk(lo, k=2, dim=-1)
    m = (t2.values[:,0] - t2.values[:,1]).cpu().numpy()
    a = t2.indices[:,0].cpu().numpy()
    gen = np.arange(P-1, len(b)-1)
    pred = a[gen]; stored = b[P:]
    dis = np.where(pred != stored)[0]
    first = int(dis[0]) if len(dis) else -1
    imin = int(np.argmin(m[gen]))
    rec = dict(window=W, stored_dir=f'{od}/{rn}', stored_dtype=dt, baseline_len=int(len(b)),
               n_disagreements=int(len(dis)), first_disagreement=first,
               margin_at_first=float(m[gen][first]) if first >= 0 else None,
               min_margin=float(m[gen][imin]), min_margin_step=int(imin),
               median_margin=float(np.median(m[gen])),
               top1_at_min=int(t2.indices[gen[imin],0]), top2_at_min=int(t2.indices[gen[imin],1]),
               stored_token_at_min=int(stored[imin]))
    # modal free-run T for the same window/magnitude family (if available)
    out[W] = rec
    print(json.dumps(rec), flush=True)
    del lo, t2
    torch.cuda.empty_cache()

with open(os.path.join(OUT, 'margin_audit.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('saved', os.path.join(OUT, 'margin_audit.json'))
