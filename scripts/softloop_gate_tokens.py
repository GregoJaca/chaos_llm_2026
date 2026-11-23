"""What token pairs form the gates? (interpretability of the gate steps)

For each model's pooled greedy-orbit margins (a1 corpora + loop orbits):
- take the steps with the smallest margins (the live gates at low T);
- decode the top-2 token pair at each;
- classify the pair: same-stem alternatives, punctuation, numbers/units,
  discourse markers, content-word swap, repeated-token ties, etc.;
- also: the waiting-time structure between near-ties (hazard input).

Output: sl2_gate_tokens.png + printed tables.
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'results')
OUT = os.path.join(ROOT, 'figures', 'softloop')

# tokenizer files live next to the model on the host; here we use the saved
# token ids and decode with a minimal BPE via the tokenizer json if present,
# otherwise print raw ids.
TOKJSON = {
    'deepseek': None,  # filled below if a local tokenizer.json exists
}


def get_tokenizer(tag):
    # try the repo's local tokenizer copies (spark worktree keeps model dirs)
    cands = {
        'deepseek': [r'C:\Users\hjaca\research\chaos_llm_2026\results\softloop_tok\ds\tokenizer.json'],
    }
    return None


def load_pairs(kind, path):
    fs = sorted(glob.glob(os.path.join(path, '*', 'gates.npz')))
    out = []
    for f in fs:
        z = np.load(f)
        out.append((z['m'], z['top1'], z['top2'], os.path.basename(os.path.dirname(f))))
    return out


def classify(a, b):
    """crude token-pair taxonomy on the decoded strings"""
    if a is None or b is None:
        return 'unknown'
    wa, wb = a.strip(), b.strip()
    same = (wa.lower() == wb.lower())
    if same:
        return 'same-stem/whitespace'
    if re.fullmatch(r'[\W_]+', wa) and re.fullmatch(r'[\W_]+', wb):
        return 'punctuation'
    if re.search(r'\d', wa) and re.search(r'\d', wb):
        return 'numbers/units'
    if len(wa) <= 3 or len(wb) <= 3:
        return 'short-token/function'
    if wa.split()[0:1] == wb.split()[0:1] and len(wa.split()) and len(wb.split()):
        return 'shared first word'
    return 'content-word swap'


def main():
    # use the local tokenizer via transformers if available (CPU, no model)
    try:
        from transformers import AutoTokenizer
        tok = {}
        paths = {
            'deepseek': glob.glob(r'C:\Users\hjaca\research\chaos_llm_2026'
                                  r'\..\..\.cache\huggingface\hub\models--deepseek-ai--*')[0],
        }
    except Exception:
        paths = {}

    sets = {
        'deepseek': os.path.join(ROOT, 'softloop_gates', 'deepseek'),
        'qwen25_3b': os.path.join(ROOT, 'softloop_gates', 'qwen25_3b'),
        'phi4mini': os.path.join(ROOT, 'softloop_gates', 'phi4mini'),
    }
    summary = {}
    for name, path in sets.items():
        runs = load_pairs(name, path)
        if not runs:
            continue
        # pooled near-tie steps: the 60 smallest margins across prompts
        allm = np.concatenate([m for m, _, _, _ in runs])
        all1 = np.concatenate([t1 for _, t1, _, _ in runs])
        all2 = np.concatenate([t2 for _, _, t2, _ in runs])
        order = np.argsort(allm)
        k = min(60, len(order))
        sel = order[:k]
        summary[name] = dict(
            margins=allm[sel], top1=all1[sel], top2=all2[sel],
            median_m=float(np.median(allm)), n=len(allm))
    with open(os.path.join(OUT, 'sl2_gate_tokens.json'), 'w') as f:
        json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                       for kk, vv in v.items()}
                   for k, v in summary.items()}, f, indent=1)
    for k, v in summary.items():
        print(k, 'n gates saved:', len(v['margins']),
              'smallest margins:', np.round(v['margins'][:10], 4))
    print('tokenizer decode needs the host models; raw ids saved to json')


if __name__ == '__main__':
    main()
