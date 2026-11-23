"""Decode the top-2 token pairs at the smallest-margin steps of every
model's greedy-orbit corpora (host side; tokenizers only, no GPU model)."""
import glob
import json
import os
import re

import numpy as np
from transformers import AutoTokenizer

MODELS = {
    'deepseek': ('~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B',
                 'results/softloop_gates/deepseek'),
    'phi4mini': ('~/LLM/models--microsoft--Phi-4-mini-instruct',
                 'results/softloop_gates/phi4mini'),
    'qwen25_3b': ('~/LLM/models--Qwen--Qwen2.5-3B-Instruct',
                  'results/softloop_gates/qwen25_3b'),
    'qwen3_4b': ('~/LLM/models--Qwen--Qwen3-4B-Instruct-2507',
                  'results/softloop_gates/qwen3_4b'),
    'mamba130': ('~/LLM/models--state-spaces--mamba-130m-hf',
                 'results/softloop_gates/mamba130'),
    'randqwen': ('~/LLM/models--randqwen-g1.0', 'results/softloop_gates/randqwen'),
}
BIG = {
    'ds14': ('~/LLM/models/DeepSeek-R1-Distill-Qwen-14B',
             'results_big/ds14/softloop_gates'),
}


def classify(a, b):
    wa, wb = a.strip(), b.strip()
    if not wa or not wb:
        return 'empty/whitespace'
    if wa.lower() == wb.lower():
        return 'same word (whitespace/case)'
    if re.fullmatch(r'[\W_]+', wa) and re.fullmatch(r'[\W_]+', wb):
        return 'punctuation'
    if re.search(r'\d', wa) and re.search(r'\d', wb):
        return 'numbers/units'
    if len(wa.split()) and len(wb.split()) and wa.split()[0] == wb.split()[0]:
        return 'shared first word'
    if len(wa) <= 3 and len(wb) <= 3:
        return 'short tokens'
    return 'content-word swap'


def resolve(path):
    p = os.path.expanduser(path)
    if os.path.exists(os.path.join(p, 'tokenizer.json')) or \
       os.path.exists(os.path.join(p, 'tokenizer_config.json')):
        return p
    for snap in sorted(glob.glob(os.path.join(p, 'snapshots', '*'))):
        if os.path.exists(os.path.join(snap, 'tokenizer_config.json')):
            return snap
    return p


out = {}
for name, (mpath, gpath) in list(MODELS.items()) + list(BIG.items()):
    fs = sorted(glob.glob(os.path.join(gpath, '*', 'gates.npz')))
    if not fs:
        continue
    try:
        tok = AutoTokenizer.from_pretrained(resolve(mpath), local_files_only=True)
    except Exception as e:
        print(name, 'tokenizer failed:', str(e)[:80])
        continue
    ms, t1s, t2s = [], [], []
    for f in fs:
        z = np.load(f)
        ms.append(z['m']); t1s.append(z['top1']); t2s.append(z['top2'])
    m = np.concatenate(ms); t1 = np.concatenate(t1s); t2 = np.concatenate(t2s)
    order = np.argsort(m)
    k = min(80, len(order))
    sel = order[:k]
    pairs = []
    cats = {}
    for i in sel:
        a = tok.decode([int(t1[i])])
        b = tok.decode([int(t2[i])])
        c = classify(a, b)
        cats[c] = cats.get(c, 0) + 1
        pairs.append((float(m[i]), a, b, c))
    out[name] = dict(
        n_steps=int(len(m)), median_m=float(np.median(m)),
        frac_m_lt_01=float((m < 0.1).mean()),
        categories={k2: v for k2, v in sorted(cats.items(), key=lambda x: -x[1])},
        pairs=pairs)
    print(f"{name}: median m={np.median(m):.2f}, frac<0.1 = {(m < 0.1).mean():.3f}")
    print('   categories:', out[name]['categories'])
    print('   example pairs (margin, top1, top2):')
    for mm, a, b, c in pairs[:8]:
        print(f'     m={mm:.4f}  {a!r}  vs  {b!r}   [{c}]')

with open('results/softloop_gate_tokens.json', 'w') as f:
    json.dump(out, f, indent=1)
print('saved results/softloop_gate_tokens.json')
