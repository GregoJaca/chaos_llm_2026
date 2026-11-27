"""Decode what the locked loop is repeating (CPU only, no model weights).

Handles both a raw model directory and a HuggingFace cache root
(models--org--name/snapshots/<hash>/...).
"""
import glob
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

RUNS = {
    "DeepSeek-1.5B": ("~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B", "results/longloop/ds15/T1.0/loop/softloop.npz"),
    "Qwen2.5-3B": ("~/LLM/models--Qwen--Qwen2.5-3B-Instruct", "results/longloop/qwen25/T1.0/loop/softloop.npz"),
    "Qwen3-4B": ("~/LLM/models--Qwen--Qwen3-4B-Instruct-2507", "results/longloop/qwen34/T1.0/loop/softloop.npz"),
    "Qwen3.5-2B": ("~/LLM/models--Qwen--Qwen3.5-2B", "results/longloop/qwen35/T1.0/loop/softloop.npz"),
    "Phi-4-mini": ("~/LLM/models--microsoft--Phi-4-mini-instruct", "results/longloop/phi/T1.0/loop/softloop.npz"),
    "Mamba-130m": ("~/LLM/models--state-spaces--mamba-130m-hf", "results/longloop/mamba/T1.0/loop/softloop.npz"),
    "Pythia-410m": ("~/LLM/models--EleutherAI--pythia-410m", "results/longloop/pythia/T1.0/loop/softloop.npz"),
    "randQwen": ("~/LLM/models--randqwen-g1.0", "results/longloop/randqwen/T1.0/loop/softloop.npz"),
    "Qwen3.5-9B": ("/home/grego/LLM/models/Qwen3.5-9B", "results/longloop/qwen35_9b/T1.0/loop/softloop.npz"),
    "Qwen3-14B": ("/home/grego/LLM/models/Qwen3-14B", "results/longloop/qwen314/T1.0/loop/softloop.npz"),
    "DeepSeek-14B": ("/home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B", "results/longloop/ds14/T1.0/loop/softloop.npz"),
    "Phi-4": ("/home/grego/LLM/models/phi-4", "results/longloop/phi4/T1.0/loop/softloop.npz"),
}


def resolve(mpath):
    mpath = os.path.expanduser(mpath)
    if os.path.exists(os.path.join(mpath, "config.json")):
        return mpath
    snaps = sorted(glob.glob(os.path.join(mpath, "snapshots", "*")))
    for s in snaps:
        if os.path.exists(os.path.join(s, "config.json")):
            return s
    return mpath


out = {}
for name, (mpath, npz) in RUNS.items():
    p = os.path.expanduser(npz)
    if not os.path.exists(p):
        print(name, "missing", npz); continue
    z = np.load(p, allow_pickle=True)
    top1 = z["top1_ref"].astype(int)
    tail = top1[-100:]
    try:
        tok = AutoTokenizer.from_pretrained(resolve(mpath), trust_remote_code=True)
        dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False)
    except Exception as e:
        print(name, "tokenizer failed:", str(e)[:120])
        dec = lambda ids: " ".join(str(int(i)) for i in ids)
    per = 0
    for pp in range(1, 25):
        if np.all(tail[pp:] == tail[:-pp]):
            per = pp; break
    uniq, cnt = np.unique(tail, return_counts=True)
    order = np.argsort(cnt)[::-1][:6]
    out[name] = dict(period=int(per),
                     last_token=dec([tail[-1]]),
                     loop_text=dec(tail[-min(per * 3 if per else 40, 40):]),
                     uniq=int(len(uniq)),
                     top=[[dec([uniq[k]]), int(cnt[k])] for k in order])
    print("=" * 70)
    print(name, "period", per, "uniq(last100)", len(uniq))
    print("  last token:", repr(out[name]["last_token"]))
    print("  last 40/loop tokens:", repr(out[name]["loop_text"]))
    print("  most frequent:", out[name]["top"])

json.dump(out, open("results/loop_repetition_tokens.json", "w"), indent=1)
print("saved results/loop_repetition_tokens.json", len(out), "models")
