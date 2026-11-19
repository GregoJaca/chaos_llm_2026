#!/bin/bash
set -e
python3 -m venv --system-site-packages ~/LLM/venv_tf
~/LLM/venv_tf/bin/pip -q install --upgrade transformers
~/LLM/venv_tf/bin/python - <<'EOF'
import transformers, torch
print("transformers", transformers.__version__, "torch", torch.__version__)
from transformers import AutoConfig
import sys
for p in ["/home/grego/LLM/models--Qwen--Qwen3.5-2B",
          "/home/grego/LLM/models--Qwen--Qwen3-4B-Instruct-2507",
          "/home/grego/LLM/models--microsoft--Phi-4-mini-instruct"]:
    try:
        c = AutoConfig.from_pretrained(p, local_files_only=True)
        print(p.split("--")[-1], type(c).__name__, c.hidden_size, c.num_hidden_layers,
              getattr(c, "sliding_window", None), getattr(c, "use_sliding_window", None))
    except Exception as e:
        print(p, "ERR", type(e).__name__, str(e)[:120])
EOF
