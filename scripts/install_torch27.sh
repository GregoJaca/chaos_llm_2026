#!/bin/bash
set -e
~/LLM/venv311/bin/pip -q install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
~/LLM/venv311/bin/python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
from transformers import AutoConfig
c = AutoConfig.from_pretrained("/home/grego/LLM/models--Qwen--Qwen3.5-2B", local_files_only=True)
print("config ok", type(c).__name__, c.model_type)
EOF
