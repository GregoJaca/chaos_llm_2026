#!/bin/bash
set -e
/usr/bin/python3.11 -m venv ~/LLM/venv311
~/LLM/venv311/bin/pip -q install --upgrade pip
~/LLM/venv311/bin/pip -q install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
~/LLM/venv311/bin/pip -q install "git+https://github.com/huggingface/transformers.git"
~/LLM/venv311/bin/pip -q install numpy scipy matplotlib accelerate
~/LLM/venv311/bin/python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
from transformers import AutoConfig
c = AutoConfig.from_pretrained("/home/grego/LLM/models--Qwen--Qwen3.5-2B", local_files_only=True)
print("loaded config", type(c).__name__, getattr(c, "model_type", "?"))
EOF
