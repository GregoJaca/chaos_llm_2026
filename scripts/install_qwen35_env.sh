#!/bin/bash
pkill -f "venv311/bin/pip" >/dev/null 2>&1
/usr/bin/python3.11 -m venv ~/LLM/venv311b
~/LLM/venv311b/bin/pip -q install --upgrade pip
~/LLM/venv311b/bin/pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2
~/LLM/venv311b/bin/pip install "transformers>=5.0.0" numpy scipy matplotlib accelerate 2>&1 | tail -2
~/LLM/venv311b/bin/python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
EOF
