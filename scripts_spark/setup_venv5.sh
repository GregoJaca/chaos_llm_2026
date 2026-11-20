#!/bin/bash
# Separate venv for Qwen3.5 (hybrid Gated-DeltaNet): needs transformers 5.x from git.
set -e
python3 -m venv ~/LLM/venv5
~/LLM/venv5/bin/pip install -q -U pip
~/LLM/venv5/bin/pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
~/LLM/venv5/bin/pip install -q "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
~/LLM/venv5/bin/pip install -q accelerate numpy scipy matplotlib
~/LLM/venv5/bin/python - <<'EOF'
import torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
try:
    from transformers import Qwen3_5ForCausalLM
    print('Qwen3_5 class available')
except Exception as e:
    print('Qwen3_5 class missing:', e)
print('VENV5_READY')
EOF
