#!/bin/bash
set -e
python3 -m venv --system-site-packages ~/LLM/venv_tf_git
~/LLM/venv_tf_git/bin/pip -q install --upgrade pip
~/LLM/venv_tf_git/bin/pip -q install "git+https://github.com/huggingface/transformers.git"
~/LLM/venv_tf_git/bin/python - <<'EOF'
import transformers
print("transformers", transformers.__version__)
from transformers import AutoConfig
c = AutoConfig.from_pretrained("/home/grego/LLM/models--Qwen--Qwen3.5-2B", local_files_only=True)
print(type(c).__name__, c.model_type)
# inspect available auto classes
import transformers.models as M
print("has qwen3_5:", [k for k in dir(M) if "qwen3_5" in k.lower()])
EOF
