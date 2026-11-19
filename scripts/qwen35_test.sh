#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
PY=~/LLM/venv311b/bin/python
MODEL=~/LLM/models--Qwen--Qwen3.5-2B
for i in $(seq 1 60); do
  $PY -c "import torch, transformers" >/dev/null 2>&1 && break
  sleep 60
done
echo "venv ready $(date)"
$PY - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "tf", transformers.__version__)
from transformers import AutoConfig, AutoTokenizer
p = "/home/grego/LLM/models--Qwen--Qwen3.5-2B"
cfg = AutoConfig.from_pretrained(p, local_files_only=True)
print("config", type(cfg).__name__, cfg.model_type, getattr(cfg, "text_config", None) is not None)
tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
ids = tok("Explain how tides work.", return_tensors="pt").input_ids
print("tokens", ids.shape)
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
import traceback
for name, cls in [("CausalLM", AutoModelForCausalLM), ("ImageText", AutoModelForImageTextToText)]:
    try:
        m = cls.from_pretrained(p, dtype=torch.float16, device_map="cuda", local_files_only=True)
        print(name, "loaded", type(m).__name__)
        with torch.inference_mode():
            o = m(input_ids=ids.cuda(), output_hidden_states=True)
        print("  forward ok, logits", tuple(o.logits.shape), "hidden_states", len(o.hidden_states) if o.hidden_states else None)
        del m
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(name, "FAILED", type(e).__name__, str(e)[:200])
EOF
echo "QWEN35 TEST DONE $(date)"
