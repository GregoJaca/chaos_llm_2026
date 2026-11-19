import sys
sys.path.insert(0, "src")
import torch
from chaos_llm.protocol import load_model
m, t, p = load_model("~/LLM/models--Qwen--Qwen3-4B-Instruct-2507", attn="eager", dtype="float16")
print(type(m).__name__, type(getattr(m, "model", None)).__name__)
inner = m.model
print([a for a in dir(inner) if "mask" in a.lower() or "causal" in a.lower()])
print("attn impl", m.config._attn_implementation)
import inspect
print(inspect.signature(inner.forward).parameters.keys())
