from transformers import AutoConfig, AutoModelForCausalLM
import torch

path = "/home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B"
cfg = AutoConfig.from_pretrained(path, local_files_only=True)
print("after AutoConfig: _attn_implementation =", repr(getattr(cfg, "_attn_implementation", "MISSING")))
print("internal =", repr(getattr(cfg, "_attn_implementation_internal", "MISSING")))
try:
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": 0})
    print("LOAD OK", m.config._attn_implementation)
    del m
    torch.cuda.empty_cache()
except Exception as e:
    print("LOAD FAIL:", type(e).__name__, str(e)[:200])
