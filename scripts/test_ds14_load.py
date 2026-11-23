import torch
from transformers import AutoModelForCausalLM

model_path = "/home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B"
free_b, _ = torch.cuda.mem_get_info()
print(f"Free before: {free_b/1e9:.2f} GB")

# Use dtype=torch.bfloat16 or torch.float16
m = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True
)
print("Model dtype:", m.dtype)
p0 = next(m.parameters())
print("First param dtype:", p0.dtype, "shape:", p0.shape)

free_a, _ = torch.cuda.mem_get_info()
print(f"Free after: {free_a/1e9:.2f} GB, Used: {(free_b - free_a)/1e9:.2f} GB")
