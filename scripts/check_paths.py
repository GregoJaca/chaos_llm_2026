import sys, os, glob
sys.path.insert(0, "src")
from chaos_llm.protocol import resolve_model_path, model_info
for p in ["~/LLM/models--microsoft--Phi-4-mini-instruct",
          "~/LLM/models--Qwen--Qwen3-4B-Instruct-2507",
          "~/LLM/models--Qwen--Qwen3.5-2B"]:
    try:
        r = resolve_model_path(p)
        print(p, "->", r, "config?", os.path.exists(os.path.join(r, "config.json")))
        print("   info", model_info(p))
    except Exception as e:
        print(p, "ERR", type(e).__name__, str(e)[:160])
