import sys
sys.path.insert(0, "src")
from chaos_llm.protocol import model_info
for p in ["~/LLM/models--microsoft--Phi-4-mini-instruct"]:
    print(model_info(p))
