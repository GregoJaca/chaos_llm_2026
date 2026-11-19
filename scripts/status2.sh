#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "procs:"
ps -eo pid,etime,args | grep -E "chain_models|model_suite2|install_torch27|python3.11" | grep -v grep | head -6
echo "gpu:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "suite tail:"
tail -2 results/ms_qwen3_4b.log
echo "chain log:"
cat results/chain_models.log 2>/dev/null | tail -3
echo "models:"
ls results/models
echo "torch311:"
~/LLM/venv311/bin/python - <<'EOF'
try:
    import torch, transformers
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "tf", transformers.__version__)
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:120])
EOF
