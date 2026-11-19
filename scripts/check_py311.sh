#!/bin/bash
echo "--- venv311"
ls -d ~/LLM/venv311 2>/dev/null && du -sh ~/LLM/venv311 2>/dev/null
ps -eo pid,etime,args | grep -E "pip|setup_py311" | grep -v grep | head -3
if [ -x ~/LLM/venv311/bin/python ]; then
  ~/LLM/venv311/bin/python - <<'EOF'
try:
    import torch, transformers
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("transformers", transformers.__version__)
except Exception as e:
    print("import error:", type(e).__name__, str(e)[:200])
EOF
fi
