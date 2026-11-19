#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
while pgrep -f "queue30.sh" > /dev/null; do sleep 30; done
echo "FIGS start $(date)"
PYTHONPATH=scripts timeout 3600 python3 - <<'EOF'
import os, sys
sys.path.insert(0, "scripts")
import plots
for tag, d in [("phi4mini", "results/models/phi4mini/a1"), ("qwen3_5", "results/models/qwen3_5/a1")]:
    if not os.path.exists(os.path.join(d, "a1.npz")):
        print("skip", tag); continue
    o = os.path.join("results/figs_models", tag)
    os.makedirs(o, exist_ok=True)
    a1 = plots.load_a1(d)
    plots.flux_tube_family(a1, o, tag, light=True)
    plots.lambda_family(a1, o, tag, light=True)
    print("done", tag, flush=True)
EOF
echo "FIGS done $(date)"
