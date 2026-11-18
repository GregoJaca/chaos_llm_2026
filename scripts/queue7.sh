#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "QUEUE7 START $(date)"

# Long trajectories: 2000 tokens each (window 4096)
python3 -m chaos_llm.collect_trajectories --prompts prompts_long2.txt --window 4096 \
  --max-new-tokens 2000 --layers 7,14,21,27 --out results/long_traj_2k || echo "FAILED lt2k"

# Two very long ones (4000 tokens)
printf 'history_of_computing\tProvide a comprehensive history of computing from mechanical calculators to modern artificial intelligence. Cover key inventions, the evolution of programming languages, operating systems, networking, and the major figures and companies involved. Explain the technical ideas behind each era in accessible language and discuss how these developments shaped society.\nclimate_models\tExplain how climate models work. Cover radiative transfer, ocean and atmosphere dynamics, grid resolution, parameterization of clouds, coupling, and the use of ensembles. Discuss how models are evaluated and where uncertainty comes from.\n' > prompts_long4k.txt
python3 -m chaos_llm.collect_trajectories --prompts prompts_long4k.txt --window 4096 \
  --max-new-tokens 4000 --layers 7,14,21,27 --out results/long_traj_4k || echo "FAILED lt4k"

echo "QUEUE7 DONE $(date)"
