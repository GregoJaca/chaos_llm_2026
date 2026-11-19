#!/bin/bash
cd ~/LLM/chaos_llm_2026
find results/figs_models -name "*.png" -printf "%f\n" | grep -v -E "__L[0-9]+_" | sort | head -60
echo "--- total"
find results/figs_models -name "*.png" | wc -l
