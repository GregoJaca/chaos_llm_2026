#!/bin/bash
cd ~/LLM/chaos_llm_2026
find results/figs_models -name "*.png" -printf "%P\n" | sort > /tmp/figlist.txt
wc -l < /tmp/figlist.txt
head -50 /tmp/figlist.txt
