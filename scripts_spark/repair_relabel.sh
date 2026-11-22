#!/bin/bash
# Repair the relabel: move the actual full-attention outputs into the *_full
# dirs (the git-tracked meta/summary already sit there), so the phase1b
# guards see them as absent and rerun the true-window core.
set -e
cd ~/LLM/chaos_llm_2026/results_big/ds14
mv a1_inter_w100/a1.npz a1_inter_w100/parts a1_inter_full/
rmdir a1_inter_w100 2>/dev/null || mv a1_inter_w100/a1_inter_w100.log . 2>/dev/null || true
mv free_inter_w100/free_run.npz free_inter_full/ 2>/dev/null || true
rm -rf free_inter_w100
ls
echo REPAIR_DONE
