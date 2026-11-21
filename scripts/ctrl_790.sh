cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
M790=~/LLM/models--state-spaces--mamba-790m-hf

wait_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$((15360 - used))" -ge "$1" ] && break
    sleep 90
  done
}

wait_free 5500
if [ ! -f results/ctrl/mamba790/a1i_txt/a1.npz ]; then
  echo "--- a1i mamba790 $(date)"
  $PY -m chaos_llm.run_a1 --model "$M790" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "1e-3,1e-2,0.1,0.3,1,1.4,3,5" --layers "0,12,24,35,47" \
    --out results/ctrl/mamba790/a1i_txt || echo "FAILED a1i790"
fi
for r in 1.4 3 5; do
  if [ ! -f results/ctrl/mamba790/fr_txt_r${r}/free_run.npz ]; then
    wait_free 5500
    echo "--- fr mamba790 r=$r $(date)"
    $PY -m chaos_llm.run_thresholds --model "$M790" --run results/ctrl/mamba790/a1i_txt \
      --radii "$r" --chunk 6 --out results/ctrl/mamba790/fr_txt_r${r} || echo "FAILED fr790 $r"
  fi
done
echo "mamba790 done $(date)"
