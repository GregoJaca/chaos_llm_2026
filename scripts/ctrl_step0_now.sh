cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
PYSTEP0=~/LLM/models--EleutherAI--pythia-410m-step0
OUT=results/ctrl/pythia410_step0

wait_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$((15360 - used))" -ge "$1" ] && break
    sleep 90
  done
}

wait_free 3500
if [ ! -f "$OUT/a1i_txt/a1.npz" ]; then
  echo "--- a1i pythia step0 $(date)"
  $PY -m chaos_llm.run_a1 --model "$PYSTEP0" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1" --layers "0,6,12,18,23" \
    --out "$OUT/a1i_txt" || echo "FAILED a1i py step0"
fi
RADS=$($PY scripts/pick_radii.py "$OUT/a1i_txt/a1.npz")
echo "--- step0 flip radii: $RADS"
for R in $RADS; do
  RT=$(echo "$R" | tr -d '.')
  if [ ! -f "$OUT/fr_r${RT}/free_run.npz" ]; then
    wait_free 3500
    echo "--- fr step0 r=$R $(date)"
    $PY -m chaos_llm.run_thresholds --model "$PYSTEP0" --run "$OUT/a1i_txt" \
      --radii "$R" --chunk 6 --out "$OUT/fr_r${RT}" || echo "FAILED fr py step0 $R"
  fi
done
echo "step0 extra done $(date)"
