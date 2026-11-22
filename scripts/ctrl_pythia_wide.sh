cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
PY410=~/LLM/models--EleutherAI--pythia-410m
OUT=results/ctrl/pythia410

wait_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$((15360 - used))" -ge "$1" ] && break
    sleep 90
  done
}

# wide-radius A1 including the flip zone, measured at the target radius
wait_free 3500
if [ ! -f "$OUT/a1i_txt_wide/a1.npz" ]; then
  echo "--- a1i pythia wide $(date)"
  $PY -m chaos_llm.run_a1 --model "$PY410" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "0.05,0.1,0.19,0.3,0.5,0.7,1,1.4,2,3" --layers "0,6,12,18,23" \
    --out "$OUT/a1i_txt_wide" || echo "FAILED a1i py wide"
fi

for R in 0.19 0.3 0.7 1.4 3; do
  RT=$(echo "$R" | tr -d '.')
  if [ ! -f "$OUT/frw_r${RT}/free_run.npz" ]; then
    wait_free 3500
    echo "--- fr pythia wide r=$R $(date)"
    $PY -m chaos_llm.run_thresholds --model "$PY410" --run "$OUT/a1i_txt_wide" \
      --radii "$R" --chunk 6 --out "$OUT/frw_r${RT}" || echo "FAILED fr py wide $R"
  fi
done

# teacher-forced vs free-run exactness for mamba at flip-zone radii (stage2 leftovers)
M130=~/LLM/models--state-spaces--mamba-130m-hf
for r in 2 3 5; do
  if [ ! -f results/ctrl/mamba130/tfp_r${r}/tf_predict.npz ]; then
    wait_free 3000
    echo "--- tfp mamba r=$r $(date)"
    $PY -m chaos_llm.run_tf_predict --model "$M130" --run results/ctrl/mamba130/a1i_txt \
      --free-run results/ctrl/mamba130/fr_txt_r${r} --radii "$r" --chunk 6 \
      --out results/ctrl/mamba130/tfp_r${r} || echo "FAILED tfp $r"
  fi
done
echo "pythia-wide + tfp done $(date)"
