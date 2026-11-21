#!/bin/bash
# Control study, stage 2: full Mamba suite (in-batch reference protocol) + Pythia
# (attention, same Pile training data) + long-horizon decay.
# Resume-safe. Run after stage 1.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
M130=~/LLM/models--state-spaces--mamba-130m-hf
M370=~/LLM/models--state-spaces--mamba-370m-hf
M790=~/LLM/models--state-spaces--mamba-790m-hf
M14=~/LLM/models--state-spaces--mamba-1.4b-hf
PY410=~/LLM/models--EleutherAI--pythia-410m
PY1B=~/LLM/models--EleutherAI--pythia-1b
mkdir -p results/ctrl

echo "=== stage2 start $(date)"

RAD12="3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1,1.4,2,3,5"

# ---------- Mamba-130m, in-batch A1, 3 prompts ----------
for PAIR in "prompts.txt:txt" "prompts_childhood.txt:childhood" "prompts_cyber.txt:cyber"; do
  P="${PAIR%%:*}"; TAG="${PAIR##*:}"
  if [ ! -f results/ctrl/mamba130/a1i_$TAG/a1.npz ]; then
    echo "--- a1i mamba130 $TAG $(date)"
    $PY -m chaos_llm.run_a1 --model "$M130" --prompt @"$P" --window 4096 \
      --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
      --radii "$RAD12" --layers "0,6,12,18,23" \
      --out results/ctrl/mamba130/a1i_$TAG || echo "FAILED a1i130 $TAG"
  fi
done

# ---------- Mamba-130m long-horizon teacher-forced decay (1000 tokens) ----------
if [ -f results/ctrl/mamba130/traj/ocean_currents.npz ] && [ ! -f results/ctrl/mamba130/a1i_txt_T1000/a1.npz ]; then
  echo "--- a1i mamba130 T1000 $(date)"
  $PY -m chaos_llm.run_a1 --model "$M130" --prompt @prompts_long2.txt --window 4096 \
    --max-new-tokens 1000 --fixed-tokens-from results/ctrl/mamba130/traj/ocean_currents.npz \
    --num-directions 8 --batch-dirs 4 --inbatch-ref \
    --radii "1e-3,0.1,1" --layers "0,6,12,18,23" \
    --out results/ctrl/mamba130/a1i_txt_T1000 || echo "FAILED a1i_T1000"
fi

# ---------- Mamba-130m free runs in the flip zone ----------
for TAG in txt childhood cyber; do
  for r in 1.4 2 3 5; do
    if [ ! -f results/ctrl/mamba130/fr_${TAG}_r${r}/free_run.npz ]; then
      echo "--- fr mamba130 $TAG r=$r $(date)"
      $PY -m chaos_llm.run_thresholds --model "$M130" --run results/ctrl/mamba130/a1i_$TAG \
        --radii "$r" --chunk 6 --out results/ctrl/mamba130/fr_${TAG}_r${r} || echo "FAILED fr130 $TAG $r"
    fi
  done
done

# ---------- teacher-forced vs free-run exactness, flip-zone radii ----------
for r in 2 3 5; do
  if [ ! -f results/ctrl/mamba130/tfp_r${r}/tf_predict.npz ]; then
    echo "--- tfp mamba130 r=$r $(date)"
    $PY -m chaos_llm.run_tf_predict --model "$M130" --run results/ctrl/mamba130/a1i_txt \
      --free-run results/ctrl/mamba130/fr_txt_r${r} --radii "$r" --chunk 6 \
      --out results/ctrl/mamba130/tfp_r${r} || echo "FAILED tfp $r"
  fi
done

# ---------- Mamba-370m ----------
RAD8="1e-3,1e-2,0.1,0.3,1,1.4,3,5"
if [ ! -f results/ctrl/mamba370/a1i_txt/a1.npz ]; then
  echo "--- a1i mamba370 txt $(date)"
  $PY -m chaos_llm.run_a1 --model "$M370" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "$RAD8" --layers "0,12,24,35,47" \
    --out results/ctrl/mamba370/a1i_txt || echo "FAILED a1i370"
fi
if [ ! -f results/ctrl/mamba370/a1i_childhood/a1.npz ]; then
  echo "--- a1i mamba370 childhood $(date)"
  $PY -m chaos_llm.run_a1 --model "$M370" --prompt @prompts_childhood.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "$RAD8" --layers "0,12,24,35,47" \
    --out results/ctrl/mamba370/a1i_childhood || echo "FAILED a1i370 childhood"
fi
for r in 1.4 3 5; do
  if [ ! -f results/ctrl/mamba370/fr_txt_r${r}/free_run.npz ]; then
    echo "--- fr mamba370 txt r=$r $(date)"
    $PY -m chaos_llm.run_thresholds --model "$M370" --run results/ctrl/mamba370/a1i_txt \
      --radii "$r" --chunk 6 --out results/ctrl/mamba370/fr_txt_r${r} || echo "FAILED fr370 $r"
  fi
done
if [ ! -f results/ctrl/mamba370/traj/ocean_currents.npz ]; then
  echo "--- traj mamba370 $(date)"
  sed -n "1,4p" prompts_long2.txt > /tmp/p4.txt
  $PY -m chaos_llm.collect_trajectories --model "$M370" --prompts /tmp/p4.txt \
    --window 4096 --max-new-tokens 1000 --layers "12,24,35,47" \
    --out results/ctrl/mamba370/traj_tmp && mkdir -p results/ctrl/mamba370/traj \
    && mv results/ctrl/mamba370/traj_tmp/*.npz results/ctrl/mamba370/traj/ || echo "FAILED traj370"
fi

# ---------- Mamba-790m ----------
if [ ! -f results/ctrl/mamba790/a1i_txt/a1.npz ]; then
  echo "--- a1i mamba790 txt $(date)"
  $PY -m chaos_llm.run_a1 --model "$M790" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "$RAD8" --layers "0,12,24,35,47" \
    --out results/ctrl/mamba790/a1i_txt || echo "FAILED a1i790"
fi
for r in 1.4 3 5; do
  if [ ! -f results/ctrl/mamba790/fr_txt_r${r}/free_run.npz ]; then
    echo "--- fr mamba790 txt r=$r $(date)"
    $PY -m chaos_llm.run_thresholds --model "$M790" --run results/ctrl/mamba790/a1i_txt \
      --radii "$r" --chunk 6 --out results/ctrl/mamba790/fr_txt_r${r} || echo "FAILED fr790 $r"
  fi
done
if [ ! -f results/ctrl/mamba790/traj/ocean_currents.npz ]; then
  echo "--- traj mamba790 $(date)"
  sed -n "1,4p" prompts_long2.txt > /tmp/p4.txt
  $PY -m chaos_llm.collect_trajectories --model "$M790" --prompts /tmp/p4.txt \
    --window 4096 --max-new-tokens 1000 --layers "12,24,35,47" \
    --out results/ctrl/mamba790/traj_tmp && mkdir -p results/ctrl/mamba790/traj \
    && mv results/ctrl/mamba790/traj_tmp/*.npz results/ctrl/mamba790/traj/ || echo "FAILED traj790"
fi

# ---------- Mamba-1.4b ----------
if [ ! -f results/ctrl/mamba14/a1i_txt/a1.npz ]; then
  echo "--- a1i mamba1.4b txt $(date)"
  $PY -m chaos_llm.run_a1 --model "$M14" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 5 --inbatch-ref \
    --radii "$RAD8" --layers "0,12,24,35,47" \
    --out results/ctrl/mamba14/a1i_txt || echo "FAILED a1i14"
fi
for r in 3 5; do
  if [ ! -f results/ctrl/mamba14/fr_txt_r${r}/free_run.npz ]; then
    echo "--- fr mamba1.4b txt r=$r $(date)"
    $PY -m chaos_llm.run_thresholds --model "$M14" --run results/ctrl/mamba14/a1i_txt \
      --radii "$r" --chunk 4 --out results/ctrl/mamba14/fr_txt_r${r} || echo "FAILED fr14 $r"
  fi
done

# ---------- Pythia-410m (attention, Pile) ----------
if [ ! -f results/ctrl/pythia410/a1i_txt/a1.npz ]; then
  echo "--- a1i pythia410 txt $(date)"
  $PY -m chaos_llm.run_a1 --model "$PY410" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "$RAD8" --layers "0,6,12,18,23" \
    --out results/ctrl/pythia410/a1i_txt || echo "FAILED a1i py410"
fi
for r in 0.3 1.4 3; do
  if [ ! -f results/ctrl/pythia410/fr_txt_r${r}/free_run.npz ]; then
    echo "--- fr pythia410 txt r=$r $(date)"
    $PY -m chaos_llm.run_thresholds --model "$PY410" --run results/ctrl/pythia410/a1i_txt \
      --radii "$r" --chunk 6 --out results/ctrl/pythia410/fr_txt_r${r} || echo "FAILED fr py410 $r"
  fi
done
if [ ! -f results/ctrl/pythia410/traj/ocean_currents.npz ]; then
  echo "--- traj pythia410 $(date)"
  sed -n "1,4p" prompts_long2.txt > /tmp/p4.txt
  $PY -m chaos_llm.collect_trajectories --model "$PY410" --prompts /tmp/p4.txt \
    --window 4096 --max-new-tokens 1000 --layers "6,12,18,23" \
    --out results/ctrl/pythia410/traj_tmp && mkdir -p results/ctrl/pythia410/traj \
    && mv results/ctrl/pythia410/traj_tmp/*.npz results/ctrl/pythia410/traj/ || echo "FAILED traj py410"
fi

# ---------- Pythia-1b ----------
if [ ! -f results/ctrl/pythia1b/a1i_txt/a1.npz ]; then
  echo "--- a1i pythia1b txt $(date)"
  $PY -m chaos_llm.run_a1 --model "$PY1B" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "$RAD8" --layers "0,4,8,12,15" \
    --out results/ctrl/pythia1b/a1i_txt || echo "FAILED a1i py1b"
fi

echo "=== stage2 done $(date)"
