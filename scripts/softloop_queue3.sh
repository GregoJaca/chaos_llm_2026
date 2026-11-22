#!/bin/bash
# Chain 3: greedy-orbit gate strengths from the existing A1 corpora.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B

while pgrep -f "bash scripts/softloop_queue2.sh" > /dev/null; do sleep 60; done
echo "GATES QUEUE START $(date)"

for A in results/a1_ds_w100_layers/a1.npz results/a1_cyber_w100/a1.npz results/a1_childhood_w100/a1.npz; do
  NAME=$(echo $A | sed 's#results/##; s#/a1.npz##')
  D=results/softloop_gates/$NAME
  if [ ! -f $D/gates.npz ]; then
    echo "=== gates $NAME $(date)"
    python3 -m chaos_llm.run_gates --model $M --a1 $A --out $D || echo "FAILED $A"
  fi
done

# loop-orbit gates: from the softloop runs themselves (ref row), for T where
# we saved top1 only -> use the recorded sig_read instead (no job needed).
echo "GATES QUEUE DONE $(date)"
