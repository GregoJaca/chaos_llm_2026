#!/bin/bash
# Sequential model downloads for the spark machine; each skipped if marker exists.
cd ~/LLM
dl() {
  name="$1"
  short=$(basename "$name")
  if [ -f "models/$short/.download_complete" ]; then echo "skip $name"; return; fi
  echo "=== downloading $name ==="
  ~/LLM/venv/bin/hf download "$name" --local-dir ~/LLM/models/"$short" > ~/LLM/dl_"$short".log 2>&1
  if [ $? -eq 0 ]; then
    touch "models/$short/.download_complete"; echo "done $name"
  else
    echo "FAILED $name"
  fi
}
dl deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
dl Qwen/Qwen3.5-9B
dl microsoft/phi-4
dl Qwen/Qwen3-14B
echo ALL_DONE
