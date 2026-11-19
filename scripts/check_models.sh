#!/bin/bash
for m in Qwen/Qwen3.5-2B Qwen/Qwen3-4B-Instruct-2507 microsoft/Phi-4-mini-instruct; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://huggingface.co/api/models/$m")
  echo "$m -> $code"
done
echo "--- disk"
df -h /home | tail -1
echo "--- hf cli"
which huggingface-cli || python3 -c "import huggingface_hub; print('hub', huggingface_hub.__version__)"
