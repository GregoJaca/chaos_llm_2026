#!/bin/bash
set -e
cd ~/LLM
echo "download start $(date)"
huggingface-cli download Qwen/Qwen3.5-2B --local-dir models--Qwen--Qwen3.5-2B
echo "qwen2b done $(date)"
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models--Qwen--Qwen3-4B-Instruct-2507
echo "qwen4b done $(date)"
du -sh models--Qwen--*
echo "DOWNLOAD DONE $(date)"
