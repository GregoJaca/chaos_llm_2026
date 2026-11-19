#!/bin/bash
cd ~/LLM
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models--Qwen--Qwen2.5-3B-Instruct
echo "Qwen2.5-3B done $(date)"
du -sh models--Qwen--Qwen2.5-3B-Instruct
