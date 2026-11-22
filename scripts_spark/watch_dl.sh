#!/bin/bash
# Wait for the initial DS-14B download to finish, mark it complete, then run the chain.
while pgrep -f "hf download deepseek-ai" > /dev/null; do sleep 30; done
touch ~/LLM/models/DeepSeek-R1-Distill-Qwen-14B/.download_complete 2>/dev/null
bash ~/LLM/dl_chain.sh
