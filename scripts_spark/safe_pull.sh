#!/bin/bash
# Safe pull for the spark clone: remove untracked files that git would
# overwrite (they are identical run-metadata copies), then merge.
cd ~/LLM/chaos_llm_2026
git diff --name-only > /dev/null 2>&1 || true
UNTRACKED=$(git status --porcelain -uall | grep '^??' | awk '{print $2}')
for f in $UNTRACKED; do
  # only remove if the incoming merge would bring the same path
  if git cat-file -e "origin/spark-bigmodels:$f" 2>/dev/null; then
    rm -f "$f"
  fi
done
git pull -q origin spark-bigmodels && git log --oneline -1
