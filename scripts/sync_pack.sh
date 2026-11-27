#!/bin/bash
# Pack result trees into /tmp/chaos_sync.tar (uncompressed: the npz payload is
# already incompressible and gzip was the bottleneck).
#   bash scripts/sync_pack.sh                 -> light trees only
#   bash scripts/sync_pack.sh heavy           -> also results/longloop*
cd ~/LLM/chaos_llm_2026 || exit 1
LIST=/tmp/chaos_list.txt
: > "$LIST"
add () { for f in $1; do [ -e "$f" ] && printf '%s\n' "$f" >> "$LIST"; done; }
add "results/longloop_prompt"
add "results/longtf"
add "results/long_a1"
add "results/update/attn_*"
add "results/update/topk_*"
if [ "${1:-}" = "heavy" ]; then
  add "results/longloop"
  add "results/longloop_v3"
  add "results/longloop_v3_prompt"
fi
if [ ! -s "$LIST" ]; then echo "nothing to pack"; exit 0; fi
tar cf /tmp/chaos_sync.tar -T "$LIST" 2>/dev/null
ls -l /tmp/chaos_sync.tar
