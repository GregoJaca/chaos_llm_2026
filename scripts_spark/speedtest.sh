#!/bin/bash
url=$(~/LLM/venv/bin/pip download --no-deps -d /tmp/x "numpy==2.5.0" 2>/dev/null | grep -o 'https://[^ ]*\.whl' | head -1)
echo "url: $url"
curl -sS -o /tmp/speedtest.bin -w "pypi speed: %{speed_download} B/s  size %{size_download}\n" --max-time 30 "$url"
rm -f /tmp/speedtest.bin
