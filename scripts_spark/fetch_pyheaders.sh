#!/bin/bash
# fetch python3.12 dev headers without sudo
set -e
mkdir -p ~/LLM/pydev && cd ~/LLM/pydev
curl -s http://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/ -o pool.html
echo "--- dev/header debs found:"
grep -oE 'href="[^"]*arm64\.deb' pool.html | grep -E 'dev|headers' | sed 's/href="//;s/"//' | head -20
