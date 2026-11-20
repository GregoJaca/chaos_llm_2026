#!/bin/bash
# verify the exact gcc invocation triton needs, with our extracted headers
INC=/home/grego/LLM/pydev/extracted/usr/include
mkdir -p /tmp/tri_test
export C_INCLUDE_PATH="$INC/python3.12:$INC"
/usr/bin/gcc /home/grego/LLM/venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.c \
  -O3 -shared -fPIC -Wno-psabi -o /tmp/tri_test/cuda_utils.cpython-312-aarch64-linux-gnu.so \
  -l:libcuda.so.1 \
  -L/home/grego/LLM/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib \
  -L/lib/aarch64-linux-gnu \
  -I/home/grego/LLM/venv/lib/python3.12/site-packages/triton/backends/nvidia/include \
  -I/tmp/tri_test -I/usr/include/python3.12 2>&1 | head -12
echo "gcc exit: ${PIPESTATUS[0]}"
ls -la /tmp/tri_test/ | tail -1
