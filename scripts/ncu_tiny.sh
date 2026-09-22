#!/usr/bin/env bash
# Profile the TinyCoder decode kernels with ncu (requires root for GPU counters).
# Usage:  sudo bash scripts/ncu_tiny.sh
# Output: /tmp/ncu_tiny.txt
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

# Run detached so an interactive terminal does not need to stay open:
#   sudo bash -c 'nohup bash scripts/ncu_tiny.sh > /tmp/ncu_tiny.log 2>&1 &'

bash scripts/profile_ncu.sh \
    /data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf \
    /tmp/ncu_tiny.txt
