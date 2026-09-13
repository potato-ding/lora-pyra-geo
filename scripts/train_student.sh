#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_student.sh CONFIG" >&2
  exit 2
fi
exec python -u -m src.student.launch --config "$1"
