#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_student.sh CONFIG" >&2
  exit 2
fi
exec torchrun --standalone --nproc_per_node=2 -m src.student.train --config "$1"
