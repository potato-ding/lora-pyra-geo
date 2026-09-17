#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/train_student_split16.sh CONFIG" >&2
  exit 2
fi
cd /home/dingyi/lora-pyra-geo
export CUDA_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
exec /home/dingyi/miniforge3/envs/pyra_geo/bin/python -u -m src.student.launch_split16 --config "$1"
