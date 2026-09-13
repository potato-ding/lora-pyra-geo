#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/train_student_certified.sh GPU_IDS CONFIG" >&2
  exit 2
fi
cd /home/dingyi/lora-pyra-geo
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u  # Conda activation hooks may read unset optional shell variables.
source /home/dingyi/miniforge3/etc/profile.d/conda.sh
conda activate pyra_geo
set -u
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
exec python -u -m src.student.launch --config "$2"
