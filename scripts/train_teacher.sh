#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG=${1:?usage: CUDA_VISIBLE_DEVICES=... train_teacher.sh CONFIG [extra args]}
shift
: "${CUDA_VISIBLE_DEVICES:?Specify the training GPUs}"
NPROC=$(python -c 'import json,sys; b=json.load(open(sys.argv[1]))["batch_size"]; assert 32 % b == 0; print(32//b)' "$CONFIG")
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#GPU_IDS[@]} -ne $NPROC ]]; then
  echo "Configured local batch requires $NPROC GPUs for 32 global pairs" >&2; exit 2
fi
exec python -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m src.training.teacher.train --config "$CONFIG" "$@"
