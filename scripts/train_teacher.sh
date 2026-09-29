#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root so paths inside the JSON work as written.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:?Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_teacher.sh configs/teacher/t0_224.json}"
shift
: "${CUDA_VISIBLE_DEVICES:?Set the four physical training GPU IDs}"

# Both resolutions use four GPUs, eight drone/satellite pairs per GPU.
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Teacher training requires exactly four visible GPUs" >&2
  exit 2
fi

# Validate the method/config before starting four Python workers.
python -m src.training.teacher.formal_config --config "$CONFIG" "$@"

exec python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m src.training.teacher.train --config "$CONFIG" "$@"
