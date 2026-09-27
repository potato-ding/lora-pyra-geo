#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:?usage: CUDA_VISIBLE_DEVICES=0,1 train_middle_teacher.sh CONFIG --teacher-checkpoint CHECKPOINT}
shift
cd "$ROOT"
: "${CUDA_VISIBLE_DEVICES:?Specify the two training GPUs}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["output_dir"])' "$CONFIG")
if [[ -d "$OUTPUT" ]] && [[ -n "$(ls -A "$OUTPUT")" ]]; then
  echo "Refusing nonempty output: $OUTPUT" >&2; exit 2
fi
mkdir -p "$OUTPUT"
export FCHAIN_EXTERNAL_TRAIN_LOG="$OUTPUT/train.log"
python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m src.middle_teacher.fchain_train --config "$CONFIG" \
  --expected-gpus "$CUDA_VISIBLE_DEVICES" "$@" 2>&1 | tee "$FCHAIN_EXTERNAL_TRAIN_LOG"
