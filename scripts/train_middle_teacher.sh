#!/usr/bin/env bash
# Train a formal Middle Teacher experiment from one of the eight method configs.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
USAGE="Usage: CUDA_VISIBLE_DEVICES=0,1 scripts/train_middle_teacher.sh configs/middle_teacher/<method>_<size>.json [--check]"

if (( $# < 1 || $# > 2 )); then
  echo "$USAGE" >&2
  exit 2
fi

CONFIG="$1"
CHECK_ONLY=false
if (( $# == 2 )); then
  if [[ "$2" != "--check" ]]; then
    echo "$USAGE" >&2
    exit 2
  fi
  CHECK_ONLY=true
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Activate the pyra_geo environment first." >&2; exit 2; }

# The config selects the method, resolution, batch protocol and Teacher asset.
# --check validates the eight templates before their asset paths are filled.
mapfile -t RUN_FIELDS < <("$PYTHON_BIN" -m src.middle_teacher.formal_config "$CONFIG")

if (( ${#RUN_FIELDS[@]} != 4 )); then
  echo "Middle config validation failed." >&2
  exit 2
fi

EXPERIMENT="${RUN_FIELDS[0]}"
CONFIG="${RUN_FIELDS[1]}"
OUTPUT_DIR="${RUN_FIELDS[2]}"
TEACHER_CHECKPOINT="${RUN_FIELDS[3]}"
echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG"
echo "GPUs: 2; local pair batch: 16; global pair batch: 32"

if "$CHECK_ONLY"; then
  echo "output_dir: ${OUTPUT_DIR:-PENDING}"
  echo "teacher_checkpoint: ${TEACHER_CHECKPOINT:-PENDING_OR_NOT_APPLICABLE}"
  exit 0
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to two GPU IDs." >&2
  exit 2
fi
IFS=, read -r GPU_A GPU_B EXTRA <<< "$CUDA_VISIBLE_DEVICES"
if [[ -z "${GPU_A:-}" || -z "${GPU_B:-}" || -n "${EXTRA:-}" ||
      ! "$GPU_A" =~ ^[0-9]+$ || ! "$GPU_B" =~ ^[0-9]+$ || "$GPU_A" == "$GPU_B" ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly two distinct GPU IDs." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Set output_dir in $CONFIG before training." >&2
  exit 2
fi
if [[ "$EXPERIMENT" != M0-INFONCE-* && -z "$TEACHER_CHECKPOINT" ]]; then
  echo "Set teacher_checkpoint in $CONFIG before KD training." >&2
  exit 2
fi
if [[ -n "$TEACHER_CHECKPOINT" && ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "Teacher checkpoint does not exist: $TEACHER_CHECKPOINT" >&2
  exit 2
fi

if [[ -d "$OUTPUT_DIR" && -n "$(ls -A "$OUTPUT_DIR")" ]]; then
  echo "Refusing a nonempty output directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
export MIDDLE_TRAIN_LOG="$OUTPUT_DIR/train.log"
"$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m src.middle_teacher.formal_train --config "$CONFIG" \
  --expected-gpus "$CUDA_VISIBLE_DEVICES" 2>&1 | tee "$MIDDLE_TRAIN_LOG"
