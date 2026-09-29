#!/usr/bin/env bash
# Train S0, TSD, ADSD or SAM-ADSD at 224/256 from a public config.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
USAGE="Usage: CUDA_VISIBLE_DEVICES=<gpu> scripts/train_student.sh configs/student/<method>_<size>.json [--check]"

if (( $# < 1 || $# > 2 )) || { (( $# == 2 )) && [[ "$2" != "--check" ]]; }; then
  echo "$USAGE" >&2
  exit 2
fi

CONFIG="$1"
CHECK_ONLY=false
if (( $# == 2 )); then CHECK_ONLY=true; fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "Activate the pyra_geo environment first." >&2
  exit 2
}

mapfile -t FIELDS < <("$PYTHON_BIN" -m src.student.formal_config "$CONFIG")
if (( ${#FIELDS[@]} != 7 )); then
  echo "Student config validation failed." >&2
  exit 2
fi
EXPERIMENT="${FIELDS[0]}"
CONFIG="${FIELDS[1]}"
OUTPUT_DIR="${FIELDS[2]}"
MIDDLE_CHECKPOINT="${FIELDS[3]}"
MIDDLE_CONFIG="${FIELDS[4]}"
SUPERVISION_ASSET="${FIELDS[5]}"
TOP_CALIBRATION="${FIELDS[6]}"

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG"
echo "GPUs: 1; local/global pair batch: 32/32"
echo "Selection: U1652 D2S/S2D R1 sum, epoch 11 onward, one GPU, batch 16"

if "$CHECK_ONLY"; then
  echo "output_dir: ${OUTPUT_DIR:-PENDING}"
  echo "middle_checkpoint: ${MIDDLE_CHECKPOINT:-PENDING_OR_NOT_APPLICABLE}"
  echo "middle_config: ${MIDDLE_CONFIG:-PENDING_OR_NOT_APPLICABLE}"
  echo "supervision_asset: ${SUPERVISION_ASSET:-PENDING_OR_NOT_APPLICABLE}"
  echo "top_calibration: ${TOP_CALIBRATION:-PENDING_OR_NOT_APPLICABLE}"
  exit 0
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to exactly one GPU ID." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Set output_dir in $CONFIG before training." >&2
  exit 2
fi
if [[ "$EXPERIMENT" != S0-INFONCE-* &&
      ( -z "$MIDDLE_CHECKPOINT" || -z "$MIDDLE_CONFIG" ||
        -z "$SUPERVISION_ASSET" || -z "$TOP_CALIBRATION" ) ]]; then
  echo "Set Middle and supervision paths in $CONFIG before distillation." >&2
  exit 2
fi
if [[ -d "$OUTPUT_DIR" && -n "$(ls -A "$OUTPUT_DIR")" ]]; then
  echo "Refusing a nonempty output directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
export STUDENT_TRAIN_LOG="$OUTPUT_DIR/train.log"
"$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m src.student.formal_train --config "$CONFIG" 2>&1 | tee "$OUTPUT_DIR/train.log"
