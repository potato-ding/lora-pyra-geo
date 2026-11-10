#!/usr/bin/env bash
# Isolated T0 -> Middle ADSD-only + KD-guided SAM control.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="configs/middle_teacher/m_sam_adsd_256.json"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != *,* ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to two GPUs." >&2
  exit 2
fi
OUTPUT_DIR="$(python -c 'from src.middle_teacher.pure_adsd_config import load_formal_config,CONFIG; print(load_formal_config(CONFIG)[1]["output_dir"])')"
if [[ -d "$OUTPUT_DIR" && -n "$(ls -A "$OUTPUT_DIR")" ]]; then
  echo "Refusing nonempty output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
export MIDDLE_TRAIN_LOG="$OUTPUT_DIR/train.log"
python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m src.middle_teacher.pure_adsd_train --config "$CONFIG" \
  --expected-gpus "$CUDA_VISIBLE_DEVICES" 2>&1 | tee "$MIDDLE_TRAIN_LOG"
