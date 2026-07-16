#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_inputs
export CUDA_VISIBLE_DEVICES="${REPRESENTATION_GPU:-1}"
run_cmd "$PYTHON_BIN" -u -m src.diagnostics.student_representation \
  --dataset all --checkpoint "$B0_CKPT" --u1652_root "$U1652_ROOT" \
  --sues_root "$SUES_ROOT" --gta_root "$GTA_ROOT" \
  --output_file "$OUTPUT_ROOT/representation_audit.json"
