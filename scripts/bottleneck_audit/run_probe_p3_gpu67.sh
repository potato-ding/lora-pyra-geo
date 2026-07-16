#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_inputs
export CUDA_VISIBLE_DEVICES="${P3_GPUS:-6,7}"
run_cmd "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=2 \
  -m src.diagnostics.train_frozen_probe --probe P3 --deepspeed \
  --student_checkpoint "$B0_CKPT" --train_data_dir "$U1652_ROOT/train" --val_data_dir "$U1652_ROOT" \
  --u1652_root "$U1652_ROOT" --sues_root "$SUES_ROOT" --gta_root "$GTA_ROOT" \
  --output_root src/checkpoint/student/diagnostic_probes
