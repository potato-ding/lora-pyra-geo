#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_inputs
export CUDA_VISIBLE_DEVICES="${P2_GPUS:-4,5}"
MASTER_PORT="$MASTER_PORT_P2"
check_master_port P2 "$MASTER_ADDR" "$MASTER_PORT"
print_distributed_launch P2 "$MASTER_ADDR" "$MASTER_PORT" 2
run_cmd "$PYTHON_BIN" -m torch.distributed.run \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" --nproc_per_node=2 \
  -m src.diagnostics.train_frozen_probe --probe P2 --deepspeed \
  --student_checkpoint "$B0_CKPT" --train_data_dir "$U1652_ROOT/train" --val_data_dir "$U1652_ROOT" \
  --u1652_root "$U1652_ROOT" --sues_root "$SUES_ROOT" --gta_root "$GTA_ROOT" \
  --output_root src/checkpoint/student/diagnostic_probes
