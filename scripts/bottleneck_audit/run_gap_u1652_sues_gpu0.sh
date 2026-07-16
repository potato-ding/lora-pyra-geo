#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_inputs
export CUDA_VISIBLE_DEVICES="${GAP_GPU:-0}"
run_gap_protocol 1652 "$U1652_ROOT" all D2S
run_gap_protocol 1652 "$U1652_ROOT" all S2D
for height in 150m 200m 250m 300m; do
  run_gap_protocol SUES-200 "$SUES_ROOT" "$height" D2S
  run_gap_protocol SUES-200 "$SUES_ROOT" "$height" S2D
done
