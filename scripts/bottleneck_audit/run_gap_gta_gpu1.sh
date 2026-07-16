#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_inputs
export CUDA_VISIBLE_DEVICES="${GTA_GPU:-1}"
run_gap_protocol GTA-UAV "$GTA_ROOT" all D2S
