#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:?usage: train_middle_teacher.sh CONFIG_JSON [extra args...]}
shift
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Preserve the active environment and its CUDA setup; use package-relative imports.
exec deepspeed --no_local_rank --module src.middle_teacher.train --config "$CONFIG" "$@"
