#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -ne 1 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/train_student.sh CONFIG" >&2
  exit 2
fi
exec python -u -m src.student.launch --config "$1"
