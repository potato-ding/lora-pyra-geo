#!/usr/bin/env bash
# Build Top128 and its TRAIN-only Student initialization calibration.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
USAGE="Usage: scripts/build_student_assets.sh top|calibration [options]"

if (( $# < 1 )); then
  echo "$USAGE" >&2
  exit 2
fi

ACTION="$1"
shift
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "$ACTION" in
  top)
    exec "$PYTHON_BIN" -m src.student.build_top_only "$@"
    ;;
  calibration)
    exec "$PYTHON_BIN" -m src.student.build_calibration "$@"
    ;;
  *)
    echo "$USAGE" >&2
    exit 2
    ;;
esac
