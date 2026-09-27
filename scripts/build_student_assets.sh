#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACTION=${1:?usage: build_student_assets.sh top|calibration|bind [arguments]}
shift
case "$ACTION" in
  top) exec python -m src.student.build_top_only "$@" ;;
  calibration) exec python -m src.student.build_calibration "$@" ;;
  bind) exec python -m src.student.bind_core_assets "$@" ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
