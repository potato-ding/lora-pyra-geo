#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

B0_CKPT="${B0_CKPT:-src/checkpoint/student/B0-2GPU-3090/best_model.pth}"
T0_CKPT="${T0_CKPT:-src/checkpoint/teacher/T0-3090/best_model.pth}"
U1652_ROOT="${U1652_ROOT:-data/U1652}"
SUES_ROOT="${SUES_ROOT:-data/SUES-200/SUES-200-512x512}"
GTA_ROOT="${GTA_ROOT:-data/GTA-UAV-LR/GTA-UAV-LR-baidu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-src/diagnostics/results/bottleneck_audit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

require_inputs() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  [[ -s "$B0_CKPT" ]] || { echo "missing B0 checkpoint: $B0_CKPT" >&2; return 1; }
  [[ -s "$T0_CKPT" ]] || { echo "missing T0 checkpoint: $T0_CKPT" >&2; return 1; }
  [[ -d "$U1652_ROOT" ]] || { echo "missing U1652: $U1652_ROOT" >&2; return 1; }
  [[ -d "$SUES_ROOT" ]] || { echo "missing SUES-200: $SUES_ROOT" >&2; return 1; }
  [[ -d "$GTA_ROOT" ]] || { echo "missing GTA-UAV: $GTA_ROOT" >&2; return 1; }
}

run_gap_protocol() {
  local dataset="$1" data_dir="$2" height="$3" direction="$4"
  local target="$OUTPUT_ROOT/teacher_advantage/$dataset/$height/$direction"
  [[ "$DRY_RUN" == "1" ]] || mkdir -p "$target"
  local -a cmd=("$PYTHON_BIN" -u -m src.diagnostics.teacher_advantage
    --dataset "$dataset" --data_dir "$data_dir" --direction "$direction"
    --student_ckpt "$B0_CKPT" --teacher_ckpt "$T0_CKPT"
    --output_dir "$OUTPUT_ROOT/teacher_advantage")
  [[ "$dataset" == "SUES-200" ]] && cmd+=(--sues_height "${height%m}")
  if [[ "$DRY_RUN" == "1" ]]; then
    run_cmd "${cmd[@]}"
  else
    run_cmd "${cmd[@]}" 2>&1 | tee "$target/run.log"
  fi
}
