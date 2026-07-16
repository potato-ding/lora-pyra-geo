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
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_P1="${MASTER_PORT_P1:-29501}"
MASTER_PORT_P2="${MASTER_PORT_P2:-29502}"
MASTER_PORT_P3="${MASTER_PORT_P3:-29503}"

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

check_master_port() {
  local probe_name="$1" master_addr="$2" master_port="$3"
  [[ "$DRY_RUN" == "1" ]] && return 0
  if ! [[ "$master_port" =~ ^[0-9]+$ ]] || (( master_port < 1 || master_port > 65535 )); then
    echo "[port-preflight][ERROR] probe=$probe_name master_addr=$master_addr master_port=$master_port status=invalid" >&2
    return 1
  fi
  if ! "$PYTHON_BIN" -c '
import socket, sys
address, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((address, port))
except OSError as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
' "$master_addr" "$master_port"; then
    echo "[port-preflight][ERROR] probe=$probe_name master_addr=$master_addr master_port=$master_port status=occupied_or_unavailable" >&2
    return 1
  fi
  echo "[port-preflight] probe=$probe_name master_addr=$master_addr master_port=$master_port status=available"
}

print_distributed_launch() {
  local probe_name="$1" master_addr="$2" master_port="$3" world_size="$4"
  echo "[distributed-launch] probe=$probe_name"
  echo "[distributed-launch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[distributed-launch] master_addr=$master_addr"
  echo "[distributed-launch] master_port=$master_port"
  echo "[distributed-launch] world_size=$world_size"
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
