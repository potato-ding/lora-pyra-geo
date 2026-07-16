#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_inputs

if [[ "$DRY_RUN" == "1" ]]; then
  for script in \
    run_gap_u1652_sues_gpu0.sh run_gap_gta_gpu1.sh run_representation_gpu1.sh \
    run_probe_p1_gpu23.sh run_probe_p2_gpu45.sh run_probe_p3_gpu67.sh; do
    DRY_RUN=1 bash "scripts/bottleneck_audit/$script"
  done
  echo "[dry-run] commands printed; validation and SUCCESS creation skipped"
  exit 0
fi

LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"
rm -f "$OUTPUT_ROOT/SUCCESS" "$OUTPUT_ROOT/FINAL_MANIFEST.txt"
declare -A PIDS=()

launch() {
  local name="$1" log="$2"
  shift 2
  ("$@") >"$log" 2>&1 &
  PIDS["$name"]=$!
  echo "[launch] task=$name pid=${PIDS[$name]} log=$log"
}

launch gap_u1652_sues "$LOG_ROOT/gap_u1652_sues.log" \
  bash scripts/bottleneck_audit/run_gap_u1652_sues_gpu0.sh
launch gap_gta "$LOG_ROOT/gap_gta.log" bash scripts/bottleneck_audit/run_gap_gta_gpu1.sh
launch probe_p1 "$LOG_ROOT/probe_p1.log" bash scripts/bottleneck_audit/run_probe_p1_gpu23.sh
launch probe_p2 "$LOG_ROOT/probe_p2.log" bash scripts/bottleneck_audit/run_probe_p2_gpu45.sh
launch probe_p3 "$LOG_ROOT/probe_p3.log" bash scripts/bottleneck_audit/run_probe_p3_gpu67.sh

failed=0
if wait "${PIDS[gap_gta]}"; then
  echo "[complete] task=gap_gta"
  launch representation "$LOG_ROOT/representation.log" \
    bash scripts/bottleneck_audit/run_representation_gpu1.sh
else
  echo "[failure] task=gap_gta pid=${PIDS[gap_gta]}" >&2
  failed=1
fi

for task in gap_u1652_sues probe_p1 probe_p2 probe_p3; do
  if wait "${PIDS[$task]}"; then
    echo "[complete] task=$task"
  else
    echo "[failure] task=$task pid=${PIDS[$task]}" >&2
    failed=1
  fi
done
if [[ -n "${PIDS[representation]:-}" ]]; then
  if wait "${PIDS[representation]}"; then
    echo "[complete] task=representation"
  else
    echo "[failure] task=representation pid=${PIDS[representation]}" >&2
    failed=1
  fi
fi
(( failed == 0 )) || exit 1

bash scripts/bottleneck_audit/validate_results.sh
