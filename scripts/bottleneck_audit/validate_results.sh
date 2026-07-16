#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] would validate all G1/G2/probe artifacts, build reports, and only then create FINAL_MANIFEST.txt and SUCCESS"
  exit 0
fi

SUCCESS_FILE="$OUTPUT_ROOT/SUCCESS"
MANIFEST="$OUTPUT_ROOT/FINAL_MANIFEST.txt"
rm -f "$SUCCESS_FILE" "$MANIFEST"

require_file() {
  [[ -s "$1" ]] || { echo "missing or empty result: $1" >&2; return 1; }
}

for direction in D2S S2D; do
  root="$OUTPUT_ROOT/teacher_advantage/1652/all/$direction"
  require_file "$root/summary.json"
  require_file "$root/representative_queries.csv"
  require_file "$root/run.log"
done
for height in 150m 200m 250m 300m; do
  for direction in D2S S2D; do
    root="$OUTPUT_ROOT/teacher_advantage/SUES-200/$height/$direction"
    require_file "$root/summary.json"
    require_file "$root/representative_queries.csv"
    require_file "$root/run.log"
  done
done
root="$OUTPUT_ROOT/teacher_advantage/GTA-UAV/all/D2S"
require_file "$root/summary.json"
require_file "$root/representative_queries.csv"
require_file "$root/run.log"
require_file "$OUTPUT_ROOT/representation_audit.json"

for probe_dir in P1-f3-linear P2-f4-linear P3-f4-mlp; do
  root="src/checkpoint/student/diagnostic_probes/$probe_dir"
  for name in train.log best_model.pth last_model.pth best_metrics.json \
    student_test_1652_best.json student_test_sues200_best.json \
    student_test_gta_uav_best.json backbone_freeze_audit.json; do
    require_file "$root/$name"
  done
done

run_cmd "$PYTHON_BIN" -m src.diagnostics.build_bottleneck_report \
  --result_root "$OUTPUT_ROOT" \
  --probe_root src/checkpoint/student/diagnostic_probes
require_file "$OUTPUT_ROOT/bottleneck_report.json"
require_file "$OUTPUT_ROOT/bottleneck_report.md"

{
  echo "commit=$(git rev-parse HEAD)"
  echo "validated_at=$(date --iso-8601=seconds)"
  find "$OUTPUT_ROOT" src/checkpoint/student/diagnostic_probes -type f \
    -printf '%p\t%s bytes\n' | sort
} >"$MANIFEST"
require_file "$MANIFEST"
printf 'validated_at=%s\n' "$(date --iso-8601=seconds)" >"$SUCCESS_FILE"
echo "[success] bottleneck audit results validated"
