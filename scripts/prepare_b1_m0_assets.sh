#!/usr/bin/env bash
# Explicit B1 prerequisite. Run only after a GPU is assigned; never auto-start training.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
  echo "Assign exactly one free physical GPU via CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
MIDDLE="$ROOT/src/checkpoint/middle_teacher/R256/M0-256/best_model.pth"
CONFIG="$ROOT/configs/middle_teacher/m0_infonce_256.json"
ASSETS="$ROOT/src/checkpoint/student/R256/middle_assets/M0-256"
# Existing outputs are verified, never overwritten. A partially corrupt asset fails closed.
if [[ ! -e "$ASSETS/top128" ]]; then
  python -m src.student.build_top_only --allow-m0-ablation     --middle-checkpoint "$MIDDLE" --middle-config "$CONFIG"     --image-size 256 --output "$ASSETS/top128" --device cuda:0
fi
if [[ ! -e "$ASSETS/top_calibration.pt" && ! -e "$ASSETS/top_calibration.pt.json" ]]; then
  python -m src.student.build_calibration --allow-m0-ablation     --middle-checkpoint "$MIDDLE" --middle-run-config "$CONFIG"     --student-pretrained "$ROOT/src/models/repvit/repvit_m1_5_distill_450e.pth"     --asset-output "$ASSETS/top_calibration.pt" --train-root "$ROOT/data/U1652/train"     --seed 0 --image-size 256 --device cuda:0
fi
python -c 'from src.student.formal_config import load_formal_config; from src.student.formal_train import validate_assets; _,c=load_formal_config("configs/student/b1_m0_provider_sam_m0_256.json"); validate_assets(c); print("B1_M0_ASSETS_READY; TRAINING_NOT_STARTED")'
