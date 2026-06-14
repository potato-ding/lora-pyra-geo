# Evaluation Protocol Audit

This note records how the current evaluation code handles University-1652, SUES-200, and GTA-UAV for teacher and student models.

## Summary

- Teacher and student evaluation now use the same dataset builders and metric functions.
- Teacher and student evaluation are single-card by default.
- `GTA-UAV` defaults to bidirectional evaluation through `--gta_query_mode both`.
- `SUES-200` evaluates all four heights by default through `--sues_height all`.
- `SUES-200` horizontal-flip test-time augmentation is disabled by default. It can be enabled explicitly with `--sues_horizontal_flip`.

## Local Dataset Roots

```text
data/U1652
data/SUES-200/SUES-200-512x512
data/GTA-UAV-LR/GTA-UAV-LR-baidu
```

The local directory structure matches the current code assumptions:

- SUES-200: `Testing/150`, `Testing/200`, `Testing/250`, `Testing/300`.
- Each SUES height contains `query_drone`, `gallery_satellite`, `query_satellite`, and `gallery_drone`.
- GTA-UAV contains `satellite`, `drone`, `cross-area-drone2sate-test.json`, and `same-area-drone2sate-test.json`.

## SUES-200

Code path:

- Dataloader: `src.dataset.teacher.val_dataloaders.build_sues200_val_dataloaders`
- Metrics: `src.utils.train_eval_utils.run_sues_val_and_get_metrics`
- Teacher entry: `src/training/teacher_test.py`
- Student entry: `src/training/student_test.py`

Implemented logic:

- Evaluate four heights: `150`, `200`, `250`, `300`.
- Evaluate both directions:
  - `D2S`: drone query to satellite gallery.
  - `S2D`: satellite query to drone gallery.
- Metrics:
  - `R@1`
  - `R@5`
  - `R@10`
  - `R@top1`
  - `AP`
- `R@top1` uses `ceil(0.01 * gallery_size)`.
- AP is computed over the ranked gallery list.
- No horizontal-flip TTA by default.

Important command:

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

Student:

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run>/best_model.pth \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

## GTA-UAV

Code path:

- Dataloader: `src.dataset.teacher.val_dataloaders.build_gta_val_dataloaders`
- Metrics: `src.utils.train_eval_utils.run_gta_val_and_get_metrics`
- Teacher entry: `src/training/teacher_test.py`
- Student entry: `src/training/student_test.py`

Implemented logic:

- Split controlled by `--gta_split`:
  - `cross-area`
  - `same-area`
- Direction controlled by `--gta_query_mode`:
  - `D2S`
  - `S2D`
  - `both`
- Default is `both` for teacher and student.
- D2S uses drone images as queries and all satellite tiles as gallery.
- S2D uses satellite tiles with matched drone images as queries and drone images as gallery.
- Positive satellite lists are read from `pair_pos_sate_img_list`.
- Metrics:
  - `R@1`
  - `R@5`
  - `R@10`
  - `R@top1`
  - `AP`
  - `SDM@1`
  - `SDM@3`
  - `SDM@5`
  - `Dis@1`
  - `Dis@3`
  - `Dis@5`
- `R@top1` uses `ceil(0.01 * gallery_size)`.
- GTA coordinate metrics use the stored drone coordinates and satellite tile-derived coordinates.

Important command:

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode both \
  --batch_size 32
```

Student:

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run>/best_model.pth \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode both \
  --batch_size 32
```

## Notes

- The teacher and student use the same dataloader functions for SUES-200 and GTA-UAV.
- The teacher and student use the same metric functions for SUES-200 and GTA-UAV.
- The only model-specific difference is feature extraction: `TeacherModel` outputs DINOv3 teacher features; `StudentModel` outputs RepViT student features.
- For paper-default evaluation, do not pass `--sues_horizontal_flip`.
- If you want optional TTA ablation on SUES-200, pass `--sues_horizontal_flip` explicitly and report it separately.
