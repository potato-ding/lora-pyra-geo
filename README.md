# Lora-Pyra-Geo

UAV–satellite cross-view geo-localization with progressive distillation:
DINOv3 ViT-7B Teacher → DINOv3 ViT-B Middle Teacher → RepViT-M1.5 Student.

## Formal experiments

| Stage | R224 and R256 methods | GPUs | Pairs per GPU |
| --- | --- | ---: | ---: |
| Teacher | T0 PairInfoNCE | 4 | 8 |
| Middle | M0 PairInfoNCE; M1 HRD; M2 HRD + semantic; M3 HRD + semantic + SAM | 2 | 16 |
| Student | S0 PairInfoNCE; S1 TSD; S2 ADSD; S3 SAM + ADSD | 1 | 32 |

The public training configs are `configs/teacher/t0_{224,256}.json`,
`configs/middle_teacher/m{0,1,2,3}_*.json`, and
`configs/student/s{0,1,2,3}_*.json`. The config chooses the method,
resolution, data, optimizer and run assets. Fill the pending checkpoint and
output paths in Middle/Student configs before training. The launch scripts
validate the selected config and GPU count.

From the repository root in the `pyra_geo` environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_teacher.sh configs/teacher/t0_224.json
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_middle_teacher.sh configs/middle_teacher/m3_sam_hrd_sem_224.json
CUDA_VISIBLE_DEVICES=0 bash scripts/train_student.sh configs/student/s3_sam_adsd_224.json
```

Use the corresponding `_256.json` config for R256. The Middle and Student
scripts accept `--check` to validate templates before their pending run
asset paths are filled. The Python entries are
`src.training.teacher.train`, `src.middle_teacher.formal_train`, and
`src.student.formal_train`. Student Top128 and calibration assets are built
through `scripts/build_student_assets.sh`.

Teacher blocks are zero-based: 0–19 frozen, 20–35 LoRA, 36–39 full
fine-tuning. Middle checkpoint selection starts at epoch 6; Student
selection starts at epoch 11. Both select the best checkpoint using U1652
D2S R@1 + S2D R@1, evaluated on one GPU with batch 16. The selected
checkpoint is `best_model.pth` beside `train.log` in the configured run
directory. Middle and Student runs also save `config.json` there.

## Evaluation

The single-GPU entry is `bash scripts/eval.sh`. It accepts
`--model-type teacher|middle|student` and
`--dataset u1652|sues200|gta|anyvisloc|all`. The checkpoint's training
resolution must match `--image-size`. Batch size is 16.

```bash
bash scripts/eval.sh --gpu 0 --model-type student   --checkpoint /path/to/run/best_model.pth   --dataset all --image-size 224 --data-root data
```

The four formal result files are written beside the checkpoint:
`test_1652.json`, `test_sues200.json`,
`test_gta_cross_area_d2s.json`, and `test_anyvisloc.json`.
Existing result files are not overwritten. U1652 evaluates D2S/S2D
R@1, R@5 and AP; SUES-200 evaluates D2S/S2D at 150/200/250/300 m;
GTA-UAV evaluates cross-area D2S; AnyVisLoc evaluates the released
Scene_01/02 aerial-map retrieval subset. See `src/evaluation/evaluate.py`
for the exact runtime checks and dataset-specific protocol metadata.

## Runtime assets

Provide the licensed DINOv3 source under `src/models/dinov3_main/`, the
pretrained model weights under `src/models/dinov3-pth/`, and datasets under
`data/`. These assets and `src/checkpoint/` are excluded from Git.
The current source manifest for new runs is
`configs/source_contract_v3.json`; the v2 manifest is retained as a
historical identity record and is not used by the new training entries.

Install the environment from `environment.yml` or
`requirements.txt`, and run `python -m pytest -q` for CPU-compatible
tests. No training or full-model evaluation runs as part of the test suite.
