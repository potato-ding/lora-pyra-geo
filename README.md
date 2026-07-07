# LoRA Pyra Geo

This repository is being cleaned around the current teacher/student training
plan.

## Current Model Plan

Teacher:

- DINOv3-7B backbone.
- LoRA on middle transformer blocks, default `[20, 36)`.
- Full fine-tuning on the last four blocks, default `[36, 40)`.
- Sample4Geo paired sampling with InfoNCE.
- Optional identity-level contrast training and hard-pool ordering.
- DeepSpeed multi-GPU training.

Student:

- RepViT-M1.5 baseline.
- `Input -> RepViT-M1.5 -> f4 -> GAP -> BN -> L2`.
- Sample4Geo paired sampling with InfoNCE.
- DeepSpeed multi-GPU training.

## Main Entrypoints

Teacher training:

```bash
deepspeed --num_gpus=8 src/training/teacher/train.py --deepspeed_config ds_config.json
```

Teacher evaluation:

```bash
python src/training/teacher/evaluate.py --checkpoint path/to/teacher_checkpoint.pth
```

Student training:

```bash
deepspeed --num_gpus=8 src/training/student_train.py --deepspeed --deepspeed_config configs/ds_student_baseline.json
```

Student evaluation:

```bash
python src/training/student_test.py --checkpoint path/to/student_checkpoint.pth
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Notes

- The active loss implementations live in `src/loss/`.
- Current entrypoints are the source of truth for new training and evaluation.
