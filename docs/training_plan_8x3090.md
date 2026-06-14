# 8 x RTX 3090 Training Plan

This document records the planned teacher/student training and evaluation commands for the current project.

Hardware assumption:

- 8 x RTX 3090
- Teacher training: 2 GPUs per run
- Teacher training-time University-1652 validation: multi-card, same DeepSpeed process group as training
- Teacher pure test: single-card
- Teacher per-GPU PID batch size: 4
- Student baseline / distillation: 1 GPU per run
- Student training-time University-1652 validation: single-card
- Student pure test: single-card
- Dataset root examples use `data/U1652`, `data/GTA-UAV-LR/GTA-UAV-LR-baidu`, and `data/SUES-200/SUES-200-512x512`

## 1. GPU Allocation

Recommended GPU grouping for teacher experiments:

| Run slot | GPUs | Usage |
|---|---:|---|
| Slot A | `localhost:0,1` | teacher experiment 1 |
| Slot B | `localhost:2,3` | teacher experiment 2 |
| Slot C | `localhost:4,5` | teacher experiment 3 |
| Slot D | `localhost:6,7` | teacher experiment 4 |

If you want to train only one teacher at a time, use any one pair, for example:

```bash
deepspeed --include localhost:6,7 src/training/teacher_train.py ...
```

Student distillation is planned as single-GPU training. Example GPU choices:

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/student_train.py ...
```

On Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python src/training/student_train.py ...
```

## 2. Teacher Experiments

The teacher code currently supports the following switches:

- Baseline: no local fusion, no soft orthogonal fusion.
- Soft orthogonal fusion: `--use_soft_orth_fusion`.
- Sample4Geo to identity curriculum: `--enable_identity_stage --stage1_end_epoch 10`.
- Hard-pool curriculum: `--enable_identity_stage --enable_hard_pool_stage`.

Important:

- The default local feature layers in code are `19,27,36`.
- `--use_soft_orth_fusion` automatically enables local fusion.
- If neither `--use_local_fusion` nor `--use_soft_orth_fusion` is used, the teacher uses the final global feature only.
- Best teacher checkpoint selection uses `D2S_R@1 + S2D_R@1`.
- Teacher training-time validation schedule:
  - epochs `1-5`: no validation;
  - Sample4Geo stage from epoch `6`: validate every epoch;
  - identity / hard-pool stages: validate every 5 epochs;
  - last 10 epochs of each identity / hard-pool stage: validate every 2 epochs;
  - final epoch: always validate.

### 2.1 Teacher Baseline: Sample4Geo + InfoNCE

Purpose:

- Baseline teacher.
- Data loading: Sample4Geo pair loading.
- Loss: symmetric InfoNCE.
- Retrieval feature: final global feature only.

```bash
deepspeed --include localhost:0,1 src/training/teacher_train.py \
  --epochs 10 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --triplet_weight 0 \
  --infonce_weight 1.0
```

### 2.2 Innovation 1: Soft Orthogonal Local Fusion

Purpose:

- Add local token branch.
- Use soft orthogonal filtering before local-global fusion.
- Default selected DINOv3 blocks: `19,27,36`.

```bash
deepspeed --include localhost:2,3 src/training/teacher_train.py \
  --epochs 10 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --triplet_weight 0 \
  --infonce_weight 1.0 \
  --use_soft_orth_fusion \
  --local_feature_layers 19,27,36 \
  --soft_orth_lambda_init 0.8 \
  --soft_orth_detach_global true
```

This is the corrected form of the command. The local layers are not `5,10,15`; they are `19,27,36` unless you intentionally run an ablation.

### 2.3 Innovation 2: Sample4Geo then Identity-Level Contrastive Training

Purpose:

- Stage 1: Sample4Geo + InfoNCE for 10 epochs.
- Stage 2: identity-level data loading and identity-level contrastive losses.
- No hard-pool yet.

Recommended total epochs: 30.

```bash
deepspeed --include localhost:4,5 src/training/teacher_train.py \
  --epochs 30 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --triplet_weight 0 \
  --infonce_weight 1.0 \
  --enable_identity_stage \
  --stage1_end_epoch 10 \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2
```

If this experiment should also include soft orthogonal fusion, add:

```bash
--use_soft_orth_fusion \
--local_feature_layers 19,27,36 \
--soft_orth_lambda_init 0.8 \
--soft_orth_detach_global true
```

### 2.4 Innovation 3: Identity-Level Training with Hard Pool

Purpose:

- Stage 1: Sample4Geo + InfoNCE.
- Stage 2: identity-level training.
- Stage 3: identity-level hard-pool training.
- Hard-pool is built from EMA teacher features.

Recommended total epochs: 40.

```bash
deepspeed --include localhost:6,7 src/training/teacher_train.py \
  --epochs 40 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --triplet_weight 0 \
  --infonce_weight 1.0 \
  --enable_identity_stage \
  --enable_hard_pool_stage \
  --stage1_end_epoch 10 \
  --stage2_end_epoch 30 \
  --build_hard_pool_epoch 30 \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --hard_drone_per_id 2 \
  --random_drone_per_id 2 \
  --hard_pool_topk 12 \
  --hard_pool_topneg_k 10 \
  --use_ema_for_hard_pool true
```

If this experiment should also include soft orthogonal fusion, add:

```bash
--use_soft_orth_fusion \
--local_feature_layers 19,27,36 \
--soft_orth_lambda_init 0.8 \
--soft_orth_detach_global true
```

## 3. Suggested Parallel Teacher Schedule

If memory allows four concurrent teacher runs, use the four GPU slots:

| Experiment | GPUs | Epochs | Key switches |
|---|---:|---:|---|
| Teacher baseline | `0,1` | 10 | Sample4Geo + InfoNCE |
| Soft orthogonal fusion | `2,3` | 10 | `--use_soft_orth_fusion` |
| Identity curriculum | `4,5` | 30 | `--enable_identity_stage` |
| Hard-pool curriculum | `6,7` | 40 | `--enable_identity_stage --enable_hard_pool_stage` |

If memory, CPU RAM, or disk I/O is tight, run only two teacher jobs at once:

- First wave: baseline + soft orthogonal fusion.
- Second wave: identity curriculum + hard-pool curriculum.

## 4. Student Baseline

Purpose:

- Pure RepViT student baseline.
- Data loading: Sample4Geo pair loading.
- Loss: symmetric InfoNCE.
- No teacher.
- No distillation.
- Best student checkpoint selection uses `D2S_R@1 + S2D_R@1`.

Single GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/student_train.py \
  --epochs 60 \
  --train_data_dir data/U1652/train \
  --val_data_dir data/U1652 \
  --batch_size 8 \
  --val_batch_size 32 \
  --num_workers 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --temperature 0.07 \
  --label_smoothing 0.1
```

PowerShell equivalent:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python src/training/student_train.py `
  --epochs 60 `
  --train_data_dir data/U1652/train `
  --val_data_dir data/U1652 `
  --batch_size 8 `
  --val_batch_size 32 `
  --num_workers 8 `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --temperature 0.07 `
  --label_smoothing 0.1
```

## 5. Student Boundary-Risk-Aware Distillation

Purpose:

- Student still uses Sample4Geo data loading.
- Student still uses InfoNCE as the main loss.
- Teacher is loaded online from:

```text
src/checkpoint/teacher/<teacher_run>/best_model.pth
```

- For each batch, teacher extracts drone/satellite features.
- Teacher cross-view similarity identifies high-risk negatives near the positive ranking boundary.
- Student is optimized with pairwise ranking:

```text
s_pos > s_neg + margin
```

Single GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/student_train.py \
  --epochs 60 \
  --train_data_dir data/U1652/train \
  --val_data_dir data/U1652 \
  --batch_size 4 \
  --val_batch_size 32 \
  --num_workers 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --temperature 0.07 \
  --label_smoothing 0.1 \
  --use_brd_distill \
  --teacher_checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth \
  --brd_weight 1.0 \
  --brd_topk 4 \
  --brd_pair_margin 0.05 \
  --brd_risk_margin 0.0 \
  --brd_risk_tau 0.05 \
  --brd_temperature 0.07
```

PowerShell equivalent:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python src/training/student_train.py `
  --epochs 60 `
  --train_data_dir data/U1652/train `
  --val_data_dir data/U1652 `
  --batch_size 4 `
  --val_batch_size 32 `
  --num_workers 8 `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --temperature 0.07 `
  --label_smoothing 0.1 `
  --use_brd_distill `
  --teacher_checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth `
  --brd_weight 1.0 `
  --brd_topk 4 `
  --brd_pair_margin 0.05 `
  --brd_risk_margin 0.0 `
  --brd_risk_tau 0.05 `
  --brd_temperature 0.07
```

If online teacher distillation causes OOM on one 3090:

- reduce student `--batch_size` to `2`;
- keep `--val_batch_size 32` or reduce it for evaluation;
- reduce `--brd_topk` from `4` to `2`.

## 6. Teacher Evaluation

Teacher pure evaluation is single-card. Do not launch these commands with DeepSpeed unless you intentionally want a separate distributed evaluation ablation.

University-1652:

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32
```

GTA-UAV:

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode both \
  --batch_size 32
```

SUES-200:

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

## 7. Student Evaluation

Student training and student pure evaluation are both single-card.

University-1652:

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run>/best_model.pth \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32
```

GTA-UAV:

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run>/best_model.pth \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode both \
  --batch_size 32
```

SUES-200:

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run>/best_model.pth \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

## 8. Recommended Experiment Table

| ID | Model | Training | Fusion | Hard pool | Main comparison |
|---|---|---|---|---|---|
| T0 | Teacher | Sample4Geo + InfoNCE | none | no | teacher baseline |
| T1 | Teacher | Sample4Geo + InfoNCE | soft orthogonal | no | innovation 1 |
| T2 | Teacher | Sample4Geo then identity | optional | no | innovation 2 |
| T3 | Teacher | Sample4Geo then identity_hard | optional | yes | innovation 3 |
| S0 | Student | Sample4Geo + InfoNCE | none | no | student baseline |
| S1 | Student | Sample4Geo + InfoNCE + BRD | teacher-guided | teacher hard negatives | distillation |

## 9. Notes

- Teacher checkpoints are saved under `src/checkpoint/teacher/<run>/`.
- Student checkpoints are saved under `src/checkpoint/student/<run>/`.
- Teacher `best_model.pth` stores EMA trainable weights.
- Student `best_model.pth` stores the full student checkpoint dictionary.
- Teacher and student training both write `best_metrics.json`.
- In `best_metrics.json`, the first fields record the best result (`epoch`, `best_R@1_sum`, `D2S`, `S2D`), followed by `validation_history` for every validation.
- For teacher evaluation, checkpoint hyperparameters are loaded automatically from `hyperparameters.json` unless `--no_checkpoint_hparams` is set.
- For soft orthogonal fusion experiments, always keep the checkpoint's hyperparameters during evaluation, otherwise the model structure may not match the saved weights.
