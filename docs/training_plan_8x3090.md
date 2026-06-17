# 8 x RTX 3090 训练计划：教师训练仅 6 组

本文档记录当前实验的主训练方案。教师模型训练种类严格只有 6 个：`T0` 到 `T5`。学生 baseline、边界风险感知蒸馏、教师纯测试、学生纯测试都属于配套流程，不计入教师训练种类。

核心原则是：无论后续是否使用 identity 训练和 hard_pool 训练，都先保留前 10 个 epoch 的 Sample4Geo 最佳结果；后续阶段全部从上一阶段的 `best_model.pth` 重新启动，而不是从上一阶段最后一个 epoch 接着跑。

## 1. 教师 6 组实验总览

教师模型分成两条完全平行的分支：

| 分支 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| 无软正交 | Sample4Geo + InfoNCE | 加载 Stage 1 best 后训练 identity | 加载 Stage 2 best 后训练 hard_pool |
| 有软正交 | Sample4Geo + InfoNCE + soft orth | 加载 Stage 1 best 后训练 identity | 加载 Stage 2 best 后训练 hard_pool |

教师训练种类固定为下面 6 个，不再额外增加一次性 curriculum 版本：

| ID | 训练设置 | 初始化来源 | 是否软正交 | 是否 hard_pool |
|---|---|---|---:|---:|
| T0 | Sample4Geo + InfoNCE | DINOv3 pretrained | no | no |
| T1 | Sample4Geo + InfoNCE | DINOv3 pretrained | yes | no |
| T2 | T0 best -> identity | T0 `best_model.pth` | no | no |
| T3 | T1 best -> identity | T1 `best_model.pth` | yes | no |
| T4 | T2 best -> hard_pool | T2 `best_model.pth` | no | yes |
| T5 | T3 best -> hard_pool | T3 `best_model.pth` | yes | yes |

一句话口径：

```text
教师训练 = T0, T1, T2, T3, T4, T5
总数 = 6
```

这样可以保证：

- T0/T1 永远是独立保存的前 10 epoch Sample4Geo 最佳结果。
- T2/T3 不受 Sample4Geo 后期降点影响，只从 Sample4Geo best 起步。
- T4/T5 不受 identity 后期降点影响，只从 identity best 起步。
- 无软正交和有软正交两条分支完全对齐，方便写论文消融。

## 2. 代码开关

教师训练新增/使用以下关键参数：

| 参数 | 用途 |
|---|---|
| `--training_stage sample4geo` | 只训练 Sample4Geo + InfoNCE 阶段 |
| `--training_stage identity` | 从 `--init_checkpoint` 直接进入 identity 训练 |
| `--training_stage hard_pool` | 从 `--init_checkpoint` 直接进入 identity_hard 训练 |
| `--init_checkpoint` | 加载上一阶段的 `best_model.pth` |
| `--init_checkpoint_strict_trainable true` | 默认开启，要求 checkpoint 覆盖当前所有可训练参数 |
| `--build_hard_pool_before_train` | hard_pool 阶段训练前先构建 hard_pool；`--training_stage hard_pool` 且没有 `--load_hard_pool_path` 时会自动开启 |

等价关系：

```text
--training_stage identity
= --enable_identity_stage --stage1_end_epoch 0

--training_stage hard_pool
= --enable_identity_stage --enable_hard_pool_stage --stage1_end_epoch 0 --stage2_end_epoch 0
```

重要一致性规则：

- 无软正交分支后续阶段不要加 `--use_soft_orth_fusion`。
- 有软正交分支后续阶段必须继续加同样的结构参数：

```bash
--use_soft_orth_fusion \
--local_feature_layers 19,27,36 \
--soft_orth_lambda_init 0.8 \
--soft_orth_detach_global true
```

- 不建议在同一条分支中改变 LoRA block、full fine-tune block、local feature layers 或 soft orth 设置。
- 如果确实要做结构消融，再使用 `--init_checkpoint_strict_trainable false`，并仔细检查日志中的 missing keys。

## 3. 硬件分配

硬件假设：

- 8 x RTX 3090
- 教师训练：2 张卡一个 run，单卡 `batch_size=4`
- 教师训练阶段 University-1652 验证：多卡，跟随当前 DeepSpeed 训练进程
- 教师纯测试：单卡
- 学生训练、蒸馏、测试：单卡

推荐 GPU 组：

| Slot | GPUs | 用途 |
|---|---:|---|
| A | `localhost:0,1` | 无软正交分支 |
| B | `localhost:2,3` | 有软正交分支 |
| C | `localhost:4,5` | 等待依赖后用于 T4，或空闲 |
| D | `localhost:6,7` | 等待依赖后用于 T5，或空闲 |

依赖关系决定训练顺序：

1. T0/T1 可以并行。
2. T2/T3 必须等 T0/T1 的 `best_model.pth` 产生后再启动。
3. T4/T5 必须等 T2/T3 的 `best_model.pth` 产生后再启动。

## 4. 教师 Stage 1：Sample4Geo 训练

你已经完成了前两次实验并保存了 `best_model.pth`。下面命令用于复现实验。

### 4.1 T0：无软正交 Sample4Geo baseline

```bash
deepspeed --include localhost:0,1 src/training/teacher_train.py \
  --training_stage sample4geo \
  --epochs 10 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --triplet_weight 0 \
  --infonce_weight 1.0
```

记录：

```text
src/checkpoint/teacher/<T0_sample4geo_no_soft_run>/best_model.pth
```

### 4.2 T1：有软正交 Sample4Geo

```bash
deepspeed --include localhost:2,3 src/training/teacher_train.py \
  --training_stage sample4geo \
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

记录：

```text
src/checkpoint/teacher/<T1_sample4geo_soft_run>/best_model.pth
```

## 5. 教师 Stage 2：加载 Sample4Geo Best 后训练 Identity

推荐训练长度：20 epoch。这样总训练量对应旧方案的 `10 + 20 = identity30`。

Identity 阶段的 batch 口径：

- `--batch_size` 仍用于 DeepSpeed micro batch 配置，但 identity dataloader 的每卡 ID 数由 `--identity_ids_per_batch` 控制。
- 每个 identity batch 会采样 `identity_ids_per_batch` 个 ID；每个 ID 再取 `identity_sat_per_id` 张 satellite 和 `identity_drone_per_id` 张 drone。
- 每个 epoch 的 batch 数按 ID 数计算，而不是按 drone 图片总数计算：

```text
identity_batches_per_epoch = ceil(num_train_ids / (identity_ids_per_batch * world_size))
```

例如 University-1652 train 有 701 个 ID，2 卡训练且 `--identity_ids_per_batch 8` 时：

```text
ceil(701 / (8 * 2)) = 44
```

所以日志中出现 `batches=44` 是当前 identity sampler 的预期行为。如果希望同样 2 卡下约 88 个 batch，需要把 `--identity_ids_per_batch` 改成 `4`。

### 5.1 T2：T0 best -> identity

```bash
deepspeed --include localhost:0,1 src/training/teacher_train.py \
  --training_stage identity \
  --epochs 20 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<T0_sample4geo_no_soft_run>/best_model.pth \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2
```

记录：

```text
src/checkpoint/teacher/<T2_identity_no_soft_run>/best_model.pth
```

### 5.2 T3：T1 best -> identity

```bash
deepspeed --include localhost:2,3 src/training/teacher_train.py \
  --training_stage identity \
  --epochs 20 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<T1_sample4geo_soft_run>/best_model.pth \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2 \
  --use_soft_orth_fusion \
  --local_feature_layers 19,27,36 \
  --soft_orth_lambda_init 0.8 \
  --soft_orth_detach_global true
```

记录：

```text
src/checkpoint/teacher/<T3_identity_soft_run>/best_model.pth
```

## 6. 教师 Stage 3：加载 Identity Best 后训练 Hard Pool

推荐训练长度：10 epoch。这样总训练量对应旧方案的 `10 + 20 + 10 = hardpool40`。

`--training_stage hard_pool` 会让第 1 个 epoch 直接进入 `identity_hard`。如果没有传入 `--load_hard_pool_path`，代码会在训练前用当前初始化后的教师模型构建 hard_pool。

### 6.1 T4：T2 best -> hard_pool

```bash
deepspeed --include localhost:4,5 src/training/teacher_train.py \
  --training_stage hard_pool \
  --epochs 10 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<T2_identity_no_soft_run>/best_model.pth \
  --build_hard_pool_epoch 0 \
  --save_hard_pool_path outputs/hard_pool_T4_from_<T2_identity_no_soft_run>_epoch{epoch}.json \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2 \
  --hard_drone_per_id 2 \
  --random_drone_per_id 2 \
  --hard_pool_topk 12 \
  --hard_pool_topneg_k 10 \
  --use_ema_for_hard_pool true
```

记录：

```text
src/checkpoint/teacher/<T4_hardpool_no_soft_run>/best_model.pth
```

### 6.2 T5：T3 best -> hard_pool

```bash
deepspeed --include localhost:6,7 src/training/teacher_train.py \
  --training_stage hard_pool \
  --epochs 10 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<T3_identity_soft_run>/best_model.pth \
  --build_hard_pool_epoch 0 \
  --save_hard_pool_path outputs/hard_pool_T5_from_<T3_identity_soft_run>_epoch{epoch}.json \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2 \
  --hard_drone_per_id 2 \
  --random_drone_per_id 2 \
  --hard_pool_topk 12 \
  --hard_pool_topneg_k 10 \
  --use_ema_for_hard_pool true \
  --use_soft_orth_fusion \
  --local_feature_layers 19,27,36 \
  --soft_orth_lambda_init 0.8 \
  --soft_orth_detach_global true
```

记录：

```text
src/checkpoint/teacher/<T5_hardpool_soft_run>/best_model.pth
```

## 7. 验证与最佳权重保存

教师训练阶段：

- Sample4Geo 阶段：每个 epoch 都验证 University-1652。
- Identity 阶段：每 5 个 epoch 验证一次；最后 10 个 epoch 每 2 个 epoch 验证一次；最后一个 epoch 必定验证。
- Hard_pool 阶段：每 5 个 epoch 验证一次；最后 10 个 epoch 每 2 个 epoch 验证一次；最后一个 epoch 必定验证。
- 训练阶段验证是多卡，跟随当前 DeepSpeed 进程。

最佳模型判断：

```text
best = max(D2S_R@1 + S2D_R@1)
```

每个 run 都会保存：

```text
src/checkpoint/teacher/<run>/best_model.pth
src/checkpoint/teacher/<run>/final_model.pth
src/checkpoint/teacher/<run>/best_metrics.json
src/checkpoint/teacher/<run>/hyperparameters.json
```

`best_metrics.json` 的第一部分是最佳结果，后面 `validation_history` 记录每一次验证。

## 8. 非教师训练：学生模型 Baseline

学生 baseline 是纯 RepViT，始终单卡训练和测试。

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

PowerShell：

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

## 9. 非教师训练：边界风险感知蒸馏

蒸馏建议使用最终教师 best，例如 T5 或你选择的最优教师 run。

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

如果显存不够：

- 先把 student `--batch_size` 从 `4` 降到 `2`。
- 再把 `--brd_topk` 从 `4` 降到 `2`。
- 验证阶段可以单独降低 `--val_batch_size`。

## 10. 教师纯测试：不计入训练种类

教师纯测试是单卡，不使用 DeepSpeed。

当前教师测试入口和保存路径规则：

- 推荐入口是 `src/training/teacher_test.py`。
- 旧入口 `src/training/test.py` 已作为兼容 wrapper 保留，会转发到同一套 teacher evaluator。
- `--checkpoint` 可以传 run 目录、`best_model.pth` 或 `final_model.pth`。如果传 run 目录，会优先使用目录下的 `best_model.pth`，没有时再使用 `final_model.pth`。
- 注意：教师训练当前保存的是 `best_model.pth` 和 `final_model.pth`，不是 `last_model.pth`；`last_model.pth` 是学生训练侧的保存命名。
- 测试脚本会自动读取 checkpoint 同目录下的 `hyperparameters.json`，恢复 LoRA、full fine-tune、local fusion、soft orth 等模型结构参数。除非明确做结构消融，不要加 `--no_checkpoint_hparams`。
- checkpoint 加载会检查 trainable 参数覆盖情况和 shape mismatch；如果报 missing/incompatible，优先检查测试参数是否和训练 run 的 `hyperparameters.json` 一致。
- 教师测试特征提取会按实际 image backbone 的 dtype 转输入。当前 DINOv3 teacher backbone 在 CUDA 上是 `bfloat16`，即使模型里有 `float32` 的 `logit_scale`、fusion 或 LoRA 参数，输入图片也会转成 backbone dtype，避免 `Input type (float) and bias type (c10::BFloat16) should be the same`。
- DINOv3 预训练权重、teacher 测试 checkpoint、teacher 续训 `--init_checkpoint` 都使用兼容式 `torch.load(weights_only=True)` 加载；正常情况下不会再打印 PyTorch 关于 `weights_only=False` 的长安全警告。
- 默认结果写到 checkpoint 所在目录的 `teacher_test_results.json`；也可以用 `--output_json` 指定路径，输出目录会自动创建。
- GTA-UAV 按原论文默认只评估 D2S，即 drone query -> satellite gallery；`--gta_query_mode` 默认值为 `D2S`。不建议把 S2D 写入主结果，除非单独做额外消融。
- GTA-UAV 只输出论文需要的 5 个指标：`R@1`、`R@5`、`AP`、`SDM@3`、`DIS@1`。其中 `R@1/R@5/AP/SDM@3` 都是百分制；例如论文中的 `SDM@3=54.07` 对应测试输出 `54.07`，不是 `0.5407`。`DIS@1` 是 top1 预测坐标与 query 坐标的欧氏距离，保持距离单位。
- GTA-UAV satellite tile 坐标按 `zoom_offset_x_y` 文件名解析，当前常量为 `GTA_SATE_LENGTH=24576`、`GTA_TILE_LENGTH=512`。
- SUES-200 默认 `--sues_height all`，会评估 `150/200/250/300` 四个高度；query/gallery 类别映射会在 dataloader 构建时显式校验。水平翻转 TTA 默认关闭，只在传 `--sues_horizontal_flip` 时启用。

### 10.1 教师 best_model / final_model 测试配置

教师每个 run 目录下会同时保存：

```text
src/checkpoint/teacher/<teacher_run>/best_model.pth
src/checkpoint/teacher/<teacher_run>/final_model.pth
src/checkpoint/teacher/<teacher_run>/hyperparameters.json
```

两个权重的含义：

| 权重 | 含义 | 推荐用途 |
|---|---|---|
| `best_model.pth` | 训练过程中按验证集最佳指标保存的权重 | 论文主结果、后续蒸馏、跨数据集测试默认用这个 |
| `final_model.pth` | 最后一个 epoch 结束时保存的权重 | 只用于检查最后 epoch 是否退化或做对照，不作为默认主结果 |

`--checkpoint` 有三种写法：

| 写法 | 实际加载 |
|---|---|
| `--checkpoint src/checkpoint/teacher/<teacher_run>` | 自动优先加载 `best_model.pth`；如果没有 best，再加载 `final_model.pth` |
| `--checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth` | 明确测试 best 权重 |
| `--checkpoint src/checkpoint/teacher/<teacher_run>/final_model.pth` | 明确测试 final 权重 |

如果要同时比较 best 和 final，不要共用默认输出名，否则后一次会覆盖同目录下的 `teacher_test_results.json`。建议显式指定不同的 `--output_json`：

```bash
# 测 best_model.pth
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32 \
  --output_json src/checkpoint/teacher/<teacher_run>/teacher_test_best_1652.json

# 测 final_model.pth
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run>/final_model.pth \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32 \
  --output_json src/checkpoint/teacher/<teacher_run>/teacher_test_final_1652.json
```

GTA-UAV 和 SUES-200 也用同样的 checkpoint 写法，只替换 `--dataset`、`--data_dir` 和对应数据集参数即可。

University-1652：

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run> \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32
```

GTA-UAV：

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run> \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode D2S \
  --batch_size 32
```

SUES-200：

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run> \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

## 11. 学生纯测试：不计入教师训练种类

学生纯测试也是单卡。

当前学生测试入口和保存路径规则：

- 推荐入口是 `src/training/student_test.py`。
- 学生训练保存 `best_model.pth` 和 `last_model.pth`。
- `--checkpoint` 可以传 student run 目录、`best_model.pth` 或 `last_model.pth`。如果传 run 目录，会优先使用目录下的 `best_model.pth`，没有时再使用 `last_model.pth`。
- 默认结果写到 checkpoint 所在目录的 `student_test_results.json`；也可以用 `--output_json` 指定路径。

University-1652：

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run> \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32
```

GTA-UAV：

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run> \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode D2S \
  --batch_size 32
```

SUES-200：

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run> \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

## 12. 记录模板

建议每次训练完成后记录以下路径：

| ID | run name | best checkpoint | best metric json | 备注 |
|---|---|---|---|---|
| T0 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `src/checkpoint/teacher/<run>/best_metrics.json` | no soft Sample4Geo |
| T1 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `src/checkpoint/teacher/<run>/best_metrics.json` | soft Sample4Geo |
| T2 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `src/checkpoint/teacher/<run>/best_metrics.json` | T0 -> identity |
| T3 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `src/checkpoint/teacher/<run>/best_metrics.json` | T1 -> identity |
| T4 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `src/checkpoint/teacher/<run>/best_metrics.json` | T2 -> hard_pool |
| T5 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `src/checkpoint/teacher/<run>/best_metrics.json` | T3 -> hard_pool |

论文中主消融可以按 `T0 -> T1 -> T3 -> T5` 展示软正交分支，也可以按 `T0 -> T2 -> T4` 展示无软正交分支；完整表格保留 T0-T5。
