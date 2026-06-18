# 8 × RTX 3090 完整训练、蒸馏与评估手册

本文档是本项目唯一维护的实验文档，覆盖：

- 教师模型 T0–T5 六组训练计划；
- 教师的数据采样、损失、Hard Pool、LoRA 与软正交融合；
- 学生 RepViT baseline；
- DeepSpeed 多卡边界风险感知蒸馏；
- University-1652、SUES-200、GTA-UAV 的统一评估协议；
- checkpoint、指标、参数和排错方法。

后续代码或实验方案变更只更新本文件。

## 1. 代码入口与目录

### 1.1 主要代码

| 功能 | 路径 |
|---|---|
| 教师训练入口 | `src/training/teacher_train.py` |
| 教师训练实现 | `src/training/teacher/train.py` |
| 教师模型 | `src/models/teacher/model.py` |
| 学生训练与蒸馏 | `src/training/student_train.py` |
| 学生模型 | `src/models/student_model.py` |
| 教师测试 | `src/training/teacher_test.py` |
| 学生测试 | `src/training/student_test.py` |
| 训练数据集与 sampler | `src/dataset/datasets.py`、`src/dataset/teacher/datasets.py` |
| 统一验证数据集 | `src/dataset/teacher/val_dataloaders.py` |
| Identity loss | `src/loss/identity_losses.py` |
| Sample4Geo InfoNCE | `src/loss/blocks_infoNCE.py` |
| 统一指标实现 | `src/utils/train_eval_utils.py` |
| 教师 DeepSpeed 配置 | `ds_config.json` |
| 学生蒸馏 DeepSpeed 配置 | `configs/ds_student_distill.json` |

### 1.2 数据目录

```text
data/U1652
data/SUES-200/SUES-200-512x512
data/GTA-UAV-LR/GTA-UAV-LR-baidu
```

University-1652 训练目录：

```text
data/U1652/train/satellite/<pid>/*
data/U1652/train/drone/<pid>/*
```

## 2. 总体实验设计

教师训练严格分为 T0–T5 六组。学生 baseline、学生蒸馏和纯测试不计入教师实验数量。

| ID | 阶段 | 初始化来源 | 软正交 | Hard Pool |
|---|---|---|---:|---:|
| T0 | Sample4Geo + InfoNCE | DINOv3 pretrained | 否 | 否 |
| T1 | Sample4Geo + InfoNCE | DINOv3 pretrained | 是 | 否 |
| T2 | Identity | T0 best | 否 | 否 |
| T3 | Identity | T1 best | 是 | 否 |
| T4 | Identity Hard | T2 best | 否 | 是 |
| T5 | Identity Hard | T3 best | 是 | 是 |

主原则：

1. T0/T1 独立训练 10 epoch，保存 Sample4Geo 最优结果。
2. T2/T3 从对应的 Sample4Geo `best_model.pth` 重新启动优化器和调度器。
3. T4/T5 从对应的 Identity `best_model.pth` 重新启动。
4. 后续阶段不从上一阶段最后一个 epoch 直接续跑。
5. 同一分支必须保持 LoRA block、full fine-tune block、local layers 和软正交设置一致。

训练依赖：

```text
T0 ──> T2 ──> T4
T1 ──> T3 ──> T5
```

## 3. 8 卡资源分配

硬件假设：8 × RTX 3090。

### 3.1 教师训练

| Slot | GPU | 建议任务 |
|---|---|---|
| A | `0,1` | T0 → T2 |
| B | `2,3` | T1 → T3 |
| C | `4,5` | T4 |
| D | `6,7` | T5 |

教师每个 run 使用 2 卡，默认每卡 `batch_size=4`。

### 3.2 学生训练

- 学生 baseline 可单卡运行。
- 风险感知蒸馏推荐使用 2–4 卡 DeepSpeed。
- 教师模型在每张蒸馏卡上保留一份冻结副本。
- 学生测试和教师纯测试默认单卡。

## 4. 教师模型与训练机制

### 4.1 三种训练模式

| mode | 数据 | 损失 |
|---|---|---|
| `sample4geo` | 每个 PID 一对 satellite/drone | 双向 InfoNCE |
| `identity` | 每个 PID 多张跨视角图像 | Cross-domain identity + same-domain triplet + weak Sample4Geo |
| `identity_hard` | Identity batch 中加入 Hard Pool drone | 与 `identity` 相同 |

推荐使用显式阶段：

```text
--training_stage sample4geo
--training_stage identity
--training_stage hard_pool
```

等价旧开关：

```text
identity
= --enable_identity_stage --stage1_end_epoch 0

hard_pool
= --enable_identity_stage --enable_hard_pool_stage
   --stage1_end_epoch 0 --stage2_end_epoch 0
```

### 4.2 Sample4Geo 采样

输出：

```python
sat_img, drone_img, label, pid
```

训练时整理为：

```text
images:    [2B, C, H, W]
labels:    [2B]
view_type: [2B]  # satellite=0, drone=1
```

`Sample4GeoBatchSampler` 先构建全局 batch，再切分到各 rank，保证整个跨卡 batch 内 PID 不重复。InfoNCE 中第 `i` 个 drone 和第 `i` 个 satellite 是唯一正样本，其余 PID 都是负样本。

### 4.3 Identity 采样

每卡每个 batch 采样：

```text
identity_ids_per_batch 个 PID
每个 PID：
  identity_sat_per_id 张 satellite
  identity_drone_per_id 张 drone
```

全局 batch 内 PID 仍保持唯一。

University-1652 训练集有 701 个 PID。2 卡、每卡 8 个 PID 时：

```text
ceil(701 / (8 × 2)) = 44 batches/epoch
```

### 4.4 Identity Hard 采样

每个 PID 采样：

```text
identity_sat_per_id 张 satellite
hard_drone_per_id 张 hard drone
random_drone_per_id 张 random drone
```

Hard Pool 不足时：

1. 先取已有 hard drone；
2. 不足部分退化到随机 drone；
3. random drone 尽量避开已选择的 hard drone；
4. 每个 epoch 汇总 fallback 统计。

`identity_hard` 与 `identity` 的损失完全相同，仅 drone 采样方式不同。

### 4.5 教师损失

#### Sample4Geo

\[
L_{\text{S4G}}
=\frac{1}{2}
\left(
\operatorname{CE}(S_{D2S}, y)
+
\operatorname{CE}(S_{S2D}, y)
\right)
\]

默认：

```text
L = infonce_weight × L_S4G
triplet_weight = 0
```

#### Identity

Cross-domain identity contrast：

- drone anchor 只与 satellite candidates 比较；
- satellite anchor 只与 drone candidates 比较；
- 支持同 PID 多正样本。

Same-domain batch-hard triplet：

- drone 域和 satellite 域分别计算；
- 每个 anchor 选择 hardest positive 与 hardest negative。

Weak Sample4Geo anchor：

- 每个 PID 生成一个 drone anchor 和一个 satellite anchor；
- anchor 可选 `first` 或 `mean`；
- 对 PID anchor 计算双向 InfoNCE。

总损失：

\[
L_{\text{identity}}
=\lambda_{\text{id}}L_{\text{cross-id}}
+\lambda_{\text{tri}}L_{\text{same-triplet}}
+\lambda_{\text{weak}}L_{\text{weak-S4G}}
\]

默认权重：

```text
identity_loss_weight=1.0
same_domain_triplet_weight=0.2
weak_sample4geo_weight=0.2
```

### 4.6 Hard Pool 构建

Hard Pool 用于寻找与错误 satellite PID 边界最接近的 drone。

构建过程：

1. 切换到 eval；
2. 可选应用 EMA 权重；
3. 提取所有训练集 satellite/drone 特征；
4. 每个 PID 计算 satellite prototype；
5. 为每张 drone 计算 boundary risk；
6. 每个 PID 保留 top-K drone。

Satellite prototype：

\[
p_i=\operatorname{Normalize}
\left(
\operatorname{Mean}\{f_s:s\in i\}
\right)
\]

Drone risk：

\[
r(d_i)
=
\operatorname{MeanTopK}_{j\ne i}
\operatorname{sim}(d_i,p_j)
-
\operatorname{sim}(d_i,p_i)
\]

风险越高表示 drone 越接近错误 PID prototype，同时越远离正确 prototype。

JSON 主要结构：

```json
{
  "meta": {
    "epoch": 0,
    "model_source": "ema",
    "hard_pool_topk": 12,
    "hard_pool_topneg_k": 10
  },
  "hard_pool": {
    "0001": [
      {
        "image_path": "...",
        "boundary_risk": 0.123,
        "pos_sim": 0.456,
        "topk_neg_mean": 0.579,
        "top1_neg_pid": "0032"
      }
    ]
  },
  "id_risk": {
    "0001": 0.118
  }
}
```

`id_risk` 是该 PID top-3 hard samples 的平均风险。

### 4.7 教师微调结构

教师为 DINOv3，默认三段式微调：

```text
底部 blocks：冻结
中间 blocks：LoRA
最后 4 个 blocks：full fine-tune
```

默认解析：

```text
full_finetune_start_block = -4
full_finetune_end_block   = 总 block 数
lora_start_block          = min(20, full_start)
lora_end_block            = full_start
```

Local/PYRA 默认从 block `19,27,36` 提取 patch tokens，通过 local cross-attention 得到局部特征。

普通融合：

\[
f_{\text{fused}}=f_{\text{global}}+\gamma f_{\text{local}}
\]

\[
\gamma=0.05\cdot\sigma(\gamma_{\text{raw}})
\]

启用 `--use_soft_orth_fusion` 后，local feature 先进行可学习软正交投影。

软正交分支后续阶段必须始终携带：

```bash
--use_soft_orth_fusion \
--local_feature_layers 19,27,36 \
--soft_orth_lambda_init 0.8 \
--soft_orth_detach_global true
```

### 4.8 教师优化器

| 参数组 | 学习率 | weight decay |
|---|---:|---:|
| LoRA decay | `lr` | `0.01` |
| LoRA no-decay | `lr` | `0` |
| full backbone decay | `lr × full_finetune_lr_mult` | `0.01` |
| full backbone no-decay | `lr × full_finetune_lr_mult` | `0` |
| local/fusion decay | `lr` | `0.01` |
| local/fusion no-decay | `lr` | `0` |
| logit scale | `lr × logit_scale_lr_mult` | `0` |

安装 DeepSpeed CPU Adam 时使用 `DeepSpeedCPUAdam`，否则使用 AdamW。

## 5. 教师 T0–T5 命令

### 5.1 T0：无软正交 Sample4Geo

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

### 5.2 T1：有软正交 Sample4Geo

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

### 5.3 T2：T0 best → Identity

```bash
deepspeed --include localhost:0,1 src/training/teacher_train.py \
  --training_stage identity \
  --epochs 20 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<T0_run>/best_model.pth \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2
```

### 5.4 T3：T1 best → Identity

```bash
deepspeed --include localhost:2,3 src/training/teacher_train.py \
  --training_stage identity \
  --epochs 20 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<T1_run>/best_model.pth \
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

### 5.5 T4：T2 best → Hard Pool

```bash
deepspeed --include localhost:4,5 src/training/teacher_train.py \
  --training_stage hard_pool \
  --epochs 10 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<T2_run>/best_model.pth \
  --build_hard_pool_epoch 0 \
  --save_hard_pool_path outputs/hard_pool_T4_epoch{epoch}.json \
  --identity_ids_per_batch 8 \
  --identity_sat_per_id 1 \
  --hard_drone_per_id 2 \
  --random_drone_per_id 2 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2 \
  --hard_pool_topk 12 \
  --hard_pool_topneg_k 10 \
  --use_ema_for_hard_pool true
```

### 5.6 T5：T3 best → Hard Pool

```bash
deepspeed --include localhost:6,7 src/training/teacher_train.py \
  --training_stage hard_pool \
  --epochs 10 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<T3_run>/best_model.pth \
  --build_hard_pool_epoch 0 \
  --save_hard_pool_path outputs/hard_pool_T5_epoch{epoch}.json \
  --identity_ids_per_batch 8 \
  --identity_sat_per_id 1 \
  --hard_drone_per_id 2 \
  --random_drone_per_id 2 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2 \
  --hard_pool_topk 12 \
  --hard_pool_topneg_k 10 \
  --use_ema_for_hard_pool true \
  --use_soft_orth_fusion \
  --local_feature_layers 19,27,36 \
  --soft_orth_lambda_init 0.8 \
  --soft_orth_detach_global true
```

如果已有 Hard Pool，可使用：

```bash
--load_hard_pool_path outputs/hard_pool_epoch0.json
```

此时不会重复构建。

## 6. 教师验证与保存

训练验证使用 EMA 权重。

验证频率：

- Sample4Geo：每个 epoch；
- Identity：每 5 epoch，最后 10 epoch 每 2 epoch，最终 epoch 必验；
- Hard Pool：规则同 Identity。

选择指标：

```text
D2S_R@1 + S2D_R@1
```

每个教师 run 保存：

```text
src/checkpoint/teacher/<run>/best_model.pth
src/checkpoint/teacher/<run>/final_model.pth
src/checkpoint/teacher/<run>/best_metrics.json
src/checkpoint/teacher/<run>/hyperparameters.json
```

| 文件 | 用途 |
|---|---|
| `best_model.pth` | 主结果、后续阶段初始化、学生蒸馏 |
| `final_model.pth` | 最后 epoch 对照 |
| `best_metrics.json` | 最佳结果及完整验证历史 |
| `hyperparameters.json` | 恢复模型结构与训练参数 |

## 7. 学生模型 Baseline

学生为 RepViT-M1.5：

```text
RepViT f4 feature map
→ global average pooling
→ BatchNorm1d(512)
→ L2 normalize
→ 512-d embedding
```

单卡 baseline：

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

学生 baseline 使用双向 Sample4Geo InfoNCE。默认 AdamW、iteration-level warmup + cosine scheduler 和 AMP。

## 8. 边界风险感知蒸馏

### 8.1 蒸馏目标

当前方法不是特征维度对齐或 KL 蒸馏，而是关系排序蒸馏。教师和学生特征维度可以不同。

对一个跨视角相似度矩阵，教师定义负样本风险：

\[
w_{ij}
=
\sigma
\left(
\frac{t_{ij}-t_{ii}+m_r}{\tau_r}
\right)
\]

风险越高，表示负样本越接近或超过正确匹配。

学生排序损失：

\[
\ell_{ij}
=
\operatorname{softplus}
\left(
\frac{s_{ij}-s_{ii}+m_p}{T}
\right)
\]

仅保留教师相似度最高的 top-K negatives，并用风险权重加权：

\[
L_{\text{BRD}}
=
\frac{1}{2}
\left(
L_{D2S}+L_{S2D}
\right)
\]

总损失：

\[
L
=L_{\text{InfoNCE}}
+\lambda_{\text{local}}L_{\text{local}}
+\lambda_{\text{BRD}}L_{\text{BRD}}
\]

其中 local alignment 默认关闭。

### 8.2 在线教师

- 每张 GPU 加载一份冻结教师；
- 教师使用 eval + inference mode；
- 不计算教师梯度；
- 学生和教师处理完全相同的本地图像；
- 教师输出只用于构造全局风险关系。

### 8.3 DeepSpeed 多卡实现

多卡实现保持原数据构造和损失定义不变：

1. 数据仍由 `U1652PairDataset` 生成；
2. 图像增强仍使用原学生 transforms；
3. sampler 先构建 PID 唯一的全局 batch；
4. 各 rank 得到互不重叠的本地分片；
5. 学生特征采用可反传 all-gather；
6. 冻结教师特征采用无梯度 all-gather；
7. 聚合后恢复为 `[all_drone, all_satellite]`，保证矩阵对角线仍是正样本；
8. InfoNCE、BRD 和可选 local alignment 在全局 batch 上计算。

### 8.4 Batch size 口径

`--batch_size` 表示每张 GPU 的 pair 数；一个 pair 包含一张 drone 和一张 satellite。

4 卡、每卡 4 pair：

```text
local pair batch             = 4
local image batch            = 8
global pair batch per step   = 16
global image batch per step  = 32
```

梯度累积：

```text
effective pair batch
= local_pair_batch × world_size × grad_accum_steps
```

注意：`grad_accum_steps` 只扩大优化器有效 batch。每次 BRD 的负样本集合仍来自当前 step 的跨卡 global pair batch。

### 8.5 DeepSpeed 配置

默认文件：

```text
configs/ds_student_distill.json
```

默认：

```text
BF16
ZeRO-1
gradient_accumulation_steps=1
```

支持 ZeRO-0/1/2，不支持 ZeRO-3。ZeRO-3 会切分模型参数，而当前 `best_model.pth` 和 `last_model.pth` 需要直接导出完整学生权重。

传入 `--no_amp` 会同时关闭 DeepSpeed BF16 和 FP16。

### 8.6 推荐多卡蒸馏命令

4 卡：

```bash
deepspeed --include localhost:0,1,2,3 src/training/student_train.py \
  --deepspeed \
  --deepspeed_config configs/ds_student_distill.json \
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

2 卡：

```bash
deepspeed --include localhost:0,1 src/training/student_train.py \
  --deepspeed \
  --batch_size 4 \
  --use_brd_distill \
  --teacher_checkpoint src/checkpoint/teacher/<teacher_run>/best_model.pth
```

DeepSpeed launcher 检测到 `WORLD_SIZE > 1` 时，即使没有显式传入 `--deepspeed`，也会自动进入 DeepSpeed 路径。建议仍显式添加，方便复现实验。

### 8.7 显存调整

按以下顺序调整：

1. `--batch_size 4` 降到 `2`；
2. `--brd_topk 4` 降到 `2`；
3. 降低 `--val_batch_size`；
4. 使用 `--grad_accum_steps` 恢复有效优化 batch。

不要只依靠梯度累积扩大 BRD negatives；BRD negatives 数量由单步 global pair batch 决定。

### 8.8 学生保存内容

```text
src/checkpoint/student/<run>/best_model.pth
src/checkpoint/student/<run>/last_model.pth
src/checkpoint/student/<run>/best_metrics.json
src/checkpoint/student/<run>/training_record.txt
src/checkpoint/student/<run>/deepspeed/last/
```

- `best_model.pth`、`last_model.pth` 可由 `student_test.py` 直接加载；
- `deepspeed/last/` 保存优化器、scheduler 和 ZeRO 状态；
- 多卡验证仍以 `D2S_R@1 + S2D_R@1` 选择 best。

### 8.9 分布式通信检查

Linux 环境：

```bash
torchrun --standalone --nproc_per_node=2 \
  tests/check_student_distributed_gather.py
```

成功输出：

```text
student distributed paired gather smoke test passed
```

## 9. 统一评估协议

教师和学生使用相同的数据构建与指标函数，区别仅在特征提取模型。

### 9.1 University-1652

方向：

```text
D2S：query_drone → gallery_satellite
S2D：query_satellite → gallery_drone
```

指标：

```text
R@1, R@5, R@10, mAP
```

训练阶段 best 指标：

```text
D2S_R@1 + S2D_R@1
```

教师训练验证和学生 DeepSpeed 蒸馏验证均为多卡；纯测试默认单卡。

### 9.2 SUES-200

默认评估高度：

```text
150, 200, 250, 300
```

每个高度都评估 D2S 和 S2D。

指标：

```text
R@1
R@5
R@10
R@top1
AP
```

其中：

```text
top1 = ceil(0.01 × gallery_size)
```

水平翻转 TTA 默认关闭。只有额外消融时传：

```bash
--sues_horizontal_flip
```

主结果不要混用 TTA 与非 TTA。

### 9.3 GTA-UAV

Split：

```text
cross-area
same-area
```

方向：

```text
D2S
S2D
both
```

默认 `D2S`，与论文主协议一致。

主指标：

```text
R@1
R@5
AP
SDM@3
DIS@1
```

`R@1`、`R@5`、`AP`、`SDM@3` 为百分制；`DIS@1` 为 top-1 坐标距离。

Satellite tile 坐标从 `zoom_offset_x_y` 文件名解析，当前常量：

```text
GTA_SATE_LENGTH=24576
GTA_TILE_LENGTH=512
```

除非单独做补充实验，论文主结果只报告 D2S。

## 10. 教师纯测试

`--checkpoint` 可以传：

```text
教师 run 目录
best_model.pth
final_model.pth
```

传目录时优先选择 `best_model.pth`，不存在时才使用 `final_model.pth`。

测试脚本会自动读取同目录的 `hyperparameters.json` 恢复 LoRA、full fine-tune、local fusion 和 soft orth 设置。正常测试不要添加 `--no_checkpoint_hparams`。

### 10.1 University-1652

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run> \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32
```

### 10.2 GTA-UAV

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run> \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode D2S \
  --batch_size 32
```

### 10.3 SUES-200

```bash
python src/training/teacher_test.py \
  --checkpoint src/checkpoint/teacher/<teacher_run> \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

默认输出：

```text
<teacher_run>/teacher_test_results.json
```

比较 best/final 时必须指定不同 `--output_json`，避免覆盖。

## 11. 学生纯测试

`--checkpoint` 可以传 student run 目录、`best_model.pth` 或 `last_model.pth`。传目录时优先 best。

### 11.1 University-1652

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run> \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32
```

### 11.2 GTA-UAV

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run> \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode D2S \
  --batch_size 32
```

### 11.3 SUES-200

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<student_run> \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

默认输出：

```text
<student_run>/student_test_results.json
```

## 12. 关键参数表

### 12.1 教师参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--epochs` | `22` | 当前 run epoch |
| `--batch_size` | `4` | Sample4Geo 每卡 pair 数 |
| `--val_batch_size` | `32` | 验证 batch |
| `--grad_accum_steps` | `1` | 梯度累积 |
| `--lr` | `1e-4` | 主学习率 |
| `--warmup_ratio` | `0.05` | warmup 比例 |
| `--ema_decay` | `0.999` | EMA |
| `--identity_ids_per_batch` | `8` | Identity 每卡 PID 数 |
| `--identity_drone_per_id` | `4` | Identity 每 PID drone |
| `--identity_sat_per_id` | `1` | 每 PID satellite |
| `--hard_drone_per_id` | `2` | Hard Pool drone |
| `--random_drone_per_id` | `2` | 随机 drone |
| `--hard_pool_topk` | `12` | 每 PID 保留 hard drone |
| `--hard_pool_topneg_k` | `10` | 风险计算 negative prototype 数 |
| `--lora_rank` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha |
| `--lora_dropout` | `0.1` | LoRA dropout |
| `--full_finetune_lr_mult` | `0.1` | full backbone LR 倍率 |
| `--local_feature_layers` | `19,27,36` | local token 层 |
| `--soft_orth_lambda_init` | `0.8` | 软正交初始化 |
| `--identity_loss_weight` | `1.0` | Cross-ID loss |
| `--same_domain_triplet_weight` | `0.2` | Same-domain triplet |
| `--weak_sample4geo_weight` | `0.2` | Weak S4G |

### 12.2 学生与蒸馏参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--epochs` | `60` | 训练 epoch |
| `--batch_size` | `8` | 每卡 pair 数 |
| `--val_batch_size` | `32` | 验证 batch |
| `--deepspeed_config` | `configs/ds_student_distill.json` | DS 配置 |
| `--grad_accum_steps` | `1` | 梯度累积 |
| `--lr` | `1e-4` | AdamW 学习率 |
| `--weight_decay` | `1e-4` | weight decay |
| `--temperature` | `0.07` | InfoNCE 初始温度 |
| `--label_smoothing` | `0.1` | InfoNCE smoothing |
| `--val_interval` | `5` | 验证间隔 |
| `--use_local_align` | `False` | 局部对齐 |
| `--local_align_weight` | `0.01` | 局部损失权重 |
| `--use_brd_distill` | `False` | 启用风险蒸馏 |
| `--brd_weight` | `1.0` | BRD 权重 |
| `--brd_topk` | `4` | 每个 anchor 的高风险 negatives |
| `--brd_pair_margin` | `0.05` | 学生排序 margin |
| `--brd_risk_margin` | `0.0` | 教师风险 margin |
| `--brd_risk_tau` | `0.05` | 风险 sigmoid 温度 |
| `--brd_temperature` | `0.07` | 学生排序温度 |
| `--brd_risk_threshold` | `0.0` | 风险权重阈值 |

## 13. 实验记录模板

### 13.1 教师

| ID | run | best checkpoint | best metrics | 备注 |
|---|---|---|---|---|
| T0 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `best_metrics.json` | no-soft S4G |
| T1 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `best_metrics.json` | soft S4G |
| T2 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `best_metrics.json` | T0 → identity |
| T3 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `best_metrics.json` | T1 → identity |
| T4 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `best_metrics.json` | T2 → hard |
| T5 |  | `src/checkpoint/teacher/<run>/best_model.pth` | `best_metrics.json` | T3 → hard |

### 13.2 学生

| ID | 教师 | GPU 数 | local batch | global batch | best R@1 sum | checkpoint |
|---|---|---:|---:|---:|---:|---|
| S0 baseline | 无 | 1 | 8 | 8 |  |  |
| S1 BRD | T5/最优教师 |  |  |  |  |  |

## 14. 验证与排错

### 14.1 全部测试

```bash
python -m pytest -q
```

### 14.2 编译检查

```bash
python -m py_compile \
  src/training/teacher_train.py \
  src/training/student_train.py \
  src/dataset/datasets.py \
  src/loss/identity_losses.py
```

### 14.3 常见问题

1. `identity_hard` 报 Hard Pool 缺失：

```text
使用 --load_hard_pool_path
或确保训练前构建 Hard Pool
```

2. 教师 checkpoint missing/incompatible：

```text
检查同目录 hyperparameters.json
检查软正交、local layers、LoRA/full block 是否一致
```

3. 学生多卡显存不足：

```text
先降 local batch，再降 brd_topk，最后降 val batch
```

4. DeepSpeed 学生训练拒绝 ZeRO-3：

```text
改用 ZeRO-1 或 ZeRO-2
```

5. BRD negatives 太少：

```text
增加 GPU 数或每卡 batch
梯度累积不会增加单步 BRD negative 数量
```

6. SUES/GTA 主结果不一致：

```text
SUES 主结果关闭 horizontal flip
GTA 主结果使用 cross-area + D2S
```

7. 教师测试 dtype mismatch：

```text
使用当前 teacher_test.py
它会按照 backbone dtype 转换输入
```

## 15. 最终论文口径

建议教师消融：

```text
无软正交：T0 → T2 → T4
有软正交：T1 → T3 → T5
```

建议学生对比：

```text
RepViT baseline
RepViT + BRD（最优教师）
RepViT + BRD + local alignment（可选消融）
```

主 checkpoint 统一使用 `best_model.pth`。主 University-1652 选择指标统一使用：

```text
D2S_R@1 + S2D_R@1
```

GTA-UAV 主结果统一使用：

```text
cross-area + D2S
```

SUES-200 主结果统一使用：

```text
150/200/250/300 全高度、D2S/S2D、关闭水平翻转 TTA
```
