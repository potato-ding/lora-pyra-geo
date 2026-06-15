# Teacher 训练文档

本文档说明当前 teacher 训练代码的入口、训练阶段、数据采样、loss、hard_pool、模型微调策略、保存文件和全部命令行超参。

代码入口：

- 训练入口：[src/training/teacher_train.py](../src/training/teacher_train.py)
- Teacher 模型：[src/models/teacher_model.py](../src/models/teacher_model.py)
- 数据集与 sampler：[src/dataset/datasets.py](../src/dataset/datasets.py)
- Identity loss：[src/loss/identity_losses.py](../src/loss/identity_losses.py)
- Sample4Geo InfoNCE：[src/loss/blocks_infoNCE.py](../src/loss/blocks_infoNCE.py)

## 1. 总体训练流程

当前 teacher 训练支持三种 mode：

| mode | 用途 | DataLoader | Loss |
|---|---|---|---|
| `sample4geo` | 原始 Sample4Geo-style 对称 InfoNCE 训练 | `Sample4GeoU1652Dataset` + `Sample4GeoBatchSampler` | symmetric InfoNCE，默认权重 `1.0` |
| `identity` | 身份级训练，每个 batch 含多个 ID、每个 ID 多张 drone 和 satellite | `IdentityU1652Dataset` + `IdentityBatchSampler` | cross-domain identity contrast + same-domain triplet + weak Sample4Geo anchor |
| `identity_hard` | 使用 hard_pool 的身份级 hard drone 采样 | `IdentityU1652Dataset(mode=identity_hard)` + `IdentityBatchSampler` | 与 `identity` 相同 |

训练循环按 epoch 切换 mode。每个 epoch 开始会打印当前 mode、sampler 配置、fusion 状态和 loss 权重。每个 epoch 结束会做 EMA evaluation，并在达到指定 epoch 后构建 hard_pool。

## 2. 多阶段 mode 切换

mode 由 `get_training_mode(epoch, args)` 控制：

```python
if not args.enable_identity_stage:
    return "sample4geo"
if epoch <= args.stage1_end_epoch:
    return "sample4geo"
if not args.enable_hard_pool_stage:
    return "identity"
if epoch <= args.stage2_end_epoch:
    return "identity"
return "identity_hard"
```

默认配置下，如果不加 `--enable_identity_stage`，训练全程都是 `sample4geo`。

一个典型三阶段设置：

| 阶段 | epoch 范围 | mode | 说明 |
|---|---|---|---|
| stage 1 | `1` 到 `stage1_end_epoch` | `sample4geo` | 使用原始跨视角 InfoNCE 建立全局对齐 |
| stage 2 | `stage1_end_epoch + 1` 到 `stage2_end_epoch` | `identity` | 使用身份级 batch 和多 loss 进行身份聚合 |
| stage 3 | `stage2_end_epoch + 1` 到结束 | `identity_hard` | 使用 hard_pool 中的困难 drone 样本强化边界 |

注意：`--enable_hard_pool_stage` 必须和 `--enable_identity_stage` 一起使用。

## 3. Dataset / Sampler / DataLoader

### 3.1 Sample4Geo DataLoader

文件：[src/dataset/datasets.py](../src/dataset/datasets.py)

相关类：

- `Sample4GeoU1652Dataset`
- `Sample4GeoBatchSampler`
- `create_1652_train_dataset`

输出格式：

```python
sat_img, drone_img, label, pid
```

训练循环中会整理成：

```python
images:    [2B, C, H, W]
labels:    [2B]
view_type: [2B]  # satellite=0, drone=1
```

Sample4Geo sampler 会尽量保证一个全局 batch 内 PID 不重复，因为原始 symmetric InfoNCE 默认 `sat_feats[i]` 和 `drone_feats[i]` 是唯一正样本，其它 batch 内样本都是负样本。

### 3.2 Identity DataLoader

相关类：

- `IdentityU1652Dataset`
- `IdentityBatchSampler`
- `collate_identity_u1652_batch`
- `create_identity_1652_train_dataset`

`identity` mode 每个 batch 采样 `P` 个 ID：

```text
P = identity_ids_per_batch
```

每个 ID 采样：

```text
identity_sat_per_id 张 satellite
identity_drone_per_id 张随机 drone
```

输出格式：

```python
{
    "images": images,
    "labels": labels,
    "view_type": view_type,
    "pids": pids,
    "image_path": image_paths,
    "image_paths": image_paths,
}
```

其中：

- `view_type=0` 表示 satellite
- `view_type=1` 表示 drone

### 3.3 Identity Hard DataLoader

`identity_hard` mode 使用同一个 `IdentityU1652Dataset`，但 `sampling_mode="identity_hard"`。

每个 ID 采样：

```text
identity_sat_per_id 张 satellite
hard_drone_per_id 张 hard drone
random_drone_per_id 张 random drone
```

hard drone 从 `hard_pool[pid]` 中采样。random drone 从该 ID 的所有 drone 图像中采样，并尽量避开已经选中的 hard drone。

如果某个 ID 的 hard_pool 不足：

- 先使用已有 hard_pool 样本；
- 不足部分 fallback 到随机 drone；
- 每个 epoch 结束打印 fallback 汇总，不会每个 batch 刷屏。

进入 `identity_hard` 时，如果 hard_pool 尚未构建或加载，会直接报错，提示检查：

- `--build_hard_pool_epoch <= --stage2_end_epoch`
- 或使用 `--load_hard_pool_path`

## 4. Loss 设计

### 4.1 Sample4Geo loss

文件：[src/loss/blocks_infoNCE.py](../src/loss/blocks_infoNCE.py)

类：`infonce`

计算方式：

- 输入 satellite features 和 drone features；
- 先 L2 normalize；
- 构造 `B x B` 相似度矩阵；
- 对 `drone -> satellite` 和 `satellite -> drone` 分别做 cross entropy；
- 两个方向平均。

默认：

```text
loss = infonce_weight * symmetric_InfoNCE
```

当前还保留旧的同域 triplet 权重 `triplet_weight`，默认是 `0.0`，因此默认 Sample4Geo 行为仍是纯 InfoNCE。

### 4.2 Cross-domain identity contrast loss

文件：[src/loss/identity_losses.py](../src/loss/identity_losses.py)

类：`CrossDomainIdentityContrastiveLoss`

输入：

```text
feats:     [N, D]
labels:    [N]
view_type: [N]
```

规则：

- 所有 feature 先 L2 normalize；
- drone anchor 只和 satellite candidates 对比；
- satellite anchor 只和 drone candidates 对比；
- same label 是 positive；
- different label 是 negative；
- 支持一个 anchor 对多个 positive；
- 如果某个 anchor 没有 positive，则跳过；
- 最终对有效方向取平均。

权重：

```text
identity_loss_weight
```

### 4.3 Same-domain batch-hard triplet loss

文件：[src/loss/identity_losses.py](../src/loss/identity_losses.py)

类：`SameDomainBatchHardTripletLoss`

规则：

- 分别在 drone 域和 satellite 域内部计算；
- same label 为 positive；
- different label 为 negative；
- 对每个 anchor 取 hardest positive 和 hardest negative；
- 如果某个 domain 内没有有效 positive 或 negative，则跳过该 domain；
- 避免 NaN，返回可反传的 zero loss。

权重：

```text
same_domain_triplet_weight
```

margin：

```text
triplet_margin
```

### 4.4 Weak Sample4Geo anchor loss

文件：[src/loss/identity_losses.py](../src/loss/identity_losses.py)

类：`WeakSample4GeoAnchorLoss`

规则：

- 在 identity batch 内，每个 ID 生成一个 drone anchor 和一个 satellite anchor；
- anchor 可以取第一张，也可以取同 ID 特征平均；
- 构造 `P x P` 相似度矩阵；
- 计算 symmetric InfoNCE；
- 如果可用 ID 数不足 2，则跳过。

权重：

```text
weak_sample4geo_weight
```

anchor 表示：

```text
s4g_anchor_repr = first 或 mean
```

### 4.5 Identity 总 loss

`identity` 和 `identity_hard` 使用相同 loss 组合：

```text
loss =
    identity_loss_weight * cross_id_loss
  + same_domain_triplet_weight * same_triplet_loss
  + weak_sample4geo_weight * weak_sample4geo_loss
```

训练日志中会分别记录：

- `cross_id`
- `same_triplet`
- `weak_s4g`

日志中记录的是加权后的 loss contribution。

## 5. Hard Pool 构建

hard_pool 用于找出每个 ID 中更容易和其它 ID 混淆的 drone 图片。

触发时机：

```text
epoch == build_hard_pool_epoch
```

并且必须满足：

```text
enable_identity_stage=True
enable_hard_pool_stage=True
没有通过 load_hard_pool_path 预先加载
```

构建发生在该 epoch 的 eval 之后。

### 5.1 特征提取

构建 hard_pool 时：

1. `model.eval()`
2. `torch.no_grad()`
3. 如果 `use_ema_for_hard_pool=True`，临时应用 EMA 权重；
4. 提取训练集所有 satellite 和 drone feature；
5. 构建完成后恢复原训练状态。

### 5.2 Satellite prototype

对每个 ID，计算 satellite prototype：

```text
satellite_proto[pid] = mean(satellite_features_of_pid)
satellite_proto[pid] = L2Normalize(satellite_proto[pid])
```

### 5.3 Drone boundary risk

对每张 drone 图：

```text
pos_sim = sim(drone_feat, satellite_proto[same_pid])
neg_sims = sim(drone_feat, satellite_proto[other_pid])
topk_neg_mean = mean(top K neg_sims)
boundary_risk = topk_neg_mean - pos_sim
```

如果 `boundary_risk` 越高，说明这张 drone 更接近其它 ID 的 satellite prototype，同时远离自身 ID 的 satellite prototype，因此更 hard。

### 5.4 hard_pool JSON

保存格式：

```json
{
  "meta": {
    "epoch": 30,
    "model_source": "ema",
    "hard_pool_topk": 12,
    "hard_pool_topneg_k": 10
  },
  "hard_pool": {
    "0001": [
      {
        "pid": "0001",
        "image_path": "...",
        "boundary_risk": 0.123,
        "pos_sim": 0.456,
        "topk_neg_mean": 0.579,
        "top1_neg_pid": "0032",
        "top1_neg_sim": 0.600
      }
    ]
  },
  "id_risk": {
    "0001": 0.118
  }
}
```

其中：

```text
id_risk[pid] = top 3 hard samples 的 boundary_risk 平均值
```

构建完成后会打印：

- hard_pool 覆盖 ID 数；
- 每个 ID 平均 hard 样本数；
- risk mean / max / min；
- top 10 hardest IDs；
- hard_pool 保存路径。

## 6. Teacher 模型微调结构

文件：[src/models/teacher_model.py](../src/models/teacher_model.py)

Teacher 使用 DINOv3 backbone。初始化时：

1. 先冻结整个 backbone；
2. 在指定中间 blocks 注入 LoRA；
3. 对最后若干 blocks 开启 full fine-tune；
4. 从指定 `local_feature_layers` 取 patch tokens；
5. 用 local cross-attention 得到 local feature；
6. 将 global feature 与 local feature 融合；
7. 输出归一化后的特征。

### 6.1 默认三段式微调

默认逻辑：

```text
bottom blocks: frozen
middle blocks: LoRA
last 4 blocks: full fine-tune
```

相关参数：

- `lora_start_block`
- `lora_end_block`
- `full_finetune_start_block`
- `full_finetune_end_block`

如果没有手动指定：

- `full_finetune_start_block` 默认解析为 `-4`，即最后 4 个 blocks；
- `full_finetune_end_block` 默认是模型总 block 数；
- `lora_start_block` 默认是 `min(20, full_start)`；
- `lora_end_block` 默认是 `full_start`。

### 6.2 Local / PYRA feature fusion

默认：

```text
local_feature_layers = 19,27,36
```

Teacher 会从这些 block 取 patch tokens，通过 `PYRALocalCrossAttention` 得到 local feature，再与最后层 CLS global feature 融合。

融合方式：

```text
fused_feat = global_feat + gamma * local_feat
```

其中：

```text
gamma = 0.05 * sigmoid(gamma_raw)
```

如果启用 `use_soft_orth_fusion`，local feature 会先做 soft orthogonal projection，再参与融合。

## 7. Optimizer 参数分组

文件：[src/utils/optimizer_and_scale.py](../src/utils/optimizer_and_scale.py)

Teacher optimizer 会按参数类型分组：

| 参数组 | 学习率 | weight decay |
|---|---:|---:|
| LoRA decay params | `lr` | `0.01` |
| LoRA no-decay params | `lr` | `0.0` |
| full fine-tune backbone decay params | `lr * full_finetune_lr_mult` | `0.01` |
| full fine-tune backbone no-decay params | `lr * full_finetune_lr_mult` | `0.0` |
| local/fusion decay params | `lr` | `0.01` |
| local/fusion no-decay params | `lr` | `0.0` |
| logit_scale | `lr * logit_scale_lr_mult` | `0.0` |

如果安装了 DeepSpeed CPU Adam，则使用 `DeepSpeedCPUAdam`；否则使用 PyTorch `AdamW`。

## 8. 验证、EMA 与保存文件

每个 epoch 结束都会运行 validation，并使用 EMA 权重：

1. `ema.apply_shadow()`
2. `model.eval()`
3. 计算 D2S / S2D recall 和 mAP；
4. `ema.restore()`
5. 恢复训练状态。

保存目录由 `get_save_pth(args)` 决定。当前默认根目录是：

```text
src/checkpoint/teacher
```

每次训练会保存：

| 文件 | 作用 |
|---|---|
| `hyperparameters.json` | 当前训练命令和全部超参 |
| `best_metrics.json` | 最佳指标 + 每次验证历史 |
| `best_model.pth` | 当前最佳 EMA trainable 权重 |
| `final_model.pth` | 最后一个 epoch 的 EMA trainable 权重 |
| `hard_pool_epoch{epoch}.json` | hard_pool 文件，路径由 `save_hard_pool_path` 控制 |

最佳模型选择指标：

```text
D2S_R@1 + S2D_R@1
```

## 9. 全部命令行超参

### 9.1 基础训练参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--epochs`, `--max_epochs` | `22` | 总训练 epoch 数；两个名字等价，内部映射为 `args.epochs` |
| `--device` | `cuda` | 训练设备 |
| `--local_rank` | `0` | 分布式训练本地 rank，通常由 DeepSpeed 注入 |
| `--deepspeed_config` | `ds_config.json` | DeepSpeed 配置文件路径 |
| `--grad_accum_steps`, `--gradient_accumulation_steps` | `1` | 梯度累积步数 |

### 9.2 学习率与调度

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--lr` | `1e-4` | 主学习率 |
| `--scheduler` | `cosine` | 学习率调度器类型 |
| `--warmup_ratio` | `0.05` | warmup step 占总 optimizer step 的比例 |
| `--lr_end` | `1e-5` | 调度器末端学习率 |
| `--ema_decay` | `0.999` | EMA 衰减系数 |

### 9.3 数据参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--batch_size` | `4` | 每 GPU batch size；Sample4Geo 下表示 PID pair 数，identity 下表示每卡 ID 数 |
| `--img_size` | `224` | 输入图像尺寸 |
| `--data_dir` | `data/U1652` | 数据集根目录 |
| `--seed` | `0` | sampler 随机种子 |
| `--prob_flip` | `0.5` | Sample4Geo pair-level 水平翻转概率 |
| `--log_interval` | `20` | 训练日志打印间隔，按 batch 计 |
| `--num_workers` | `4` | DataLoader worker 数 |

### 9.4 多阶段训练参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--stage1_end_epoch` | `10` | stage 1 结束 epoch |
| `--stage2_end_epoch` | `30` | stage 2 结束 epoch |
| `--build_hard_pool_epoch` | `30` | hard_pool 构建 epoch |
| `--enable_identity_stage` | `False` | 启用 identity 阶段 |
| `--enable_hard_pool_stage` | `False` | 启用 identity_hard 阶段 |

### 9.5 Identity sampler 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--identity_ids_per_batch` | `8` | identity / identity_hard 每卡每 batch 的 ID 数 |
| `--identity_drone_per_id` | `4` | identity mode 每个 ID 随机 drone 数 |
| `--identity_sat_per_id` | `1` | 每个 ID 的 satellite 数 |
| `--hard_drone_per_id` | `2` | identity_hard 每个 ID 的 hard drone 数 |
| `--random_drone_per_id` | `2` | identity_hard 每个 ID 的 random drone 数 |

### 9.6 Hard pool 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--hard_pool_topk` | `12` | 每个 ID 保留 top-K hard drone |
| `--hard_pool_topneg_k` | `10` | 计算 boundary risk 时使用 top-K negative prototype |
| `--use_ema_for_hard_pool` | `True` | 构建 hard_pool 时是否使用 EMA 权重 |
| `--save_hard_pool_path` | `outputs/hard_pool_epoch{epoch}.json` | hard_pool 保存路径模板 |
| `--load_hard_pool_path` | `None` | 加载已有 hard_pool JSON |

### 9.7 LoRA / full fine-tune 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--lora_start_block` | `None` | LoRA 起始 block，默认解析为 `min(20, full_start)` |
| `--lora_end_block` | `None` | LoRA 结束 block，左闭右开，默认等于 full fine-tune 起点 |
| `--full_finetune_start_block` | `None` | full fine-tune 起始 block，默认 `-4` |
| `--full_finetune_end_block` | `None` | full fine-tune 结束 block，默认模型总 block 数 |
| `--full_finetune_lr_mult` | `0.1` | full fine-tune backbone 学习率倍率 |
| `--logit_scale_lr_mult` | `1.0` | logit_scale 学习率倍率 |
| `--lora_rank` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha |
| `--lora_dropout` | `0.1` | LoRA dropout |
| `--lora_target_names` | `qkv,proj` | 注入 LoRA 的 Linear 名称关键词 |

### 9.8 Feature fusion 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--local_feature_layers` | `19,27,36` | 取 local patch tokens 的 block index，0-based |
| `--use_soft_orth_fusion` | `False` | 启用 learnable soft orthogonal fusion |
| `--soft_orth_lambda_init` | `0.8` | soft orthogonal projection 强度的 sigmoid 初始化值 |
| `--soft_orth_detach_global` | `True` | projection 中 global feature 是否 detach |

### 9.9 Loss 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--triplet_weight` | `0.0` | Sample4Geo 阶段旧同域 triplet 权重，默认关闭 |
| `--infonce_weight` | `1.0` | Sample4Geo symmetric InfoNCE 权重 |
| `--identity_loss_weight` | `1.0` | cross-domain identity contrast loss 权重 |
| `--same_domain_triplet_weight` | `0.2` | same-domain batch-hard triplet loss 权重 |
| `--weak_sample4geo_weight` | `0.2` | weak Sample4Geo anchor loss 权重 |
| `--triplet_margin` | `0.3` | triplet margin |
| `--identity_temperature` | `0.07` | identity contrast / weak Sample4Geo temperature |
| `--s4g_anchor_repr` | `mean` | weak Sample4Geo anchor 表示，可选 `first` 或 `mean` |

## 10. 推荐命令

### 10.1 只训练原始 Sample4Geo teacher

```bash
deepspeed --num_gpus=1 src/training/teacher_train.py \
  --epochs 22 \
  --batch_size 4 \
  --infonce_weight 1.0 \
  --triplet_weight 0.0
```

### 10.2 两阶段训练：Sample4Geo + Identity

```bash
deepspeed --num_gpus=1 src/training/teacher_train.py \
  --enable_identity_stage \
  --epochs 30 \
  --stage1_end_epoch 10 \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --identity_loss_weight 1.0 \
  --same_domain_triplet_weight 0.2 \
  --weak_sample4geo_weight 0.2
```

### 10.3 三阶段训练：Sample4Geo + Identity + Identity Hard

```bash
deepspeed --num_gpus=1 src/training/teacher_train.py \
  --enable_identity_stage \
  --enable_hard_pool_stage \
  --epochs 40 \
  --stage1_end_epoch 10 \
  --stage2_end_epoch 30 \
  --build_hard_pool_epoch 30 \
  --identity_ids_per_batch 8 \
  --identity_drone_per_id 4 \
  --identity_sat_per_id 1 \
  --hard_drone_per_id 2 \
  --random_drone_per_id 2 \
  --hard_pool_topk 12 \
  --hard_pool_topneg_k 10
```

### 10.4 使用已有 hard_pool 继续训练

```bash
deepspeed --num_gpus=1 src/training/teacher_train.py \
  --enable_identity_stage \
  --enable_hard_pool_stage \
  --epochs 40 \
  --stage1_end_epoch 0 \
  --stage2_end_epoch 0 \
  --load_hard_pool_path outputs/hard_pool_epoch30.json
```

该命令会从第 1 个 epoch 开始进入 `identity_hard`。如果不希望跳过前两个阶段，不要把 `stage1_end_epoch` 和 `stage2_end_epoch` 设为 `0`。

### 10.5 Recommended staged-restart curriculum

This is the preferred route when the Sample4Geo stage peaks before epoch 10. Each stage starts a new run, reloads the previous stage `best_model.pth`, and rebuilds optimizer/scheduler from scratch.

Step 1: Sample4Geo + InfoNCE for 10 epochs.

```bash
deepspeed --include localhost:0,1 src/training/teacher_train.py \
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

Step 2: identity-only continuation from the Sample4Geo best checkpoint.

```bash
deepspeed --include localhost:2,3 src/training/teacher_train.py \
  --epochs 20 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<sample4geo_run>/best_model.pth \
  --enable_identity_stage \
  --stage1_end_epoch 0 \
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

Step 3: hard-pool-only continuation from the identity best checkpoint.

```bash
deepspeed --include localhost:4,5 src/training/teacher_train.py \
  --epochs 10 \
  --device cuda \
  --deepspeed_config ds_config.json \
  --data_dir data/U1652 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --init_checkpoint src/checkpoint/teacher/<identity_run>/best_model.pth \
  --enable_identity_stage \
  --enable_hard_pool_stage \
  --stage1_end_epoch 0 \
  --stage2_end_epoch 0 \
  --build_hard_pool_before_train \
  --build_hard_pool_epoch 0 \
  --save_hard_pool_path outputs/hard_pool_from_<identity_run>_epoch{epoch}.json \
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

Notes:

- `--init_checkpoint` loads the previous stage's trainable EMA weights after the original DINOv3 pretrained backbone is constructed.
- Keep architecture flags identical across stages unless this is an intentional ablation.
- `--build_hard_pool_before_train` is required when `stage1_end_epoch=0` and `stage2_end_epoch=0`, because epoch 1 starts directly in `identity_hard`.

## 11. 最小检查方法

### 11.1 检查代码能否编译

```bash
python -m py_compile \
  src/training/teacher_train.py \
  src/dataset/datasets.py \
  src/loss/identity_losses.py
```

### 11.2 检查 identity loss

```bash
python - <<'PY'
import torch
from src.loss.identity_losses import (
    CrossDomainIdentityContrastiveLoss,
    SameDomainBatchHardTripletLoss,
    WeakSample4GeoAnchorLoss,
)

feats = torch.randn(15, 32, dtype=torch.bfloat16).requires_grad_(True)
labels = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2])
views = torch.tensor([0,1,1,1,1, 0,1,1,1,1, 0,1,1,1,1])

losses = [
    CrossDomainIdentityContrastiveLoss()(feats, labels, views),
    SameDomainBatchHardTripletLoss()(feats, labels, views),
    WeakSample4GeoAnchorLoss()(feats, labels, views),
]
total = sum(losses)
total.backward()
print([float(x.detach()) for x in losses])
print(torch.isfinite(total).item(), feats.grad is not None)
PY
```

### 11.3 检查 hard_pool 构建

```bash
deepspeed --num_gpus=1 src/training/teacher_train.py \
  --enable_identity_stage \
  --enable_hard_pool_stage \
  --epochs 1 \
  --stage1_end_epoch 1 \
  --stage2_end_epoch 1 \
  --build_hard_pool_epoch 1 \
  --hard_pool_topk 2 \
  --hard_pool_topneg_k 2 \
  --num_workers 0 \
  --save_hard_pool_path "outputs/test_hard_pool_epoch{epoch}.json"
```

检查 JSON：

```bash
python - <<'PY'
import json
p = json.load(open("outputs/test_hard_pool_epoch1.json", encoding="utf-8"))
print(p.keys())
pid = next(iter(p["hard_pool"]))
print(pid, p["hard_pool"][pid][0])
PY
```

## 12. 常见注意事项

1. `identity_hard` 必须有 hard_pool。可以在训练中构建，也可以通过 `--load_hard_pool_path` 加载。
2. 如果 `--enable_hard_pool_stage=True`，建议保证：

```text
build_hard_pool_epoch <= stage2_end_epoch
```

否则进入 `identity_hard` 时可能还没有 hard_pool。

3. `identity_hard` 的 loss 和 `identity` 完全相同，差异只在 dataloader 的 drone 采样方式。
4. `hard_pool_topneg_k` 不能超过实际 negative ID 数；代码中会自动取 `min(hard_pool_topneg_k, num_ids - 1)`。
5. `weak_sample4geo_weight=0` 时不会计算 weak Sample4Geo anchor loss。
6. 所有 identity loss 在计算相似度前都会 L2 normalize feature，并在 bf16 训练下转 float 计算 loss，避免 dtype/device mismatch。
7. `best_metrics.json` 中保存最佳结果和每次 validation history，`best_model.pth` 保存最佳 EMA trainable 权重。
