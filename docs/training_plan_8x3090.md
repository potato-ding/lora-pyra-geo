# RepViT-M1.5 学生 Baseline 训练手册

更新日期：2026-06-19

本文档是当前仓库唯一维护的学生训练说明，内容以现有代码为准。当前学生训练处于
干净 baseline 状态：

```text
模型：RepViT-M1.5 -> f4 -> GAP -> BN -> L2
损失：双向 symmetric InfoNCE
采样：Sample4Geo paired sampling
多卡：DeepSpeed + 可微 all-gather
教师：默认不加载
```

普通 similarity KD 只保留命令行入口，尚未实现。

---

## 1. 代码入口与文件职责

| 文件 | 职责 |
|---|---|
| `src/training/student_train.py` | 单卡与 DeepSpeed 多卡训练入口 |
| `src/models/student_model.py` | RepViT-M1.5 学生模型 |
| `src/models/repvit_backbone.py` | RepViT 预训练权重加载与多阶段特征提取 |
| `src/loss/blocks_infoNCE.py` | 学生 symmetric InfoNCE |
| `src/dataset/datasets.py` | 学生 pair dataset 与 dataloader |
| `src/dataset/teacher/datasets.py` | `Sample4GeoBatchSampler` 的共享实现 |
| `src/dataset/transforms.py` | 训练与测试图像增强 |
| `src/utils/optimizer_and_scale.py` | 学生 AdamW 参数分组 |
| `src/utils/scheduler.py` | warmup + cosine 学习率调度 |
| `configs/ds_student_baseline.json` | DeepSpeed baseline 配置 |
| `src/training/student_test.py` | U1652、GTA-UAV、SUES-200 统一测试入口 |

---

## 2. 学生模型

### 2.1 前向结构

```text
输入图像 [B, 3, H, W]
  -> RepViT-M1.5 feature extractor
  -> f4 [B, 512, h, w]
  -> AdaptiveAvgPool2d(1)
  -> flatten
  -> BatchNorm1d(512)
  -> L2 normalize
  -> embedding [B, 512]
```

数学形式：

\[
f_4 = \operatorname{RepViT}(x)
\]

\[
z = \operatorname{L2Norm}
\left(
\operatorname{BN}
\left(
\operatorname{GAP}(f_4)
\right)
\right)
\]

模型没有额外分类头、置信度头、视角适配器或局部匹配分支。

### 2.2 RepViT 初始化权重

默认加载：

```text
src/models/repvit/repvit_m1_5_distill_450e.pth
```

启动前检查：

```bash
ls -lh src/models/repvit/repvit_m1_5_distill_450e.pth
```

加载器会打印：

```text
matched keys
matched feature keys
feature load ratio
missing keys
unexpected keys
```

feature 权重匹配率低于 95% 时会直接报错，避免误用错误 checkpoint。

### 2.3 相似度温度

模型维护可学习参数 `logit_scale`：

\[
\operatorname{logit\_scale}_{0}
=
\log\left(\frac{1}{T}\right)
\]

默认：

```text
temperature = 0.07
initial exp(logit_scale) ≈ 14.2857
```

每次更新后，代码将 `logit_scale` 限制在：

```text
0 <= logit_scale <= log(100)
1 <= exp(logit_scale) <= 100
```

---

## 3. 训练数据

### 3.1 University-1652 目录结构

学生训练默认读取：

```text
data/U1652/train/
├── satellite/
│   ├── 0001/
│   │   └── *.jpg
│   ├── 0002/
│   └── ...
└── drone/
    ├── 0001/
    │   └── *.jpg
    ├── 0002/
    └── ...
```

验证默认读取：

```text
data/U1652/
```

### 3.2 Pair 构建规则

`U1652PairDataset` 对每个 PID：

1. 按文件名排序 satellite 图片；
2. 固定取第一张 satellite；
3. 将该 satellite 分别与 PID 下所有 drone 图片组成 pair。

每个样本固定返回：

```python
(drone_tensor, satellite_tensor, label, pid)
```

不会返回图片路径或额外样本标识，也不会读取教师缓存或额外采样文件。

### 3.3 图像增强

训练输入统一 resize 到：

```text
img_size × img_size
```

默认 `img_size=224`，归一化参数为 ImageNet mean/std。

共同增强包括：

- ColorJitter；
- 平移与缩放；
- Hue/Saturation/Value 扰动；
- AdvancedBlur；
- GridDropout；
- CoarseDropout；
- JPEG compression；
- 同一 pair 同步执行概率为 0.5 的水平翻转。

satellite 额外执行：

```text
RandomRotate90(p=1.0)
```

drone 不执行该 90 度旋转。

验证只执行 resize、normalize 和 tensor 转换。

---

## 4. Baseline sampler

### 4.1 采样目标

InfoNCE 将 batch 中非对角样本全部视为负样本，因此一个训练 batch 内不能出现重复
PID。否则同 ID 样本会被错误地作为负样本。

### 4.2 单卡

单卡训练每个 epoch 调用 dataset 的 paired shuffle：

- 随机打乱所有 pair；
- 每个 batch 内 PID 唯一；
- 只保留完整的 PID-unique batch；
- 同一个 pair 在一个 epoch 内不重复使用。

### 4.3 多卡

多卡使用 `Sample4GeoBatchSampler`：

1. 先构造全局 batch；
2. 保证全局 batch 内 PID 唯一；
3. 再按 rank 切分为每卡 local batch；
4. 不同 rank 获得互不重叠的 pair index。

必要条件：

```text
batch_size × world_size <= 训练 PID 总数
```

否则 sampler 会直接报错。

---

## 5. Symmetric InfoNCE

设当前 global pair batch 为 \(N\)，归一化后的 drone 和 satellite 特征分别为：

\[
D \in \mathbb{R}^{N\times512},
\qquad
S \in \mathbb{R}^{N\times512}
\]

相似度矩阵：

\[
M = \exp(\text{logit\_scale}) D S^\top
\]

正样本标签固定为矩阵对角线：

\[
y_i=i
\]

双向损失：

\[
L_{\text{D2S}}
=
\operatorname{CE}(M,y)
\]

\[
L_{\text{S2D}}
=
\operatorname{CE}(M^\top,y)
\]

\[
L_{\text{InfoNCE}}
=
\frac{L_{\text{D2S}}+L_{\text{S2D}}}{2}
\]

当前总损失严格等于：

\[
L_{\text{total}}=L_{\text{InfoNCE}}
\]

默认 label smoothing：

```text
0.1
```

---

## 6. DeepSpeed 多卡机制

### 6.1 特征顺序

每张卡的模型输入顺序：

```text
[local_drone, local_satellite]
```

代码先拆分两个视角，再分别 all-gather，最终得到：

```text
[all_drone, all_satellite]
```

这样 global 相似度矩阵的对角线仍然是正确正样本。

### 6.2 梯度

学生特征使用带 autograd 的 `GatherLayer`：

```text
local features
  -> differentiable all-gather
  -> global gallery
  -> symmetric InfoNCE
  -> backward to every rank
```

不允许将学生 gather 特征 detach。

### 6.3 ZeRO 支持

当前支持：

```text
ZeRO stage 0
ZeRO stage 1
ZeRO stage 2
```

当前不支持 ZeRO stage 3，原因是 model-only checkpoint 导出需要另一套参数合并路径。

### 6.4 精度设置

默认 DeepSpeed 配置：

```json
"bf16": {"enabled": true},
"fp16": {"enabled": false}
```

RTX 3090 属于 Ampere 架构，可以使用 BF16。传入 `--no_amp` 时，代码会同时关闭
DeepSpeed 的 BF16 和 FP16。

### 6.5 Rank 日志

DeepSpeed step 日志、epoch 日志、验证日志以及模型保存只由 rank0 执行。各 rank
仍共同参与训练、all-gather 和分布式验证。

---

## 7. Batch size 口径

`--batch_size` 表示每张 GPU 的 pair 数，不是图片数。

```text
local pair batch     = batch_size
local image batch    = 2 × batch_size
global pair batch    = batch_size × world_size
global image batch   = 2 × batch_size × world_size
optimizer pair batch = batch_size × world_size × grad_accum_steps
```

常用配置：

| GPU 数 | 每卡 pair | 每卡图片 | 单步 global pair | 单步 global 图片 |
|---:|---:|---:|---:|---:|
| 1 | 8 | 16 | 8 | 16 |
| 2 | 8 | 16 | 16 | 32 |
| 4 | 8 | 16 | 32 | 64 |
| 8 | 4 | 8 | 32 | 64 |
| 8 | 8 | 16 | 64 | 128 |

梯度累积只扩大 optimizer 有效 batch，不扩大单次 forward 的 InfoNCE gallery：

```text
InfoNCE gallery size = batch_size × world_size
```

例如：

```text
8 GPU × 4 pair/GPU × grad_accum_steps 2

单次 gallery       = 32 pair
optimizer 有效 batch = 64 pair
```

---

## 8. 优化器与学习率

### 8.1 AdamW

默认：

```text
optimizer = AdamW
lr = 1e-4
weight_decay = 1e-4
betas = (0.9, 0.999)
```

参数分组：

- 权重矩阵使用 weight decay；
- bias、BatchNorm、归一化层及一维参数不使用 weight decay。

### 8.2 Warmup + cosine

调度器按 iteration 更新：

```text
warmup -> cosine decay -> base_lr × min_lr_ratio
```

当前默认：

```text
warmup_epochs = 0.1
min_lr_ratio = 0.01
```

注意：`warmup_epochs=0.1` 表示 0.1 个 epoch，不是总 epoch 的 10%。

若训练前期不稳定，可尝试：

```text
--warmup_epochs 1
```

---

## 9. 推荐启动命令

### 9.1 启动前检查

```bash
python src/training/student_train.py --help
ls data/U1652/train/satellite
ls data/U1652/train/drone
ls src/models/repvit/repvit_m1_5_distill_450e.pth
```

### 9.2 8×RTX 3090 保守配置

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
  src/training/student_train.py \
  --deepspeed \
  --deepspeed_config configs/ds_student_baseline.json \
  --epochs 60 \
  --train_data_dir data/U1652/train \
  --val_data_dir data/U1652 \
  --img_size 224 \
  --batch_size 4 \
  --val_batch_size 32 \
  --num_workers 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --warmup_epochs 0.1 \
  --min_lr_ratio 0.01 \
  --temperature 0.07 \
  --label_smoothing 0.1 \
  --grad_accum_steps 1 \
  --print_freq 20 \
  --val_interval 5
```

该配置的 batch 口径：

```text
local pair batch  = 4
local image batch = 8
global pair batch = 32
global image batch = 64
```

### 9.3 8×RTX 3090 扩大 gallery

显存允许时：

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
  src/training/student_train.py \
  --deepspeed \
  --deepspeed_config configs/ds_student_baseline.json \
  --batch_size 8 \
  --grad_accum_steps 1
```

此时单步 global gallery 为 64 pair。

### 9.4 4 卡

```bash
deepspeed --include localhost:0,1,2,3 \
  src/training/student_train.py \
  --deepspeed \
  --deepspeed_config configs/ds_student_baseline.json \
  --epochs 60 \
  --batch_size 8 \
  --val_batch_size 32 \
  --num_workers 8
```

### 9.5 单卡

```bash
python src/training/student_train.py \
  --epochs 60 \
  --train_data_dir data/U1652/train \
  --val_data_dir data/U1652 \
  --img_size 224 \
  --batch_size 8 \
  --val_batch_size 32 \
  --num_workers 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --temperature 0.07 \
  --label_smoothing 0.1 \
  --val_interval 5
```

### 9.6 指定输出目录

默认输出到：

```text
src/checkpoint/student/<YYYYMMDD_HHMMSS>/
```

也可以显式指定：

```bash
python src/training/student_train.py \
  --output_dir src/checkpoint/student/baseline_seed0
```

---

## 10. 参数表

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--train_data_dir` | `data/U1652/train` | 训练集目录 |
| `--val_data_dir` | `data/U1652` | U1652 验证集目录 |
| `--output_root` | `src/checkpoint/student` | 自动创建 run 目录的根目录 |
| `--output_dir` | `None` | 显式输出目录 |
| `--deepspeed` | `False` | 启用 DeepSpeed；多进程启动时也会自动进入该路径 |
| `--deepspeed_config` | `configs/ds_student_baseline.json` | DeepSpeed 配置 |
| `--grad_accum_steps` | `1` | 梯度累积步数 |
| `--seed` | `0` | 每个 rank 使用 `seed + rank` |
| `--epochs` | `60` | 训练 epoch 数 |
| `--img_size` | `224` | 输入高宽 |
| `--batch_size` | `8` | 每卡 pair 数 |
| `--val_batch_size` | `32` | 验证图片 batch |
| `--num_workers` | `8` | dataloader worker 数 |
| `--lr` | `1e-4` | AdamW 基础学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_epochs` | `0.1` | warmup epoch 数 |
| `--min_lr_ratio` | `0.01` | 最终学习率与基础学习率比例 |
| `--temperature` | `0.07` | `logit_scale` 初始化温度 |
| `--label_smoothing` | `0.1` | InfoNCE 交叉熵 smoothing |
| `--amp` | `True` | 单卡 AMP；DeepSpeed 精度由 runtime config 同步控制 |
| `--no_amp` | - | 关闭 AMP/BF16/FP16 |
| `--grad_clip` | `0.0` | 梯度裁剪；0 表示关闭 |
| `--print_freq` | `20` | step 日志间隔 |
| `--val_interval` | `5` | 验证间隔；最后一个 epoch 总会验证 |
| `--best_metric_name` | `R1_sum` | 代码会强制使用 `R1_sum` |
| `--save_last` | `True` | 保存 last checkpoint |
| `--no_save_last` | - | 禁止保存 last checkpoint |
| `--use_kd_distill` | `False` | 预留入口，当前启用会报未实现错误 |

---

## 11. 日志说明

### 11.1 Step 日志

单卡：

```text
pair_batch
global_pair_batch
data time
batch time
loss_infonce
logit_scale
learning rate
```

DeepSpeed：

```text
local_pair_batch
global_pair_batch
data time
batch time
loss_infonce
logit_scale
learning rate
```

### 11.2 正常启动检查

建议确认启动日志包含：

```text
[RepViTBackbone] matched feature keys: ... (>=95%)
[Params] total=... | trainable=... | frozen=...
[Optimizer] student ...
[Scheduler] ...
[DeepSpeedBatch] ...
```

并确认：

```text
global_pair_batch = local_pair_batch × world_size
```

---

## 12. 验证与 checkpoint

### 12.1 验证方向

```text
D2S: query_drone -> gallery_satellite
S2D: query_satellite -> gallery_drone
```

每个方向记录：

```text
R@1
R@5
R@10
mAP
```

best 选择指标：

\[
\text{R1\_sum}
=
\text{D2S R@1}
+
\text{S2D R@1}
\]

### 12.2 输出文件

```text
<run_dir>/
├── best_model.pth
├── last_model.pth
├── best_metrics.json
├── training_record.txt
└── deepspeed/
    └── last/
```

说明：

- `best_model.pth`：验证指标最优权重；
- `last_model.pth`：最后保存的训练状态或模型权重；
- `best_metrics.json`：best 指标与验证历史；
- `training_record.txt`：命令行、参数、最后验证结果；
- `deepspeed/last/`：DeepSpeed optimizer、scheduler 与恢复状态。

单卡 checkpoint 包含 model、optimizer、scheduler；DeepSpeed 导出的
`best_model.pth`/`last_model.pth` 是便于测试的 model-only checkpoint。

---

## 13. 模型测试

### 13.1 University-1652

`--checkpoint` 可以传 run 目录，也可以直接传 `.pth` 文件。传目录时优先加载
`best_model.pth`。

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<run_dir> \
  --dataset 1652 \
  --data_dir data/U1652 \
  --batch_size 32 \
  --img_size 224 \
  --num_workers 8
```

### 13.2 GTA-UAV

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<run_dir> \
  --dataset GTA-UAV \
  --data_dir data/GTA-UAV-LR/GTA-UAV-LR-baidu \
  --gta_split cross-area \
  --gta_query_mode both \
  --batch_size 32
```

### 13.3 SUES-200

```bash
python src/training/student_test.py \
  --checkpoint src/checkpoint/student/<run_dir> \
  --dataset SUES-200 \
  --data_dir data/SUES-200/SUES-200-512x512 \
  --sues_height all \
  --batch_size 32
```

默认结果保存到 checkpoint 同目录：

```text
student_test_results.json
```

---

## 14. 测试与代码审计

### 14.1 完整测试

```bash
python -m pytest -q
```

当前基准：

```text
35 passed
```

### 14.2 学生定向测试

```bash
python -m pytest tests/test_student_baseline.py -q
```

### 14.3 多卡 gather 冒烟测试

```bash
torchrun --standalone --nproc_per_node=2 \
  tests/check_student_distributed_gather.py
```

成功输出：

```text
student distributed paired gather smoke test passed
```

该测试同时检查：

- global 排列为 `[all_drone, all_satellite]`；
- global pair batch 正确；
- gather 后梯度能回传到每个 rank。

### 14.4 中间特征检查

```bash
python tests/check_student_intermediate_shapes.py
```

检查 `f4` 通道数与最终 512 维 descriptor。

---

## 15. 常见问题

### 15.1 `unrecognized arguments`

当前训练入口只接受参数表中的参数。历史实验参数已不属于当前 baseline，继续传入会被
`argparse` 拒绝。

先执行：

```bash
python src/training/student_train.py --help
```

### 15.2 启用 `--use_kd_distill` 后报错

这是预期行为。普通 similarity KD 尚未实现，当前参数仅用于锁定后续接口名称。

baseline 训练不要传：

```text
--use_kd_distill
```

### 15.3 找不到 RepViT 权重

确认文件存在：

```bash
ls src/models/repvit/repvit_m1_5_distill_450e.pth
```

### 15.4 全局 batch 大于 PID 数

报错形式：

```text
global_batch_size is larger than PID count
```

降低：

```text
batch_size
```

或减少训练 GPU 数。

### 15.5 DeepSpeed 单进程启动报错

不要只执行：

```bash
python src/training/student_train.py --deepspeed
```

应使用 DeepSpeed launcher：

```bash
deepspeed --include localhost:0,1,2,3 \
  src/training/student_train.py --deepspeed
```

### 15.6 CUDA OOM

按以下顺序处理：

1. 降低 `--batch_size`；
2. 降低 `--val_batch_size`；
3. 增加 `--grad_accum_steps` 恢复 optimizer 有效 batch；
4. 必要时降低 `--img_size`。

注意：增加梯度累积不能恢复被降低的单步 InfoNCE gallery 大小。

### 15.7 DataLoader 很慢

依次尝试：

- 检查数据是否位于机械硬盘或网络盘；
- 调整 `--num_workers`；
- 检查 CPU、内存和磁盘占用；
- 确认 OpenCV 与 Albumentations 安装正常。

### 15.8 RepViT registry warning

测试或启动时可能出现：

```text
Overwriting repvit_m1_5 in registry
```

这是当前模型注册模块重复注册名称的 warning，不会中断训练。真正需要关注的是后续是否
出现 traceback、权重加载率不足或 CUDA 错误。

---

## 16. 实验记录建议

每次训练至少记录：

| 项目 | 示例 |
|---|---|
| run 名称 | `baseline_8x3090_b4_seed0` |
| 日期 | `2026-06-19` |
| GPU | `8×RTX 3090` |
| global pair batch | `32` |
| optimizer pair batch | `32` |
| 分辨率 | `224` |
| epochs | `60` |
| lr | `1e-4` |
| warmup | `0.1 epoch` |
| temperature | `0.07` |
| label smoothing | `0.1` |
| best epoch | 训练后填写 |
| D2S R@1/mAP | 训练后填写 |
| S2D R@1/mAP | 训练后填写 |
| R1_sum | 训练后填写 |

建议输出目录使用可读名称：

```bash
--output_dir src/checkpoint/student/baseline_8x3090_b4_seed0
```

---

## 17. 普通 similarity KD 后续约束

未来实现普通 similarity KD 时，应保持以下边界：

1. 只有 `--use_kd_distill` 启用时才加载教师；
2. 教师 forward 使用 inference/no-grad；
3. 教师特征不参与梯度；
4. 学生特征及跨卡 gather 保留梯度；
5. `loss_kd_raw` 与 `loss_kd_weighted` 分开记录；
6. 总损失明确写为 `InfoNCE + kd_weight × KD`；
7. 默认关闭 KD 时，模型、sampler、loss、日志和 checkpoint 路径必须与本文 baseline
   完全一致。

当前代码尚未实现上述 KD，现阶段唯一可训练配置仍是纯 symmetric InfoNCE baseline。
