# RepViT-M1.5 学生训练与 Plain Distillation 手册

更新日期：2026-06-20

本文档是当前仓库唯一维护的学生训练说明。当前学生训练只支持两种模式：

1. 纯 RepViT-M1.5 baseline；
2. 普通在线 teacher-student distillation（`distill_type=plain`）。

学生训练中不存在 BRD、风险感知蒸馏、边界负样本挖掘、margin 筛选、top-k
hard-negative 蒸馏或 no-boundary anchor 跳过逻辑。

---

## 1. 当前训练结构

### 1.1 Baseline

```text
image
  -> RepViT-M1.5
  -> f4
  -> global average pooling
  -> BatchNorm1d(512)
  -> L2 normalize
  -> 512-d retrieval embedding
```

Baseline loss：

```text
loss_retrieval = symmetric Sample4Geo InfoNCE
total_loss = loss_retrieval
```

不传 `--distill`，或显式传入 `--distill false` 时：

- 不加载 teacher；
- 不创建 feature projection head；
- dataset、student forward、sampler 和 optimizer 与纯 baseline 一致；
- loss 只有 symmetric InfoNCE；
- checkpoint 和验证使用的仍然是 512 维 retrieval embedding。

### 1.2 Plain distillation

启用：

```text
--distill true
--distill_type plain
```

后，每个 batch 同时计算：

```text
loss_retrieval
loss_kd_feat
loss_kd_sim
```

总损失为：

\[
L_{\text{total}}
=
L_{\text{retrieval}}
+
\lambda_{\text{feat}}L_{\text{kd-feat}}
+
\lambda_{\text{sim}}L_{\text{kd-sim}}
\]

对应代码参数：

```text
total_loss
  = loss_retrieval
  + kd_feat_weight * loss_kd_feat
  + kd_sim_weight * loss_kd_sim
```

当前没有 KD warmup，也没有动态风险权重。

---

## 2. 相关文件

| 文件 | 职责 |
|---|---|
| `src/training/student_train.py` | 单卡和 DeepSpeed 多卡训练、plain KD loss、日志 |
| `src/models/student_model.py` | RepViT-M1.5 学生和可选 feature projection |
| `src/models/teacher/model.py` | 在线 DINOv3 teacher |
| `src/training/teacher/evaluate.py` | teacher checkpoint 加载 |
| `src/loss/blocks_infoNCE.py` | symmetric Sample4Geo InfoNCE |
| `src/dataset/datasets.py` | baseline paired dataset 和 dataloader |
| `src/utils/optimizer_and_scale.py` | student AdamW 参数分组 |
| `src/utils/gather_features_and_labels_and_views.py` | 可微与无梯度 all-gather |
| `configs/ds_student_baseline.json` | DeepSpeed 配置 |
| `src/training/student_test.py` | U1652、GTA-UAV、SUES-200 测试入口 |

---

## 3. 数据与视角顺序

训练 dataset 返回：

```python
(drone_tensor, satellite_tensor, label, pid)
```

训练入口拼接为：

```text
images = cat([drone_imgs, satellite_imgs], dim=0)
```

学生和 teacher 使用完全相同的输入顺序：

```text
[local_drone, local_satellite]
```

多卡 gather 时不是直接 gather 混合 tensor，而是分别 gather 两个视角：

```text
global_drone
  = [rank0_drone, rank1_drone, ..., rankN_drone]

global_satellite
  = [rank0_satellite, rank1_satellite, ..., rankN_satellite]

global_features
  = [global_drone, global_satellite]
```

因此 teacher、student 和正样本对角线保持相同顺序。Plain KD 不需要额外
`view_type` mask，也不根据 label、margin 或负样本难度筛选样本。

---

## 4. Teacher 约束

Teacher 只在 `--distill true` 时加载。

加载后执行：

```python
teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad_(False)
```

在线 forward 使用：

```python
with torch.no_grad():
    teacher_features = teacher(images)
```

Teacher feature 随后执行：

```text
detach -> float32 -> L2 normalize
```

Teacher：

- 不传给 `deepspeed.initialize`；
- 不加入 student optimizer；
- 不接收梯度；
- 不写入 student checkpoint；
- 每个 rank 只 forward 当前 local image batch；
- 使用 `teacher_micro_batch_size` 分块 forward，降低显存占用。

---

## 5. Feature distillation

### 5.1 维度对齐

学生 retrieval embedding 固定为：

```text
student_dim = 512
```

当前 DINOv3 ViT-7B teacher 默认为：

```text
teacher_dim = 4096
```

当维度不一致且 `kd_feat_weight > 0` 时，学生模型增加：

```text
distill_projection = Linear(student_dim, teacher_dim, bias=False)
```

该 projection：

- 只用于 feature KD；
- 不改变 student retrieval forward 输出；
- 在创建 optimizer 和 DeepSpeed engine 之前创建；
- 参数自动加入 student optimizer；
- 随 student checkpoint 保存；
- 验证时会忽略该训练专用 projection，只使用原始 512 维 retrieval embedding。

如果 teacher checkpoint 的实际 feature dim 与 `--teacher_dim` 不一致，训练会直接报错。

### 5.2 Feature loss

投影后的 student feature 与 detached teacher feature 都执行 L2 normalize，然后使用
cosine distance：

\[
L_{\text{kd-feat}}
=
\frac{1}{2B}
\sum_i
\left(
1-\cos(z_i^S,z_i^T)
\right)
\]

这里同时包含所有 drone 和 satellite 样本，不使用任何筛选 mask。

---

## 6. Similarity distillation

全局 batch 拆分为：

```text
student_drone_feat
student_satellite_feat
teacher_drone_feat
teacher_satellite_feat
```

D2S similarity：

```python
student_sim_d2s = student_drone_feat @ student_satellite_feat.T
teacher_sim_d2s = teacher_drone_feat @ teacher_satellite_feat.T
```

S2D similarity：

```python
student_sim_s2d = student_sim_d2s.T
teacher_sim_s2d = teacher_sim_d2s.T
```

两个方向都使用完整的 `global_pair_batch × global_pair_batch` 矩阵：

\[
L_{\text{D2S}}
=
T^2
\operatorname{KL}
\left(
\operatorname{softmax}(S_T/T)
\parallel
\operatorname{softmax}(S_S/T)
\right)
\]

\[
L_{\text{S2D}}
=
T^2
\operatorname{KL}
\left(
\operatorname{softmax}(S_T^\top/T)
\parallel
\operatorname{softmax}(S_S^\top/T)
\right)
\]

\[
L_{\text{kd-sim}}
=
0.5L_{\text{D2S}}+0.5L_{\text{S2D}}
\]

Similarity KD：

- 使用完整正负相似度分布；
- 不只蒸馏 top-k negative；
- 不构造 boundary/risk/margin mask；
- 不跳过任何 anchor；
- teacher 与 student 的原始 feature dim 可以不同。

---

## 7. DeepSpeed 多卡行为

学生 retrieval feature 和 projection feature 使用可微 all-gather：

```text
local student features
  -> differentiable all-gather
  -> global feature matrix
  -> retrieval / KD loss
  -> backward to student and projection
```

Teacher feature 使用 detached all-gather：

```text
local teacher features
  -> detach
  -> no-grad all-gather
  -> global teacher targets
```

当前支持：

```text
ZeRO stage 0
ZeRO stage 1
ZeRO stage 2
```

不支持 ZeRO stage 3 的 model-only 导出路径。

DeepSpeed step、epoch、验证和 plain KD 日志只由 rank 0 打印。
模型结构、DINOv3/RepViT 权重加载、optimizer 和 scheduler 的启动日志同样只由
rank 0 打印。每个 rank 仍会独立构造本地 student/teacher，这是分布式训练的正常行为，
只是非 rank0 不再重复输出。

---

## 8. Optimizer

Optimizer 为 AdamW：

```text
lr = 1e-4
weight_decay = 1e-4
betas = (0.9, 0.999)
```

包含：

- RepViT-M1.5 student backbone；
- BatchNorm neck；
- retrieval `logit_scale`；
- `kd_feat_weight > 0` 且维度不一致时的 `distill_projection`。

不包含：

- teacher 的任何参数。

关闭 distillation 时不会创建 projection，因此 optimizer 参数集合与 baseline 一致。

---

## 9. 参数表

### 9.1 训练参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--train_data_dir` | `data/U1652/train` | 训练集目录 |
| `--val_data_dir` | `data/U1652` | 验证集目录 |
| `--epochs` | `60` | epoch 数 |
| `--batch_size` | `8` | 每张 GPU 的 pair 数 |
| `--val_batch_size` | `32` | 验证图片 batch |
| `--num_workers` | `8` | dataloader worker |
| `--lr` | `1e-4` | AdamW 学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--temperature` | `0.07` | retrieval InfoNCE 初始温度 |
| `--label_smoothing` | `0.1` | retrieval loss label smoothing |
| `--grad_accum_steps` | `1` | 梯度累积步数 |
| `--deepspeed_config` | `configs/ds_student_baseline.json` | DeepSpeed 配置 |

### 9.2 Plain distillation 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--distill` | `False` | 是否启用普通在线蒸馏；支持 `--distill true` |
| `--distill_type` | `plain` | 当前唯一支持的蒸馏类型 |
| `--kd_feat_weight` | `0.05` | feature cosine KD 权重 |
| `--kd_sim_weight` | `0.05` | similarity KL KD 权重 |
| `--kd_temperature` | `0.1` | similarity soft target 温度 |
| `--teacher_ckpt` | `None` | teacher checkpoint 文件或 run 目录 |
| `--teacher_arch` | `dinov3_vit7b16` | teacher 架构 |
| `--teacher_dim` | `4096` | teacher descriptor dim |
| `--student_dim` | `512` | student retrieval descriptor dim |
| `--teacher_precision` | `bf16` | `bf16`、`fp16` 或 `fp32` |
| `--teacher_micro_batch_size` | `1` | 每次 teacher forward 的图片数 |

以下旧参数已不再接受：

```text
--use_kd_distill
--kd_type
--teacher_checkpoint
--kd_weight
--kd_warmup_epochs
--kd_d2s_weight
--kd_s2d_weight
```

所有 BRD、risk、boundary 参数同样不属于当前学生训练接口。

---

## 10. 日志

Baseline step 日志：

```text
loss_retrieval
logit_scale
learning rate
local/global pair batch
data/batch time
```

Plain distillation 额外打印：

```text
loss_retrieval
loss_kd_feat
loss_kd_sim
total_loss
kd_feat_weight
kd_sim_weight
teacher_feat_norm_mean
student_feat_norm_mean
teacher_sim_mean
student_sim_mean
```

不打印 teacher margin、boundary anchor ratio、boundary negative count、risk weight 或
violation ratio。

Teacher checkpoint 是可训练参数增量文件。启动时会看到类似：

```text
[TeacherDelta] matched=125 | trainable_covered=125/125 |
missing_nontrainable=566 | unexpected=0 | incompatible=0
[TeacherDelta] coverage OK: all trainable teacher parameters were restored;
missing non-trainable keys keep their pretrained DINOv3/base initialization.
```

这里 `missing_nontrainable` 不是权重加载失败。它表示这些冻结参数不在增量 checkpoint
中，继续使用此前已经严格加载的官方 DINOv3 基座权重。真正需要关注的是：

```text
trainable_covered 必须完整
unexpected 必须为 0
incompatible 必须为 0
```

---

## 11. 推荐命令

### 11.1 纯 baseline

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
  src/training/student_train.py \
  --deepspeed \
  --deepspeed_config configs/ds_student_baseline.json \
  --epochs 60 \
  --train_data_dir data/U1652/train \
  --val_data_dir data/U1652 \
  --batch_size 4 \
  --val_batch_size 32 \
  --num_workers 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --temperature 0.07 \
  --label_smoothing 0.1
```

### 11.2 8 卡 plain distillation

下面使用已经存在的教师目录示例，不要在 Bash 命令中保留 `<teacher_run>` 形式的尖括号：

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
  src/training/student_train.py \
  --deepspeed \
  --deepspeed_config configs/ds_student_baseline.json \
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
  --distill true \
  --distill_type plain \
  --kd_feat_weight 0.05 \
  --kd_sim_weight 0.05 \
  --kd_temperature 0.1 \
  --teacher_ckpt src/checkpoint/teacher/2026-06-14_22-26/best_model.pth \
  --teacher_arch dinov3_vit7b16 \
  --teacher_dim 4096 \
  --student_dim 512 \
  --teacher_precision bf16 \
  --teacher_micro_batch_size 1
```

如果显存不足，优先：

1. 保持 `--teacher_micro_batch_size 1`；
2. 降低 `--batch_size`；
3. 使用 `--grad_accum_steps` 恢复 optimizer 有效 batch。

---

## 12. 启动前检查

```bash
python -m py_compile src/training/student_train.py
python -m pytest tests/test_student_baseline.py -q
```

检查 teacher checkpoint：

```bash
ls -lh src/checkpoint/teacher/2026-06-14_22-26/best_model.pth
```

检查 CLI：

```bash
python src/training/student_train.py --help
```

---

## 13. 静态审计结论

学生训练主流程已检查以下关键词：

```text
brd
boundary
risk-aware
risk_loss
risk_weight
margin_loss
teacher_d2s_margin
teacher_s2d_margin
student_d2s_margin
hard negative
boundary mask
risk mask
margin mask
violation
```

`student_train.py`、`student_model.py`、学生 dataset 和 student optimizer 中没有上述
风险感知蒸馏逻辑。

仓库的独立 teacher 训练模块 `src/training/teacher/train.py` 仍包含教师训练阶段自己的
hard-pool 代码，但它不被学生训练入口导入或调用，也不会影响 plain student
distillation。
