# 项目说明

本项目采用分层结构，便于数据管理、代码维护和实验复现。

- data/：数据目录，建议用软链接指向大数据文件夹。
- src/：主代码目录，包含数据处理、模型、训练、工具等模块。
- configs/：YAML配置文件。
- notebooks/：Jupyter分析与可视化。
- outputs/：模型权重、日志、结果等输出，建议 .gitignore。

详细结构见 README 顶部注释。

## 数据软链接建议

如数据集较大，建议用软链接方式引用，避免重复拷贝：

```bash
ln -s /your/real/data/path/university_1652 ./data/university_1652
```

## 常用命令示例

### 训练（多卡）
```bash
python -m torch.distributed.run --nproc_per_node=2 src/training/train_teacher.py
```

### 推理/评估
```bash
python src/inference/inference_dinov3_u1652.py --help
python src/inference/teacher_dinov3_u1652_no_grad.py --help
```

### 依赖安装
```bash
pip install -r requirements.txt
```

## 目录结构说明
- data/：数据目录（建议软链接）
- src/：主代码目录
- configs/：YAML配置
- notebooks/：分析与可视化
- outputs/：模型、日志、结果
- requirements.txt：依赖
- README.md：说明
- .gitignore：忽略规则