import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from deepspeed.ops.adam import DeepSpeedCPUAdam
def build_optimizer_and_scale(model, args):
    """
    构建带有差分学习率的优化器，并初始化可学习的对比损失温度系数 (logit_scale)。
    """
    logit_scale = model.logit_scale
    # 2. 精细化抽取参数组
    lora_params = []
    classifier_params = []
    pyra_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "logit_scale" or name.endswith(".logit_scale"):
            continue
        if "lora_" in name:
            lora_params.append(param)
        elif args.use_ce and ("classifier" in name or "fc" in name):
            classifier_params.append(param)
        elif args.use_pyra and ("pyra" in name or "wr" in name or "wd" in name):
            pyra_params.append(param)
        else:
            other_params.append(param)

    print(f"优化器参数分布: LoRA: {len(lora_params)}, 分类头: {len(classifier_params)}, PYRA: {len(pyra_params)}, 其他: {len(other_params)}, logit_scale: 1")
    # 3. 组装高级优化器参数字典
    optimizer_grouped_parameters = []
    
    if lora_params:
        optimizer_grouped_parameters.append({"params": lora_params, "lr": args.lr})

    # 头部参数合并（给 10 倍学习率）
    head_params = classifier_params + pyra_params
    if head_params:
        optimizer_grouped_parameters.append({"params": head_params, "lr": args.lr * 10})

    if other_params:
        optimizer_grouped_parameters.append({"params": other_params, "lr": args.lr})

    # 🌟 极其关键：把 logit_scale 塞进优化器，让它能被更新！
    optimizer_grouped_parameters.append({
        "params": [logit_scale], 
        "lr": args.lr, 
        "weight_decay": 0.0  # 标量不需要权重衰减
    })
    print("优化器参数组装完成！")
    # 4. 真正实例化你的 AdamW 优化器
    optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, weight_decay=0.01)
    print("优化器实例化成功！")
    return optimizer, logit_scale