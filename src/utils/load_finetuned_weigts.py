import os
import torch
import torch.distributed as dist
import re  # <--- 记得文件顶部要有这个导入

def load_finetuned_weights(model, weight_path, device, is_main=False):
    """
    专门用于加载 LoRA/DoRA 等局部微调权重的封装函数
    """

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"[Error] 找不到权重文件: {weight_path}")

    if is_main:
        print(f"正在从 {weight_path} 直接加载轻量级微调权重到 {device}...")

    # 1. 既然只是轻量级权重，直接 load 到目标 device，跳过 CPU 内存中转
    checkpoint = torch.load(weight_path, map_location=device)

    # 2. 提取纯净的 state_dict
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

    # 3. 剥离 DDP 前缀 & 修复 Checkpoint Wrapper 多余层级
    new_state_dict = {}
    for k, v in state_dict.items():
        # 第一步：剥离 DDP 带来的 'module.' 前缀
        new_key = k[7:] if k.startswith('module.') else k
        
        # 第二步：【核心修复】抹平 SmartCheckpointWrapper 造成的多余层级！
        # 自动识别类似 blocks.39.xxxxx.attn 的路径，强行将其缝合还原为 blocks.39.attn
        new_key = re.sub(r'(blocks\.\d+\.)[^.]+\.(attn|mlp|norm1|norm2|ls1|ls2)', r'\1\2', new_key)
        
        new_state_dict[new_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    if is_main:
        # 如果有 unexpected_keys，说明权重文件里有模型中不存在的层
        if len(unexpected_keys) > 0:
            print(f"[警告] 发现了 {len(unexpected_keys)} 个无法匹配的权重层！")
            # 打印前 3 个没对上的名字，方便后续排错
            print(f"       例如: {unexpected_keys[:3]}") 
        else:
            print(">>> LoRA/DoRA 微调权重成功注入模型！")