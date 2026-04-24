from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch
def build_optimizer_and_scale(model, args):
    """
    构建带有差分学习率的优化器，并初始化可学习的对比损失温度系数 (logit_scale)。
    """
    logit_scale = model.logit_scale
    
    # 2. 精细化抽取参数组
    lora_weight_decay = []
    lora_no_weight_decay = [] # 专门存放不能做 weight decay 的一维参数（如 DoRA 的 m）
    classifier_params = []
    mix_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "logit_scale" or name.endswith(".logit_scale"):
            continue
            
        if "lora_" in name:
            lora_weight_decay.append(param)    # 存放 lora_A.weight, lora_B.weight
        elif name.endswith(".m"): #把 DoRA 的 m 都保护起来！
            lora_no_weight_decay.append(param)
        elif "ap_gates" in name or "global_ap_scale" in name or "gamma" in name:
            mix_params.append(param)
        elif "feature_adapter" in name or "cross_attn" in name or "query_norm" in name:
            classifier_params.append(param)
        else:
            print(f"警告：参数 {name} 没有被正确分类，默认放入其他参数组，建议检查命名是否符合预期！")
            other_params.append(param)

    # 打印一下当前的参数分布，你可以借此二次确认 m 参数是不是顺利归队了
    print(f"优化器参数分布： LoRA方向矩阵: {len(lora_weight_decay)}, DoRA幅度向量(m): {len(lora_no_weight_decay)}, 分类头: {len(classifier_params)}, 多层融合权重: {len(mix_params)}, 其他兜底: {len(other_params)}, logit_scale: 1")
    
    # 3. 组装高级优化器参数字典
    optimizer_grouped_parameters = []

    # 方向矩阵 A 和 B (保留正常的 weight decay)
    if lora_weight_decay:
        optimizer_grouped_parameters.append({
            "params": lora_weight_decay, 
            "lr": args.lr, 
            "weight_decay": 0.01
        })
        
    #幅度参数 m (坚决设为 0.0 weight decay！)
    if lora_no_weight_decay:
        optimizer_grouped_parameters.append({
            "params": lora_no_weight_decay, 
            "lr": args.lr, 
            "weight_decay": 0.0 
        })

    # 头部参数合并 (保留原来的 10 倍学习率逻辑)
    head_params = classifier_params
    if head_params:
        optimizer_grouped_parameters.append({
            "params": head_params, 
            "lr": args.lr * 2, 
            "weight_decay": 0.01 
        })
    if mix_params:
        optimizer_grouped_parameters.append({
            "params": mix_params,
            "lr": args.lr * 10,  # 如果后续发现浅层特征融合得太慢，这里甚至可以考虑给 args.lr * 5 或 * 10
            "weight_decay": 0.0  
        })

    # 兜底的其他参数 (由于 DINOv3 冻结，这个列表应该是空的，写上保底防报错)
    if other_params:
        optimizer_grouped_parameters.append({
            "params": other_params, 
            "lr": args.lr, 
            "weight_decay": 0.01
        })

    optimizer_grouped_parameters.append({
        "params": [logit_scale],
        "lr": args.lr,
        "weight_decay": 0.0 
    })
    
    
    # 去掉外层的全局 weight_decay 参数
    optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters)
    
    print("优化器实例化成功！")
    
    return optimizer


def build_student_optimizer(
    model,
    backbone_lr=1e-4,
    head_lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
):
    """
    为 StudentModel 构建 AdamW optimizer

    参数分组策略：
    1. backbone 参数：较小 lr
    2. 新增头部参数（gem_pool / bottleneck / fc_main / aux_heads）：较大 lr
    3. bias / BN / norm / 标量参数：不做 weight decay
    """

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 是否属于 backbone
        is_backbone = name.startswith("backbone.")

        # 是否不做 weight decay
        name_lower = name.lower()
        is_no_decay = (
            param.ndim <= 1              # bias / BN weight / 标量参数
            or name.endswith(".bias")
            or "bn" in name_lower
            or "norm" in name_lower
        )

        if is_backbone:
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        else:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_decay,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": backbone_no_decay,
                "lr": backbone_lr,
                "weight_decay": 0.0,
            },
            {
                "params": head_decay,
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": head_no_decay,
                "lr": head_lr,
                "weight_decay": 0.0,
            },
        ],
        betas=betas,
    )

    print("[Optimizer] backbone_decay params   :", len(backbone_decay))
    print("[Optimizer] backbone_no_decay params:", len(backbone_no_decay))
    print("[Optimizer] head_decay params       :", len(head_decay))
    print("[Optimizer] head_no_decay params    :", len(head_no_decay))

    return optimizer
