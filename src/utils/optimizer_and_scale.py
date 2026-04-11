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
        elif name.endswith(".m"): #把 DoRA 的 m 和 GeM 的 p 都保护起来！
            lora_no_weight_decay.append(param)
        elif "ap_gates" in name:
            mix_params.append(param)
        elif "ap_gates" in name or "global_ap_scale" in name:
            mix_params.append(param)
            
        # elif args.use_ce and ("classifier" in name or "fc" in name):
        #     classifier_params.append(param)
        # else:
        #     other_params.append(param)

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
            "lr": args.lr * 10, 
            "weight_decay": 0.01 
        })
    if mix_params:
        optimizer_grouped_parameters.append({
            "params": mix_params,
            "lr": args.lr,  # 如果后续发现浅层特征融合得太慢，这里甚至可以考虑给 args.lr * 5 或 * 10
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
    
    return optimizer, logit_scale


def build_student_optimizer_and_scale(model, args):
    """
    专为 RepViT 学生模型构建的优化器。
    保留了头部 10x 学习率和 Bias/Norm 免衰减机制，去除了多余的 LoRA/DeepSpeed 逻辑。
    """
    # 兼容解包逻辑，如果学生模型没有 logit_scale，则返回 None
    logit_scale = getattr(model, 'logit_scale', None)

    # 1. 准备参数组
    head_params_wd = []
    head_params_no_wd = []
    backbone_params_wd = []
    backbone_params_no_wd = []
    
    # 设定默认的 weight_decay，如果在 args 中没有则默认 0.05 (AdamW 常用)
    weight_decay = getattr(args, 'weight_decay', 0.05)

    # 2. 遍历参数并精细分组
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 判定是否属于头部结构 (新初始化的层)
        is_head = any(keyword in name for keyword in ["aux_head", "fc_main", "uapa_bottleneck"])
        
        # 判定是否需要免除 Weight Decay
        # 规则: 1D 张量 (Bias, Norm层权重) 或 显式包含特定关键词 (如 GeM 的 p)
        no_decay = (len(param.shape) == 1) or name.endswith(".bias") or ("p" in name.split('.')[-1])

        if is_head:
            if no_decay:
                head_params_no_wd.append(param)
            else:
                head_params_wd.append(param)
        else:
            if no_decay:
                backbone_params_no_wd.append(param)
            else:
                backbone_params_wd.append(param)

    # 3. 组装优化器参数字典
    optimizer_grouped_parameters = [
        # 主干网络 (1x LR)
        {"params": backbone_params_wd, "lr": args.lr, "weight_decay": weight_decay},
        {"params": backbone_params_no_wd, "lr": args.lr, "weight_decay": 0.0},
        
        # 任务头/辅助头 (10x LR，加速收敛)
        {"params": head_params_wd, "lr": args.lr * 10, "weight_decay": weight_decay},
        {"params": head_params_no_wd, "lr": args.lr * 10, "weight_decay": 0.0},
    ]

    # 4. 实例化原生 AdamW (速度远快于 CPUAdam)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # 打印分布情况，方便你核对
    print(f"🚀 学生模型优化器构建成功!")
    print(f" - Backbone (带衰减, 1x LR): {len(backbone_params_wd)} 个参数组")
    print(f" - Backbone (无衰减, 1x LR): {len(backbone_params_no_wd)} 个参数组 (含 Bias/Norm)")
    print(f" - Heads    (带衰减, 10x LR): {len(head_params_wd)} 个参数组")
    print(f" - Heads    (无衰减, 10x LR): {len(head_params_no_wd)} 个参数组")

    return optimizer, logit_scale
