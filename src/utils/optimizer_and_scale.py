from deepspeed.ops.adam import DeepSpeedCPUAdam

def build_optimizer_and_scale(model, args):
    """
    构建带有差分学习率的优化器，并初始化可学习的对比损失温度系数 (logit_scale)。
    """
    logit_scale = model.logit_scale
    
    # 2. 精细化抽取参数组
    lora_weight_decay = []
    lora_no_weight_decay = [] # 🌟 新增：专门存放不能做 weight decay 的一维参数（如 DoRA 的 m）
    classifier_params = []
    pyra_params = []
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
            
        # elif args.use_ce and ("classifier" in name or "fc" in name):
        #     classifier_params.append(param)
        # else:
        #     other_params.append(param)

    # 打印一下当前的参数分布，你可以借此二次确认 m 参数是不是顺利归队了
    print(f"优化器参数分布： LoRA方向矩阵: {len(lora_weight_decay)}, DoRA幅度向量(m): {len(lora_no_weight_decay)}, 分类头: {len(classifier_params)}, PYRA: {len(pyra_params)}, 其他兜底: {len(other_params)}, logit_scale: 1")
    
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
    head_params = classifier_params + pyra_params
    if head_params:
        optimizer_grouped_parameters.append({
            "params": head_params, 
            "lr": args.lr * 10, 
            "weight_decay": 0.01 
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