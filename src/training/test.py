# === 0. 路径与依赖 ===
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import argparse
from src.data.datasets import U1652StrictTestDataset
import numpy as np
from src.utils.result_method import compute_mAP
from src.data.transforms import get_test_transforms

def extract_feats(model, subset, device="cuda", batch_size=64):
    """
    批量提取特征与标签 (高精度修正版)
    """
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    feats, labels = [], []
    total = len(loader)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            imgs, batch_labels, * _ = batch
            imgs = imgs.to(device)
            batch_labels = batch_labels.to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                f = model(imgs)
                
                # 新增：剥离 Tuple，拿到真正的特征 Tensor
                if isinstance(f, tuple):
                    f = f[0] 
            
            # 1. 强制将特征拉回 32 位浮点数
            f = f.float() 
            
            # 2. 在 32 位精度下执行 L2 归一化 (平方 -> 求和 -> 开根号)
            f = F.normalize(f, p=2, dim=1)

            feats.append(f.cpu())
            labels.append(batch_labels.cpu())

            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"\r[特征提取] Batch {idx + 1}/{total}", end="")
                
    print() # 换行

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

def evaluate_retrieval_standard(q_feats, q_labels, g_feats, g_labels):
    """
    学术标准评估流水线：
    1. GPU 矩阵乘法算出相似度矩阵 (快)
    2. CPU 循环调用 compute_mAP (准)
    """
    # [1] 计算相似度矩阵 (余弦距离)
    # 确保已经做了 L2 归一化，直接矩阵乘法
    print("[Score] 计算特征相似度矩阵...")
    score_matrix = torch.matmul(q_feats, g_feats.T)
    
    # [2] 对得分进行排序
    print("[Sort] 正在进行全局检索排序...")
    _, indices = torch.sort(score_matrix, dim=1, descending=True)
    indices = indices.cpu().numpy()
    q_labels = q_labels.cpu().numpy()
    g_labels = g_labels.cpu().numpy()

    # [3] 逐行计算 AP 和 CMC
    num_q = q_labels.shape[0]
    all_ap = []
    all_cmc = torch.IntTensor(g_labels.shape[0]).zero_()
    
    from tqdm import tqdm
    for i in tqdm(range(num_q), desc="计算排名与精度"):
        target = q_labels[i]
        index = indices[i]
        
        # 找出当前 Query 对应的正确项和干扰项索引
        good_index = np.argwhere(g_labels == target).flatten()
        junk_index = np.argwhere(g_labels == -1).flatten()
        
        # 调用核心算法
        ap, cmc = compute_mAP(index, good_index, junk_index)
        
        if ap != -1:
            all_ap.append(ap)
            all_cmc += cmc

    # [4] 汇总指标
    mAP = np.mean(all_ap) * 100
    # Recall@k (CMC)
    recall_1 = all_cmc[0] / len(all_ap) * 100
    recall_5 = all_cmc[4] / len(all_ap) * 100
    recall_10 = all_cmc[9] / len(all_ap) * 100

    return {'Recall@1': recall_1, 'Recall@5': recall_5, 'Recall@10': recall_10, 'mAP': mAP}

def get_ckpt_path(epoch, use_lora, use_contrastive, use_triplet, use_adapter, use_pyra, classifier_type,zero_shot=False, use_ce=False):
	"""
	根据参数自动生成权重路径。
	优先寻找指定 epoch 的权重；如果找不到，自动 fallback 到 best_lora_model.pth。
	"""
	if zero_shot:
		print("\n🚀 [Zero-Shot 模式] 已启用！跳过权重搜索，直接评估 DINOv3 预训练底座能力。")
		return None
	# 1. 构建与训练时完全一致的目录名
	save_dir = os.path.join(
		'src/checkpoint',
		'dino' + 
		('_lora' if use_lora else '') + 
        ('_ce' if use_ce else '') +
		('_contrastive' if use_contrastive else '') + 
		('_triplet' if use_triplet else '') + 
		('_adapter' if use_adapter else '') + 
		('_pyra' if use_pyra else '') + 
		(f'_{classifier_type}')
	)
	
	# 2. 优先尝试寻找指定 epoch 的文件
	epoch_file = f"epoch{epoch}.pth"
	epoch_path = os.path.join(save_dir, epoch_file)
	
	if os.path.exists(epoch_path):
		print(f"[*] 命中目标！准备加载指定轮次权重: {epoch_file}")
		return epoch_path
		
	# 3. 如果指定 epoch 的权重不存在，触发 Fallback 机制
	print(f"[!] 未找到 {epoch_file}，自动回退 (Fallback) 到最佳权重...")
	best_file = "best_epoch.pth"
	best_path = os.path.join(save_dir, best_file)
	
	if os.path.exists(best_path):
		print(f"[*] 成功接管！准备加载最佳权重: {best_file}")
		return best_path
		
	# 4. 如果连 best 模型都找不到，说明目录错了或者根本没存下来，直接报错
	raise FileNotFoundError(
		f"\n❌ 致命错误：权重文件全部丢失！\n"
		f"目标目录: {save_dir}\n"
		f"既没有找到 {epoch_file}，也没有找到 {best_file}。\n"
		f"请检查训练是否成功保存，或者推理参数 (--use_xxx) 是否与训练时完全对应。"
	)

def handleModelEvaluateReuslt(model, strict_datasets, args):
	# 2. 剥离分类头 (确保 extract_feats 拿到的是特征向量，而不是 logits)
	if hasattr(model, 'head'):
		model.head = nn.Identity()
	elif hasattr(model, 'classifier'):
		model.classifier = nn.Identity()
	elif hasattr(model, 'fc'):
		model.fc = nn.Identity()
	
	print("[*] 模型准备就绪，开始提取特征...")
	# 2. 开启 D2S 任务
	print("\n🚀 [Task 1] Drone -> Satellite")
	q_d2s_f, q_d2s_l = extract_feats(model, strict_datasets['q_drone'], args.device, args.batch_size)
	g_d2s_f, g_d2s_l = extract_feats(model, strict_datasets['g_sat'], args.device, args.batch_size)
	d2s_results = evaluate_retrieval_standard(q_d2s_f, q_d2s_l, g_d2s_f, g_d2s_l)

	# 3. 开启 S2D 任务
	print("\n🚀 [Task 2] Satellite -> Drone")
	q_s2d_f, q_s2d_l = extract_feats(model, strict_datasets['q_sat'], args.device, args.batch_size)
	g_s2d_f, g_s2d_l = extract_feats(model, strict_datasets['g_drone'], args.device, args.batch_size)
	s2d_results = evaluate_retrieval_standard(q_s2d_f, q_s2d_l, g_s2d_f, g_s2d_l)

	# 4. 最终打印 (这才是论文里表格的样子)
	print("\n" + "="*50)
	print(f"D2S: R@1:{d2s_results['Recall@1']:.2f} | R@5:{d2s_results['Recall@5']:.2f} | R@10:{d2s_results['Recall@10']:.2f} | mAP:{d2s_results['mAP']:.2f}")
	print(f"S2D: R@1:{s2d_results['Recall@1']:.2f} | R@5:{s2d_results['Recall@5']:.2f} | R@10:{s2d_results['Recall@10']:.2f} | mAP:{s2d_results['mAP']:.2f}")
	print("="*50)
	print("[LoRA] 检索推理全部完成！")

def handleLoadckpt(model, ckpt_path):
	if ckpt_path is None:
		print("[Zero-Shot] 跳过权重加载，直接使用预训练底座进行评测。")
		return
	print(f"[*] 正在加载权重: {ckpt_path} ...")
	# 1. 把权重读到内存 (先放 CPU，后面统一 to(device))
	checkpoint = torch.load(ckpt_path, map_location='cpu')
	
	missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
	
	print(f"[*] 权重加载成功！")

# === 2. main入口 ===
if __name__ == "__main__":
	import traceback
	parser = argparse.ArgumentParser(description="LoRA 检索推理与评测脚本")
	parser.add_argument('--device', type=str, default='cuda', help='推理设备')
	parser.add_argument('--batch_size', type=int, default=64, help='推理batch size')
	parser.add_argument('--use_lora', action='store_true', help='是否启用LoRA模块', default=False)
	parser.add_argument('--use_ce', action='store_true', help='是否启用交叉熵损失', default=False)
	parser.add_argument('--use_adapter', action='store_true', help='是否启用Adapter模块', default=False)
	parser.add_argument('--use_pyra', action='store_true', help='是否启用PYRA模块', default=False)
	parser.add_argument('--use_contrastive', action='store_true', help='是否启用对比学习', default=False)
	parser.add_argument('--use_triplet', action='store_true', help='是否启用三元组损失', default=False)
	parser.add_argument('--epoch', type=int, default=0, help='推理所用的epoch权重编号,默认0表示使用best_model.pth')
	parser.add_argument('--classifier_type', type=str, default='resBottle', help='分类器类型 (可选: "layerNormBottle" | "transBottle" | "resBottle")')
	parser.add_argument('--zero_shot', action='store_true', help='开启零样本测试，不加载任何微调权重')
	args = parser.parse_args()
	try:
		from src.models.teacher_model import create_model
		model = create_model(
			use_lora=args.use_lora,
			use_adapter=args.use_adapter,
			use_pyra=args.use_pyra,
			classifier_type=args.classifier_type,
		)
		ckpt_path = get_ckpt_path(
			epoch=args.epoch,
			use_lora=args.use_lora,
			use_contrastive=args.use_contrastive,
			use_triplet=args.use_triplet,
			use_adapter=args.use_adapter,
			use_pyra=args.use_pyra,
			classifier_type=args.classifier_type,
			zero_shot=args.zero_shot,
			use_ce=args.use_ce
		)

		# 根据权重文件路径加载训练权重
		handleLoadckpt(model, ckpt_path)

		print("🌟 严格模式 (Strict Protocol) 测试集加载中...")
		val_transforms = get_test_transforms(img_size=[384, 384]) # 注意测试时的尺寸调整
		strict_datasets = {
			'q_drone': U1652StrictTestDataset('data/university_1652/test/query_drone', val_transforms),
			'g_sat': U1652StrictTestDataset('data/university_1652/test/gallery_satellite', val_transforms),
			'q_sat': U1652StrictTestDataset('data/university_1652/test/query_satellite', val_transforms),
			'g_drone': U1652StrictTestDataset('data/university_1652/test/gallery_drone', val_transforms)
		}

		# 1. 搬运模型到 GPU
		model = model.to(args.device)
		model.eval() # 🚨 必须进入 eval 模式，否则 BatchNorm 和 Dropout 会乱套

		handleModelEvaluateReuslt(model, strict_datasets, args)
		
	except Exception as e:
		print("\n[Error] Exception occurred during inference:")
		traceback.print_exc()
		import sys
		sys.exit(1)
