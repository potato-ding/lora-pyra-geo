# metrics.py
# 评估指标相关函数，可从原 utils/metrics.py 迁移
import numpy as np
import torch

def compute_mAP(index: np.ndarray, good_index: np.ndarray, junk_index: np.ndarray):
	ap = 0.0
	cmc = torch.IntTensor(len(index)).zero_()
	if good_index.size == 0:
		cmc[0] = -1
		return ap, cmc
	mask = np.in1d(index, junk_index, invert=True)
	index = index[mask]
	ngood = len(good_index)
	mask = np.in1d(index, good_index)
	rows_good = np.argwhere(mask == True).flatten()
	cmc[rows_good[0]:] = 1
	for i in range(ngood):
		d_recall = 1.0 / ngood
		precision = (i + 1) * 1.0 / (rows_good[i] + 1)
		if rows_good[i] != 0:
			old_precision = i * 1.0 / rows_good[i]
		else:
			old_precision = 1.0
		ap = ap + d_recall * (old_precision + precision) / 2.0
	return ap, cmc

def evaluate_retrieval(query_feats, query_labels, gallery_feats, gallery_labels):
	"""
	计算Recall@1, Recall@5, Recall@10, mAP
	query_feats, gallery_feats: (N, C) torch.Tensor
	query_labels, gallery_labels: (N,) torch.Tensor
	"""
	device = query_feats.device
	qf = query_feats.to(device)
	gf = gallery_feats.to(device)
	ql = query_labels.cpu().numpy()
	gl = gallery_labels.cpu().numpy()
	num_gallery = gl.shape[0]
	CMC = torch.IntTensor(num_gallery).zero_()
	ap = 0.0
	for i, q_label in enumerate(ql):
		query = qf[i].view(-1, 1)
		scores = torch.mm(gf, query).squeeze(1).cpu().numpy()
		index = np.argsort(scores)[::-1]
		good_index = np.argwhere(gl == q_label)
		junk_index = np.argwhere(gl == -1)
		ap_tmp, cmc_tmp = compute_mAP(index, good_index, junk_index)
		if cmc_tmp[0] == -1:
			continue
		CMC = CMC + cmc_tmp
		ap += ap_tmp
	CMC = CMC.float() / len(ql)
	mAP = ap / len(ql)
	recall1 = CMC[0].item() * 100
	recall5 = CMC[4].item() * 100 if num_gallery > 4 else float('nan')
	recall10 = CMC[9].item() * 100 if num_gallery > 9 else float('nan')
	mAP = mAP * 100
	return {'Recall@1': recall1, 'Recall@5': recall5, 'Recall@10': recall10, 'mAP': mAP}
# metrics.py
# 评估指标相关函数，可从原 utils/metrics.py 迁移
