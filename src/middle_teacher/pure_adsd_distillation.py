"""ADSD-only T0 supervision; no HRD or semantic feature extraction."""
from __future__ import annotations

import torch

from src.middle_teacher.checkpoint import sha256
from src.middle_teacher.core_config import validate_teacher_identity
from src.training.teacher.formal_reload import load_teacher_v3
from src.utils.gather_features_and_labels_and_views import concat_all_gather


class ADSDDistillationRuntime:
    def __init__(self, config, checkpoint, device, chunk_size=4):
        if not checkpoint:
            raise ValueError("Teacher checkpoint is required for Middle KD")
        size = config["data"]["input_size"]
        encoder, audit = load_teacher_v3(checkpoint, device=device, image_size=size)
        validate_teacher_identity(audit["checkpoint_metadata"], size)
        self.encoder = encoder
        self.teacher = encoder.model
        self.audit = dict(audit, sha256=sha256(checkpoint))
        self.config = config["distillation"]
        self.adsd_config = config["adsd"]
        self.epoch = 1
        from src.student.adsd import ADSDGate
        self.gate = ADSDGate(config["adsd"]["gate_initial_d"]).to(device)
        self.gate_loss = None
        self.local_pair_batch = config["data"]["local_pair_batch"]
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0 or any(p.requires_grad for p in self.teacher.parameters()):
            raise ValueError("Teacher must be frozen and chunk size positive")

    def compose_all(
        self, base, md, ms, images, ids, model, step,
        hidden=None, return_kd_objective=False,
    ):
        if self.config != {'base_loss': 'pair_infonce'} or hidden is not None:
            raise RuntimeError("ADSD-only runtime forbids HRD and Semantic")
        with torch.no_grad():
            chunks = []
            for start in range(0, images.shape[0], self.chunk_size):
                value = self.teacher(images[start:start+self.chunk_size].to(dtype=torch.bfloat16))
                if isinstance(value, dict):
                    value = value.get('final_descriptor', value.get('descriptor'))
                chunks.append(value.detach().float())
            final = torch.nn.functional.normalize(torch.cat(chunks), dim=1, eps=1e-6)
            if final.shape != (images.shape[0], 4096) or not torch.isfinite(final).all():
                raise RuntimeError("Invalid T0 final descriptors")
            td = concat_all_gather(final[:self.local_pair_batch])
            ts = concat_all_gather(final[self.local_pair_batch:])
        from src.student.adsd import isolated_gradient_signal, gate_objective, kd_warmup_factor
        desc=torch.cat((md,ms));target=torch.cat((td,ts));pairs=md.shape[0]
        sup=model.adsd_supervision
        _,dual=sup(desc,target,pairs)
        w_top,w_random=self.gate();w_top,w_random=w_top.detach(),w_random.detach()
        top_signal,random_signal,cosine=isolated_gradient_signal(sup,desc,target,dual,pairs)
        self.gate_loss=gate_objective(self.gate,top_signal,random_signal,self.epoch,self.adsd_config['warmup_epochs'])
        effective=self.adsd_config['weight']*kd_warmup_factor(self.epoch,self.adsd_config['warmup_epochs'])
        adsd=effective*(w_top*dual['top_loss']+w_random*dual['random_loss'])
        kd = adsd
        stats = {"teacher_requires_grad_count": 0, "teacher_optimizer_param_count": 0,
                 "teacher_descriptor_source": "FINAL_CLS", "global_pool": int(md.shape[0]),
                 "hrd_active": False, "semantic_active": False}
        stats.update(adsd_weighted_loss=float(adsd.detach()),adsd_effective_weight=effective,
                     adsd_top_loss=float(dual['top_loss'].detach()),adsd_random_loss=float(dual['random_loss'].detach()),
                     adsd_w_top=float(w_top),adsd_w_random=float(w_random),adsd_gradient_cosine=float(cosine),
                     adsd_global_pairs=pairs)
        if return_kd_objective:
            return base + kd, stats, kd
        return base + kd, stats
