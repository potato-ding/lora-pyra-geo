#!/usr/bin/env python3
"""CPU-only B2--B7 launch preflight. Does not create output dirs or start training."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import torch
from src.student.ablation_config import VARIANTS
from src.student.formal_config import ROOT, load_formal_config
from src.student.formal_train import validate_assets, collect_student_source_identity
from src.student.component_ablation import prepare
from src.student.shared_subspaces import validate_tensors, initialization_hashes
from src.student.formal_engine import StudentTrainingModel
from src.student.model import StudentModel
from src.student.optimizer import build_student_optimizer
from src.student.formal_top import prepare_student_precision_groups
from src.student.distill_sam import collect_optimizer_trainable_params
from src.student.subspace_utils import tensor_sha256


def main():
    torch.set_num_threads(2)
    common = None
    for key in VARIANTS:
        path,cfg=load_formal_config(ROOT/f"configs/student/{key}_sam_m3_256.json")
        output=Path(cfg["output_dir"])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Refusing occupied output directory: "+str(output))
        assets=validate_assets(cfg)
        torch.manual_seed(cfg["seed"])
        student=StudentModel(str(ROOT/cfg["pretrained_checkpoint"]),temperature=cfg["task_loss"]["temperature"])
        cal=torch.load(cfg["top_calibration"],map_location="cpu",weights_only=True)
        sup,gate,seed=prepare(cfg,assets,cal)
        validate_tensors(sup.state_dict(),cfg["subspace_reference"])
        model=StudentTrainingModel(student,sup)
        opt=build_student_optimizer(model,cfg["lr"],cfg["weight_decay"])
        prepare_student_precision_groups(model,opt)
        named=collect_optimizer_trainable_params(model,opt)
        assert all(not n.startswith("supervision.projector_random") for n,p in named) if key=="b3_top128" else True
        assert all(not n.startswith("supervision.projector_top") for n,p in named) if key=="b4_random32" else True
        if gate is not None:
            assert all(p is not gate.d for n,p in named)
        initial=initialization_hashes(student)
        if common is None:common=initial
        assert common==initial,"Student initialization differs between controls"
        collect_student_source_identity(cfg)
        print("ABLATION_PREFLIGHT="+json.dumps({"variant":key,"PASS":True,
            "output_dir":str(output),"device":"cpu","random_generation_count":sup.random_basis_generation_count,
            "random_basis_sha256":tensor_sha256(sup.random32_basis),"gbw":gate is not None,
            "trainable_parameters":sum(p.numel() for n,p in named),"student_initialization_matches":True}),flush=True)
    print("ALL_SIX_READY_FOR_GPU_START; NO_TRAINING_STARTED")

if __name__=="__main__":main()
