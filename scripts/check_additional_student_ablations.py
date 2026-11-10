#!/usr/bin/env python3
"""CPU-only B1/B8 readiness report; never builds assets or starts a run."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.student.formal_config import ROOT,load_formal_config
from src.student.formal_train import validate_assets,collect_student_source_identity,sam_enabled
from src.student.component_ablation import prepare
from src.student.formal_assets import validate_middle_provider
from src.student.shared_subspaces import validate_tensors
from src.student.subspace_utils import tensor_sha256

torch.set_num_threads(2)
for filename in ("b1_m0_provider_sam_m0_256.json","b8_no_sam_m3_256.json"):
    _,cfg=load_formal_config(ROOT/"configs/student"/filename)
    output=Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    collect_student_source_identity(cfg)
    try:
        assets=validate_assets(cfg)
    except FileNotFoundError as exc:
        if cfg["ablation"]["variant"]!="b1_m0_provider" or "B1_ASSET_BUILD_PENDING" not in str(exc):raise
        print(json.dumps({"variant":"b1_m0_provider","config":"PASS","M0_checkpoint":"STRICT_PASS", "status":"ASSET_BUILD_PENDING", "action":"scripts/prepare_b1_m0_assets.sh after GPU is assigned"}))
        continue
    cal=torch.load(cfg["top_calibration"],map_location="cpu",weights_only=True)
    sup,gate,seed=prepare(cfg,assets,cal)
    if "subspace_reference" in cfg:validate_tensors(sup.state_dict(),cfg["subspace_reference"])
    assert sup.random_basis_generation_count==0
    print(json.dumps({"variant":cfg["ablation"]["variant"],"status":"CPU_READY", "sam_enabled":sam_enabled(cfg),"random_generation_count":0,"random_basis_sha256":tensor_sha256(sup.random32_basis)}))
print("NO_GPU_WORK; NO_TRAINING; NO_AUTOMATIC_QUEUE")
