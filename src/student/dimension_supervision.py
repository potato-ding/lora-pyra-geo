"""Nested dimensional supervision; unchanged cosine/GBW/SAM equations."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .formal_supervision import BandProjector,SubspaceSupervision
from .part2 import FP32Module,ScalarGate
from .adsd import ADSDGate
from .subspace_utils import tensor_sha256
from .artifacts import file_sha256


def load_asset(cfg):
    path=Path(cfg["dimension_asset"]);m=json.loads(path.read_text())
    if (m.get("schema"),m.get("resolution"),m.get("split"),m.get("train_ids"),m.get("bank_rows"))!=("NESTED_STUDENT_SUBSPACES_V1",256,"train",701,1402):
        raise ValueError("Dimension asset protocol mismatch")
    if m.get("reference")!=cfg["subspace_reference"]:raise ValueError("Dimension reference changed")
    source=json.loads(Path(cfg["supervision_asset"]).read_text())
    if m["middle_sha256"]!=source["teacher_sha256"] or m["middle_config_sha256"]!=source["teacher_config_sha256"] or m["train_path_identity_sha256"]!=source["train_path_identity_sha256"]:
        raise ValueError("Dimension provider/TRAIN provenance mismatch")
    tensors={}
    for name,shape in [("teacher_mean",(768,)),("top_basis",(768,512)),("random_basis",(768,64))]:
        t=torch.load(path.parent/(name+".pt"),map_location="cpu",weights_only=True)
        if t.shape!=shape or t.dtype!=torch.float32 or not torch.isfinite(t).all() or tensor_sha256(t)!=m["tensor_sha256"][name]:raise ValueError("Dimension asset tensor mismatch")
        tensors[name]=t
    for name in ("top_basis","random_basis"):
        t=tensors[name]
        if (t.T@t-torch.eye(t.shape[1])).abs().max()>1e-5:raise ValueError("Nonorthogonal dimension asset")
    hashes=cfg["subspace_reference"]["tensor_sha256"]
    for name,t in [("teacher_mean",tensors["teacher_mean"]),("top128_basis",tensors["top_basis"][:,:128]),("random32_basis",tensors["random_basis"][:,:32])]:
        if tensor_sha256(t.contiguous())!=hashes[name]:raise ValueError("Original subspace prefix changed")
    return tensors,m


def validate_assets(cfg):
    from .formal_train import validate_assets as validate_original
    from .ablation_config import reference_config
    assets=validate_original(reference_config(cfg));load_asset(cfg)
    assets["dimension_asset"]=file_sha256(cfg["dimension_asset"])
    if "random_basis_control" in cfg:
        from .dimension_random import load_random
        load_random(cfg)
        assets["random_dimension_asset"]=file_sha256(cfg["random_basis_control"]["path"])
    return assets

class Residual(FP32Module):
    def __init__(self,dim):
        super().__init__();self.fc1=nn.Linear(512,920,dtype=torch.float32);self.fc2=nn.Linear(920,dim,dtype=torch.float32)
    def forward(self,x):return self.fc2(F.gelu(self.fc1(x.float())))

class TopHead(nn.Module):
    def __init__(self,base,dim,calibration):
        super().__init__();self.linear=base.linear
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0);self.residual=Residual(dim)
        self.gate=ScalarGate()
        with torch.no_grad():
            a=self.linear(calibration.float()).square().mean().sqrt();b=self.residual(calibration.float()).square().mean().sqrt()
            if not torch.isfinite(a+b) or min(float(a),float(b))<=0:raise ValueError("Dimension calibration invalid")
            self.residual.fc2.weight.mul_(a/b);self.residual.fc2.bias.mul_(a/b)
    def forward(self,z):
        with torch.autocast(device_type=z.device.type,enabled=False):
            raw=F.linear(z.float(),self.linear.weight.float(),self.linear.bias.float())+self.gate.alpha*self.residual(z.float())
            return F.normalize(raw,dim=-1),raw

class DimensionSupervision(nn.Module):
    def __init__(self,cfg,tensors,calibration,canonical):
        super().__init__();v=cfg["ablation"];self.top_dim=v["top_dim"];self.random_dim=v["random_dim"]
        self.register_buffer("teacher_mean",tensors["teacher_mean"].clone())
        self.register_buffer("top_basis",tensors["top_basis"][:,:self.top_dim].clone())
        if self.random_dim:self.register_buffer("random_basis",tensors["random_basis"][:,:self.random_dim].clone())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20260914)
            base=BandProjector(self.top_dim)
            with torch.no_grad():
                base.linear.weight[:128].copy_(canonical.projector_top.linear.weight)
                base.linear.bias[:128].copy_(canonical.projector_top.linear.bias)
            self.projector_top=TopHead(base,self.top_dim,calibration)
            if self.random_dim:
                self.projector_random=BandProjector(self.random_dim)
                with torch.no_grad():
                    self.projector_random.linear.weight[:32].copy_(canonical.projector_random.linear.weight)
                    self.projector_random.linear.bias[:32].copy_(canonical.projector_random.linear.bias)
        self.random_basis_generation_count=0
    def _apply(self,fn,recurse=True):
        assets={id(v) for v in self._buffers.values()}
        def preserve(t):
            if id(t) in assets:
                probe=fn(torch.empty(0,dtype=torch.float32,device=t.device));return t.to(device=probe.device,dtype=torch.float32)
            return fn(t)
        return super()._apply(preserve,recurse=recurse)
    def forward(self,z,teacher,pairs):
        with torch.autocast(device_type=z.device.type,enabled=False):
            centered=teacher.detach().float()-self.teacher_mean
            audit={};total=z.new_zeros((),dtype=torch.float32)
            for branch,basis in [("top",self.top_basis)]+([("random",self.random_basis)] if self.random_dim else []):
                prediction,_=getattr(self,"projector_"+branch)(z.float())
                target=F.normalize(centered@basis,dim=-1).detach()
                loss,drone,sat,*_=SubspaceSupervision._branch_loss(prediction,target,pairs)
                audit.update({branch+"_loss":loss,branch+"_drone_loss":drone,branch+"_satellite_loss":sat})
                total=total+loss
            if not torch.isfinite(total):raise FloatingPointError("Dimension KD nonfinite")
            return total,audit


def prepare(cfg,assets,calibration):
    from .formal_methods import prepare_adsd
    from .shared_subspaces import load_reference
    from .ablation_config import reference_config
    base=reference_config(cfg);old_assets={k:v for k,v in assets.items() if k not in ("dimension_asset","random_dimension_asset")}
    ref=load_reference(cfg["subspace_reference"],base,old_assets)
    canonical=prepare_adsd(cfg["supervision_asset"],assets["middle_checkpoint"],calibration,cfg["subspace_reference"]["random_basis_seed"],cfg["seed"],shared_tensors=ref)
    from .dimension_random import knowledge_tensors,basis_seed
    tensors,_=knowledge_tensors(cfg);sup=DimensionSupervision(cfg,tensors,calibration,canonical)
    gate=ADSDGate(0.) if cfg["ablation"]["target"]=="dual" else None
    return sup,gate,basis_seed(cfg) if gate is not None else None


def write_identity(output,cfg,sup,student):
    from .shared_subspaces import initialization_hashes
    from .dimension_random import basis_seed
    identity={"random_basis_seed":basis_seed(cfg) if sup.random_dim else None,"random_basis_control":cfg.get("random_basis_control"),"secondary_subspace":cfg.get("secondary_subspace"),"training_seed":cfg["seed"],"reference":cfg["subspace_reference"],
        "dimensions":cfg["ablation"],"dimension_asset_sha256":file_sha256(cfg["dimension_asset"]),
        "tensors":{n:tensor_sha256(t) for n,t in sup.named_buffers()},"random_generation_count":0}
    (output/"subspace_identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    (output/"initialization_identity.json").write_text(json.dumps({"student":initialization_hashes(student),"heads":initialization_hashes(sup),"training_seed":cfg["seed"]},indent=2)+"\n")
    print("DIMENSION_IDENTITY="+json.dumps(identity),flush=True)
