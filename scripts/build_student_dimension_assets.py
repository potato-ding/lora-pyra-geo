#!/usr/bin/env python3
"""TRAIN-only nested PCA extension of final S3 Top128 and Random32."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from src.student.formal_config import ROOT,load_formal_config
from src.student.formal_train import validate_assets
from src.student.shared_subspaces import load_reference
from src.student.build_bank import TrainImages
from src.student.subspace import identity_centroids
from src.student.subspace_utils import tensor_sha256
from src.student.artifacts import file_sha256


def extend_top(rows,mean,original):
    x=rows.double()-mean.double();p=original.double()
    def remove(z):return z-(z@p)@torch.linalg.solve(p.T@p,p.T)
    residual=remove(x)
    _,_,vh=torch.linalg.svd(residual,full_matrices=False)
    tail=remove(vh[:384]).T
    tail=torch.linalg.qr(tail,mode="reduced").Q.float()
    result=torch.cat([original,tail],dim=1).contiguous()
    if (result.T@result-torch.eye(512)).abs().max()>1e-5:raise ValueError("Nested Top512 orthogonality")
    if not torch.equal(result[:,:128],original):raise ValueError("Top128 prefix changed")
    return result


def extend_random(original,seed):
    # Same run-level isotropic Gaussian/QR family, with exact original32 prefix.
    gen=torch.Generator().manual_seed(seed)
    gaussian=torch.randn(768,32,generator=gen,dtype=torch.float64)
    q=torch.linalg.qr(gaussian,mode="reduced").Q
    piv=q.abs().argmax(0);q*=torch.where(q[piv,torch.arange(32)]<0,-1.,1.)
    if not torch.equal(q.float(),original):raise ValueError("Reference Random32 generator mismatch")
    extra=torch.randn(768,32,generator=gen,dtype=torch.float64)
    p=original.double();extra-=p@torch.linalg.solve(p.T@p,p.T@extra)
    tail=torch.linalg.qr(extra,mode="reduced").Q
    piv=tail.abs().argmax(0);tail*=torch.where(tail[piv,torch.arange(32)]<0,-1.,1.)
    result=torch.cat([original,tail.float()],dim=1).contiguous()
    if (result.T@result-torch.eye(64)).abs().max()>1e-5:raise ValueError("Nested Random64 orthogonality")
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--device",default="cuda:0");args=parser.parse_args()
    out=Path(args.output);assert not out.exists(),str(out)
    torch.set_num_threads(4)
    _,cfg=load_formal_config(ROOT/"configs/student/s3_sam_adsd_shared_m3_256.json")
    assets=validate_assets(cfg);ref=load_reference(cfg["subspace_reference"],cfg,assets)
    from src.evaluation.model_loader import load_encoder
    model,_=load_encoder("middle",cfg["middle_checkpoint"],cfg["middle_config"],args.device,image_size=256)
    rows=[];path_hashes={}
    for domain in ("drone","satellite"):
        ds=TrainImages(domain,ROOT/"data/U1652/train",256)
        assert len(ds.ids)==701
        path_hashes[domain]=hashlib.sha256("\n".join(str(p.relative_to(ROOT/"data/U1652/train"))+":"+str(label) for p,label in ds.rows).encode()).hexdigest()
        chunks=[];labels=[]
        with torch.inference_mode():
            for step,(images,ids) in enumerate(DataLoader(ds,batch_size=32,num_workers=4,shuffle=False,pin_memory=True),1):
                chunks.append(model(images.to(args.device)).cpu());labels.append(ids)
                if step%200==0:print("DIMENSION_BANK",domain,step,flush=True)
        centroids,identities=identity_centroids(torch.cat(chunks),torch.cat(labels));assert len(identities)==701
        rows.append(centroids)
    all_rows=torch.cat(rows);assert all_rows.shape==(1402,768)
    tensors={"teacher_mean":ref["teacher_mean"],"top_basis":extend_top(all_rows,ref["teacher_mean"],ref["top128_basis"]),
             "random_basis":extend_random(ref["random32_basis"],cfg["subspace_reference"]["random_basis_seed"])}
    manifest={"schema":"NESTED_STUDENT_SUBSPACES_V1","resolution":256,"split":"train","train_ids":701,"bank_rows":1402,
        "middle_sha256":assets["middle_checkpoint"],"middle_config_sha256":assets["middle_config"],"reference":cfg["subspace_reference"],
        "train_path_identity_sha256":path_hashes,"top_protocol":"fixed original mean/Top128; PCA of TRAIN identity-centroid residual in Top128 orthogonal complement; nested prefixes",
        "random_protocol":"exact original Gaussian-QR32 followed by Gaussian-QR orthogonal completion to64",
        "tensor_sha256":{k:tensor_sha256(v) for k,v in tensors.items()},"builder_sha256":file_sha256(__file__)}
    original=json.loads(Path(cfg["supervision_asset"]).read_text())
    assert manifest["train_path_identity_sha256"]==original["train_path_identity_sha256"]
    assert file_sha256(cfg["middle_checkpoint"])==assets["middle_checkpoint"]
    out.mkdir(parents=True)
    for k,v in tensors.items():
        with (out/(k+".pt")).open("xb") as f:torch.save(v,f)
    with (out/"manifest.json").open("x") as f:json.dump(manifest,f,indent=2)
    print("DIMENSION_ASSET_BUILD=PASS",str(out),flush=True)

if __name__=="__main__":main()
