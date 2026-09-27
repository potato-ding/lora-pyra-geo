"""Fit canonical TRAIN mean/Top128 from a validated E3 Middle."""
import argparse,json,hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .build_bank import TrainImages,ROOT
from .subspace import identity_centroids
from .subspace_utils import tensor_sha256
from .artifacts import file_sha256

def fit_top128(rows):
    x=torch.cat(rows).double();assert x.shape==(1402,768)
    mean64=x.mean(0);mean=mean64.float().contiguous()
    _,_,vh=torch.linalg.svd(x-mean64,full_matrices=False)
    oldtop=vh[:32].T.float().contiguous()
    torch.set_num_threads(4)
    _,_,vh=torch.linalg.svd(x-mean.double(),full_matrices=False)
    top=vh[:128].T.contiguous()
    top[:,:32]*=torch.where((top[:,:32]*oldtop.double()).sum(0)<0,-1.,1.)
    top=top.float().contiguous();top[:,:32]=oldtop
    values={'teacher_mean':mean,'top128_basis':top}
    return values

def main(argv=None):
    from src.evaluation.model_loader import load_encoder
    from .middle_source import validate_e3_middle
    from .formal_supervision import load_top_source
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('middle-checkpoint','middle-config','output'):p.add_argument('--'+field,required=True)
    p.add_argument('--image-size',type=int,choices=(224,256),required=True)
    p.add_argument('--train-root',default=str(ROOT/'data/U1652/train'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--num-workers',type=int,default=4)
    a=p.parse_args(argv);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    _,teacher_sha=validate_e3_middle(a.middle_checkpoint,a.middle_config,a.image_size)
    config_sha=file_sha256(a.middle_config)
    torch.set_num_threads(8)
    model,audit=load_encoder('middle',a.middle_checkpoint,a.middle_config,a.device,image_size=a.image_size)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    datasets={d:TrainImages(d,a.train_root,a.image_size) for d in ('drone','satellite')}
    if datasets['drone'].ids!=datasets['satellite'].ids or len(datasets['drone'].ids)!=701:
        raise ValueError('Exactly 701 matching TRAIN identities required')
    rows=[];path_hashes={}
    for domain,ds in datasets.items():
        paths='\n'.join(str(path.relative_to(Path(a.train_root)))+':'+str(label) for path,label in ds.rows)
        path_hashes[domain]=hashlib.sha256(paths.encode()).hexdigest()
        chunks=[];labels=[]
        with torch.inference_mode():
            for images,ids in DataLoader(ds,batch_size=32,shuffle=False,num_workers=a.num_workers,pin_memory=True):
                chunks.append(model(images.to(a.device)).cpu());labels.append(ids)
        rows.append(identity_centroids(torch.cat(chunks),torch.cat(labels))[0])
    values=fit_top128(rows)
    manifest=dict(schema='TOP128_CANONICAL_V1',teacher_sha256=teacher_sha,teacher_config_sha256=config_sha,
        image_size=a.image_size,split='train',train_ids=701,bank_rows=1402,
        compatibility='canonical protocol resolution refit',teacher_checkpoint=str(Path(a.middle_checkpoint).resolve()),
        teacher_config=str(Path(a.middle_config).resolve()),train_path_identity_sha256=path_hashes,
        fitting_protocol='identity_centroids_drone_then_satellite_FP64_SVD_legacy32_sign_alignment',
        tensor_sha256={k:tensor_sha256(v) for k,v in values.items()},builder_sha256=file_sha256(__file__))
    assert file_sha256(a.middle_checkpoint)==teacher_sha and file_sha256(a.middle_config)==config_sha
    assert not audit['missing'] and not audit['unexpected'] and audit['sha256']==teacher_sha
    out.mkdir(parents=True)
    for key,value in values.items():
        with (out/(key+'.pt')).open('xb') as f:torch.save(value,f)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    load_top_source(out/'manifest.json',teacher_sha)
    print('TOP128_CANONICAL_REFIT=PASS',flush=True)
if __name__=='__main__':main()
