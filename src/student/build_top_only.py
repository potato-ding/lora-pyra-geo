"""Build canonical TRAIN mu/V128; preserve exact R224 reconstruction and permit R384 refitting."""
import argparse,json,hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .build_bank import TrainImages
from .subspace import identity_centroids
from .bandwidth_assets import tensor_sha256
from .dual_stst import file_sha256

def main():
    from src.evaluation.model_loader import load_encoder
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--image-size',type=int,choices=(224,384),default=224)
    p.add_argument('--middle-checkpoint');p.add_argument('--middle-config')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    inv=json.loads(Path(a.inventory).read_text());meta=inv['bank_validation.json']['content']['metadata']
    checkpoint=meta['teacher_checkpoint'];cfg=meta['teacher_config']
    assert file_sha256(checkpoint)==meta['teacher_sha256'] and file_sha256(cfg)==meta['teacher_config_sha256']
    if a.image_size==384:
        if not a.middle_checkpoint or not a.middle_config:raise ValueError('R384 refit requires explicit Middle source')
        checkpoint=a.middle_checkpoint;cfg=a.middle_config
        config=json.loads(Path(cfg).read_text())
        if config['data']['input_size']!=384 or config.get('sam',{}).get('enabled'):raise ValueError('R384 non-SAM required')
    elif a.middle_checkpoint or a.middle_config:raise ValueError('Historical R224 source must remain unchanged')
    teacher_sha=file_sha256(checkpoint);config_sha=file_sha256(cfg)
    torch.set_num_threads(8)
    model,audit=load_encoder('middle',checkpoint,cfg,'cuda:0')
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    rows=[]
    for domain in ('drone','satellite'):
        ds=TrainImages(domain,meta['train_root'],a.image_size)
        paths='\n'.join(str(path.relative_to(Path(meta['train_root'])))+':'+str(label) for path,label in ds.rows)
        assert hashlib.sha256(paths.encode()).hexdigest()==meta['train_path_identity_sha256'][domain]
        chunks=[];labels=[]
        with torch.inference_mode():
            for step,(images,ids) in enumerate(DataLoader(ds,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)):
                chunks.append(model(images.cuda()).cpu());labels.append(ids)
                if step%100==0:print(json.dumps(dict(domain=domain,step=step,total=len(ds))),flush=True)
        rows.append(identity_centroids(torch.cat(chunks),torch.cat(labels))[0])
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
    expected=inv['top128_random32.pt']['tensors']
    for key,tensor in values.items():
        actual=tensor_sha256(tensor);print(key,actual,flush=True)
        if a.image_size==224 and actual!=expected[key]['tensor_sha256']:raise RuntimeError('BLOCKER historical tensor mismatch: '+key)
    manifest=dict(schema='TOP128_CANONICAL_V1',teacher_sha256=teacher_sha,teacher_config_sha256=config_sha,
        image_size=a.image_size,split='train',train_ids=701,bank_rows=1402,
        compatibility='historical tensor SHA256 exact' if a.image_size==224 else 'canonical protocol resolution refit',
        teacher_checkpoint=str(checkpoint),teacher_config=str(cfg),
        train_path_identity_sha256=meta['train_path_identity_sha256'],
        fitting_protocol='identity_centroids_drone_then_satellite_FP64_SVD_legacy32_sign_alignment',
        tensor_sha256={k:tensor_sha256(v) for k,v in values.items()},historical_inventory_sha256=file_sha256(a.inventory),
        builder_sha256=file_sha256(__file__))
    assert file_sha256(checkpoint)==teacher_sha and file_sha256(cfg)==config_sha
    assert not audit['missing'] and not audit['unexpected'] and audit['sha256']==teacher_sha
    out.mkdir(parents=True)
    for key,value in values.items():
        with (out/(key+'.pt')).open('xb') as f:torch.save(value,f)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('TOP128_HISTORICAL_BITWISE_MATCH=PASS' if a.image_size==224 else 'TOP128_R384_CANONICAL_REFIT=PASS',flush=True)
if __name__=='__main__':main()
