"""Fresh TRAIN extraction and strictly nested extension of the unchanged D0 bank."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .build_bank import TrainImages, ROOT
from .subspace import identity_centroids
from .dual_stst import file_sha256,load_stst_asset
from .part1 import build_extended_tensors,load_extended_asset
from .artifacts import write_json

def main(argv=None):
    from src.evaluation.model_loader import load_encoder
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--middle-checkpoint',required=True);p.add_argument('--middle-run-config',required=True)
    p.add_argument('--original-asset',required=True);p.add_argument('--asset-output',required=True)
    p.add_argument('--train-root',default=str(ROOT/'data/U1652/train'))
    p.add_argument('--image-size',type=int,default=224);p.add_argument('--device',default='cuda:0')
    args=p.parse_args(argv)
    EXTENDED=Path(args.asset_output).resolve();DEFAULT_BANK=Path(args.original_asset).resolve()
    checkpoint=Path(args.middle_checkpoint).resolve();middle_config=Path(args.middle_run_config).resolve()
    config=json.loads(middle_config.read_text())
    if config.get('sam',{}).get('enabled') or config['data']['input_size']!=args.image_size:raise ValueError('New-chain Middle protocol mismatch')
    EXTENDED.parent.mkdir(parents=True,exist_ok=True);PREFLIGHT=EXTENDED.parent
    if EXTENDED.exists(): raise FileExistsError(EXTENDED)
    torch.set_num_threads(4)
    teacher_sha=file_sha256(checkpoint);old_sha=file_sha256(DEFAULT_BANK)
    original=load_stst_asset(DEFAULT_BANK,teacher_sha)
    datasets={d:TrainImages(d,args.train_root,args.image_size) for d in ['drone','satellite']}
    assert datasets['drone'].ids==datasets['satellite'].ids and len(datasets['drone'].ids)==701
    # Never reuse the previous audit's NOT_FOR_TRAINING cache.
    model,audit=load_encoder('middle',checkpoint,middle_config,args.device)
    features={};labels={};start=time.time()
    for domain,ds in datasets.items():
        chunks=[];targets=[]
        loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=8,pin_memory=True)
        with torch.inference_mode():
            for step,(images,ids) in enumerate(loader):
                chunks.append(model(images.to(args.device,non_blocking=True)).cpu());targets.append(ids)
                if step%200==0: print(json.dumps(dict(domain=domain,done=sum(len(x) for x in chunks),total=len(ds))),flush=True)
        features[domain]=torch.cat(chunks);labels[domain]=torch.cat(targets)
    centroids={d:identity_centroids(features[d],labels[d]) for d in features}
    assert centroids['drone'][1]==centroids['satellite'][1]
    bank_rows=torch.cat([centroids[d][0] for d in ['drone','satellite']]).double()
    asset,checks=build_extended_tensors(bank_rows,original,split='train')
    asset['metadata']=dict(dataset='University-1652',split='train',train_only=True,
        teacher_sha256=teacher_sha,teacher_config_sha256=file_sha256(middle_config),original_stst_asset_sha256=old_sha,train_ids=701,bank_rows=1402,teacher_dim=768,
        top_max_dim=128,random_A_dim=32,random_A_source='original_D0',random_A_seed=20260808,
        random_B_dim=32,random_B_seed=20260914,random64_dim=64,
        input_definition='original identity_centroids; sorted numeric identities; drone701 then satellite701',
        extraction='fresh images, deterministic transform, certified Middle encoder, batch32',
        image_counts={d:len(ds) for d,ds in datasets.items()},image_size=args.image_size,SVD_dtype='float64',
        centering='exact ORIGINAL teacher_mean converted to float64',source_file_sha256=file_sha256(Path(__file__)),
        target_code_sha256=file_sha256(Path(__file__).with_name('part1.py')),checks=checks)
    assert file_sha256(DEFAULT_BANK)==old_sha and file_sha256(checkpoint)==teacher_sha
    with EXTENDED.open('xb') as f: torch.save(asset,f)
    load_extended_asset(EXTENDED,DEFAULT_BANK,teacher_sha)
    write_json(PREFLIGHT/'EXTENDED_ASSET_REPORT.json',dict(**checks,FINAL_MIDDLE_SHA256=teacher_sha,
        ORIGINAL_STST_ASSET_SHA256=old_sha,EXTENDED_STST_ASSET_SHA256=file_sha256(EXTENDED),
        extended_path=str(EXTENDED),metadata=asset['metadata'],strict_load=audit,seconds=time.time()-start,ASSET_PASS=True))
    print(json.dumps(checks,indent=2),flush=True)
    print('EXTENDED_ASSET_SHA256='+file_sha256(EXTENDED),flush=True)

if __name__=='__main__':main()
