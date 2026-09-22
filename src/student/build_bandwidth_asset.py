"""Fresh TRAIN extraction: immutable V128/R32 prefixes extended to V256/R128."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .build_bank import TrainImages, ROOT
from .subspace import identity_centroids
from .dual_stst import file_sha256,load_stst_asset
from .part1 import load_extended_asset
from .bandwidth_assets import build_tensors, tensor_sha256, SHAPES, SCHEMA, RANDOM_EXTENSION_SEED, ORTHO_ATOL
from .artifacts import write_json

def main(argv=None):
    from src.evaluation.model_loader import load_encoder
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--middle-checkpoint',required=True);p.add_argument('--middle-run-config',required=True)
    p.add_argument('--anchor-asset',required=True);p.add_argument('--original-asset',required=True);p.add_argument('--asset-output',required=True)
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
    original=load_extended_asset(args.anchor_asset,DEFAULT_BANK,teacher_sha)
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
    asset,checks=build_tensors(bank_rows,original)
    asset['metadata']=dict(dataset='University-1652',split='train',train_only=True,
        teacher_sha256=teacher_sha,teacher_config_sha256=file_sha256(middle_config),original_stst_asset_sha256=old_sha,train_ids=701,bank_rows=1402,teacher_dim=768,
        schema=SCHEMA,top_max_dim=256,random_A_dim=32,random_A_source='original_D0',random_A_seed=20260808,
        random_B_dim=32,random_B_seed=RANDOM_EXTENSION_SEED,random64_dim=64,
        input_definition='original identity_centroids; sorted numeric identities; drone701 then satellite701',
        extraction='fresh images, deterministic transform, certified Middle encoder, batch32',
        image_counts={d:len(ds) for d,ds in datasets.items()},image_size=args.image_size,SVD_dtype='float64',
        centering='exact ORIGINAL teacher_mean converted to float64',source_file_sha256=file_sha256(Path(__file__)),
        target_code_sha256=file_sha256(Path(__file__).with_name('bandwidth_assets.py')),checks=checks,
        anchor_asset_path=str(Path(args.anchor_asset).resolve()),anchor_asset_sha256=file_sha256(args.anchor_asset),
        available_top_dims=[128,256],available_random_dims=[32,64,128],
        random_extension_seed=RANDOM_EXTENSION_SEED,generator_type='torch.Generator(device=cpu), independent',
        tensor_sha256={k:tensor_sha256(asset[k]) for k in SHAPES},
        middle_checkpoint_path=str(checkpoint),middle_config_path=str(middle_config),
        orthogonality_atol=ORTHO_ATOL,FP32_contract=True,
        old_mu_sha256=tensor_sha256(original['teacher_mean']),
        old_V128_sha256=tensor_sha256(original['top128_basis']),old_R32_sha256=tensor_sha256(original['random32_A']),
        new_V256_sha256=tensor_sha256(asset['top256_basis']),new_R64_sha256=tensor_sha256(asset['random64_basis']),
        new_R128_sha256=tensor_sha256(asset['random128_basis']))
    assert file_sha256(DEFAULT_BANK)==old_sha and file_sha256(checkpoint)==teacher_sha
    with EXTENDED.open('xb') as f: torch.save(asset,f)
    load_extended_asset(EXTENDED,DEFAULT_BANK,teacher_sha)
    write_json(PREFLIGHT/'ASSET_MANIFEST.json',dict(asset_path=str(EXTENDED),asset_sha256=file_sha256(EXTENDED),metadata=asset['metadata'],
        calibration_path=str(Path(args.anchor_asset).parent/'diagnostic_inputs.pt'),
        calibration_sha256=file_sha256(Path(args.anchor_asset).parent/'diagnostic_inputs.pt'),
        anchor_files={str(p):file_sha256(p) for p in Path(args.anchor_asset).parent.iterdir() if p.is_file()}))
    write_json(PREFLIGHT/'EXTENDED_ASSET_REPORT.json',dict(**checks,FINAL_MIDDLE_SHA256=teacher_sha,
        ORIGINAL_STST_ASSET_SHA256=old_sha,EXTENDED_STST_ASSET_SHA256=file_sha256(EXTENDED),
        extended_path=str(EXTENDED),metadata=asset['metadata'],strict_load=audit,seconds=time.time()-start,ASSET_PASS=True))
    print(json.dumps(checks,indent=2),flush=True)
    print('EXTENDED_ASSET_SHA256='+file_sha256(EXTENDED),flush=True)

if __name__=='__main__':main()
