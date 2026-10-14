"""Fixed-protocol evaluation for the downloaded University-1652 paper checkpoints."""
import argparse,json,time
from pathlib import Path
import torch
from .comparison_models import load_comparison,SPECS,ROOT
from .protocols import RETRIEVAL_PROTOCOL,gta_eval_splits
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders,build_sues200_val_dataloaders,build_gta_val_dataloaders
from .metrics import getdist_1652_val_and_get_recall,run_sues_val_and_get_metrics,run_gta_val_and_get_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=SPECS)
    p.add_argument('--dataset',required=True,choices=('u1652','sues200','gta'))
    p.add_argument('--gta-split',choices=('cross-area','same-area','both'),default='both')
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--compatibility-trial',action='store_true',help='Write unconfirmed model adaptation results separately')
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    a=p.parse_args();device=torch.device(a.device);torch.set_num_threads(2)
    if a.compatibility_trial and a.model not in ('mccg','infogeo_rd_trial','spikevimformer'):raise ValueError('Compatibility trial is adapted models only')
    if a.model in ('mccg','infogeo_rd_trial','spikevimformer') and not a.smoke_only and not a.compatibility_trial:
        raise ValueError('Adapted checkpoint forward is unconfirmed; pass --compatibility-trial to keep results separate')
    if device.type=='cuda' and torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
    if device.type=='cpu' and not a.smoke_only:raise ValueError('CPU is for adapter smoke checks only')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,audit,size=load_comparison(a.model,device)
    if a.smoke_only:
        with torch.inference_mode():z=model.encode(torch.zeros(2,3,size,size,device=device))
        assert torch.allclose(z.norm(dim=1),torch.ones(2,device=device),atol=1e-5)
        print(json.dumps(dict(audit=audit,smoke_descriptor_shape=list(z.shape),finite=True)),flush=True);return
    result_folder='infogeo_rd_trial' if a.model=='infogeo_rd_trial' else ('compatibility_trial' if a.compatibility_trial else 'fixed_protocol')
    out=ROOT/'src/checkpoint/others'/SPECS[a.model][0]/'results'/result_folder
    out.mkdir(parents=True,exist_ok=True)
    splits=gta_eval_splits(a.gta_split) if a.dataset=='gta' else (None,)
    for split in splits:
        filename=('gta_same_area.json' if split=='same-area' else a.dataset+'.json')
        if (out/filename).exists():raise FileExistsError(out/filename)
    for split in splits:
        dest=out/(('gta_same_area.json' if split=='same-area' else a.dataset+'.json'))
        if dest.exists():raise FileExistsError(dest)
        common=dict(img_size=[size,size],batch_size=16,num_workers=a.num_workers)
        start=time.monotonic();results={}
        protocol=dict(input_size=size,batch_size=16,channels='RGB',resize='OpenCV INTER_LINEAR_EXACT',normalization='ImageNet',
                      horizontal_flip=False,multiscale=False,similarity='cosine',target_finetuning=False)
        if a.dataset=='u1652':
            protocol['retrieval_protocol']=RETRIEVAL_PROTOCOL
            loaders=build_1652_val_dataloaders(data_dir=str(a.data_root/'U1652'),distributed=False,**common)
            for direction,pair in loaders.items():
                results[direction]=dict(zip(('R@1','R@5','R@10','AP'),getdist_1652_val_and_get_recall(model,*pair,device)))
        elif a.dataset=='sues200':
            protocol['retrieval_protocol']=RETRIEVAL_PROTOCOL
            protocol.update(gallery_policy='non-query identities marked -1 and excluded after ranking', experiment='University-1652 to SUES-200 zero-shot', preprocessing_policy='DAC/MEAN linear resize and single forward', resize='OpenCV INTER_LINEAR_EXACT', horizontal_flip=False, feature_fusion='none')
            loaders=build_sues200_val_dataloaders(data_dir=str(a.data_root/'SUES-200/SUES-200-512x512'),heights=['150','200','250','300'],**common)
            for height,directions in loaders.items():
                results[height]={d:run_sues_val_and_get_metrics(model,*pair,device) for d,pair in directions.items()}
        elif a.dataset=='gta':
            protocol.update(split=split,query_mode='D2S',DIS_unit='meter (m)',SDM_scale='unitless [0,1]',
                            positive_pairs='pair_pos_sate_img_list',postprocess_matching=False)
            pair=build_gta_val_dataloaders(data_dir=str(a.data_root/'GTA-UAV-LR/GTA-UAV-LR-baidu'),split_type=split,query_mode='D2S',mode='pos',**common)['D2S']
            protocol.update(pair[0].dataset.protocol_audit)
            results['D2S']=run_gta_val_and_get_metrics(model,*pair,device)
        payload=dict(model=a.model,dataset=a.dataset,protocol=protocol,checkpoint=audit,results=results,elapsed_seconds=time.monotonic()-start)
        with dest.open('x') as f:json.dump(payload,f,indent=2)
        print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
