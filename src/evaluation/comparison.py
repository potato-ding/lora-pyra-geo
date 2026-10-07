"""Fixed-protocol evaluation for the downloaded University-1652 paper checkpoints."""
import argparse,json,time
from pathlib import Path
import torch
from .comparison_models import load_comparison,SPECS,ROOT
from .protocols import RETRIEVAL_PROTOCOL,GTA_PROTOCOL
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders,build_sues200_val_dataloaders,build_gta_val_dataloaders
from .metrics import getdist_1652_val_and_get_recall,run_sues_val_and_get_metrics,run_gta_val_and_get_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=SPECS)
    p.add_argument('--dataset',required=True,choices=('u1652','sues200','gta','anyvisloc'))
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    a=p.parse_args();device=torch.device(a.device);torch.set_num_threads(2)
    if device.type=='cuda' and torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
    if device.type=='cpu' and not a.smoke_only:raise ValueError('CPU is for adapter smoke checks only')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,audit,size=load_comparison(a.model,device)
    if a.smoke_only:
        with torch.inference_mode():z=model.encode(torch.zeros(2,3,size,size,device=device))
        assert torch.allclose(z.norm(dim=1),torch.ones(2,device=device),atol=1e-5)
        print(json.dumps(dict(audit=audit,smoke_descriptor_shape=list(z.shape),finite=True)),flush=True);return
    out=ROOT/'src/checkpoint/others'/SPECS[a.model][0]/'results/fixed_protocol'
    out.mkdir(parents=True,exist_ok=True);dest=out/(a.dataset+'.json')
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
        protocol['gta_protocol']=GTA_PROTOCOL
        pair=build_gta_val_dataloaders(data_dir=str(a.data_root/'GTA-UAV-LR/GTA-UAV-LR-baidu'),split_type='cross-area',query_mode='D2S',mode='pos',**common)['D2S']
        protocol.update(pair[0].dataset.protocol_audit)
        results['D2S']=run_gta_val_and_get_metrics(model,*pair,device)
    else:
        from .anyvisloc import evaluate_anyvisloc, validate_anyvisloc_root
        root=a.data_root/'test_anyvisloc_scenes01_02'
        scenes=validate_anyvisloc_root(root)
        progress_path=out/'anyvisloc_progress.jsonl'
        if progress_path.exists():raise FileExistsError(progress_path)
        protocol.pop('channels');protocol.pop('resize')
        protocol.update(scene_subset=['Scene_01','Scene_02'],subset_identity='current_public_release_01_02',
            reference_mode='aerial',pose_priori='yp',patch_scale=1.0,retrieval_cover_percent=50,
            zero_shot=True,postprocess_matching=False,paper_table2_subset_verified=False,
            training_dataset='University-1652',query_channels='RGB',gallery_channels='official_bgr',
            normalization='ImageNet mean/std applied in input array order',
            query_resize='Pillow bicubic',gallery_resize='OpenCV INTER_LINEAR',
            retrieval_ks=[1,3,5],pdm_lambda=6.0,pdm_alpha=0.9,metric_unit='Recall/PDM fraction [0,1]',
            query_counts={scene:len(values[-1]) for scene,values in scenes.items()})
        def progress(row):
            with progress_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
        results=evaluate_anyvisloc(model,root,size,device,batch_size=16,
                                  query_callback=progress,gallery_channel_mode='official_bgr')
    payload=dict(model=a.model,dataset=a.dataset,protocol=protocol,checkpoint=audit,results=results,elapsed_seconds=time.monotonic()-start)
    with dest.open('x') as f:json.dump(payload,f,indent=2)
    print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
