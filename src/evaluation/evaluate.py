"""Unified formal Teacher/Middle/Student evaluation. No training dependencies."""
import argparse
import json
from pathlib import Path
import torch
import time
from .model_loader import load_encoder
from .metrics import getdist_1652_val_and_get_recall,run_sues_val_and_get_metrics,run_gta_val_and_get_metrics
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders,build_sues200_val_dataloaders,build_gta_val_dataloaders

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-type',required=True,choices=['teacher','middle','student'])
    p.add_argument('--checkpoint',required=True);p.add_argument('--config')
    p.add_argument('--dataset',required=True,choices=['u1652','sues200','gta','all'])
    p.add_argument('--data-root',default='data');p.add_argument('--u1652-dir');p.add_argument('--sues200-dir');p.add_argument('--gta-dir')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--device',default='cuda');p.add_argument('--output-dir',required=True)
    p.add_argument('--gta-split',choices=['cross-area'],default='cross-area')
    p.add_argument('--gta-query-mode',choices=['D2S'],default='D2S')
    p.add_argument('--certification-only',action='store_true',help='Extract once and compare independent metrics, without issuing formal results')
    p.add_argument('--feature-cache-dir',help='Explicit certification cache directory, not included in result packages')
    p.add_argument('--reuse-certified-cache',action='store_true',help='Formal metrics from hash-verified previously certified descriptors')
    return p.parse_args(argv)

@torch.no_grad()
def main(argv=None):
    args=parse_args(argv);device=torch.device(args.device)
    model,load_audit=load_encoder(args.model_type,args.checkpoint,args.config,device)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    (output/'checkpoint_load.json').write_text(json.dumps(load_audit,indent=2))
    records=[]
    if args.certification_only or args.reuse_certified_cache:
        if not args.feature_cache_dir:raise ValueError('--feature-cache-dir required')
        from .certification import certify_pair,reuse_pair
        cache=Path(args.feature_cache_dir);cache.mkdir(parents=True,exist_ok=True)
        identity=cache/'model_identity.json'
        if identity.exists():
            previous=json.loads(identity.read_text())
            if previous['sha256']!=load_audit['sha256']:raise RuntimeError('Cache checkpoint identity mismatch')
        else:
            if args.reuse_certified_cache:raise RuntimeError('No certified model identity')
            identity.write_text(json.dumps(load_audit,indent=2))
    def evaluate_pair(dataset,direction,pair):
        if args.certification_only:
            record=certify_pair(model,pair,dataset,direction,device,args.feature_cache_dir)
            records.append(record);return record['clean']
        if args.reuse_certified_cache:
            return reuse_pair(model,pair,dataset,direction,device,args.feature_cache_dir)
        if dataset=='u1652':
            r1,r5,_,ap=getdist_1652_val_and_get_recall(model,*pair,device)
            return {'R@1':r1,'R@5':r5,'AP':ap}
        if dataset=='sues200':
            result=run_sues_val_and_get_metrics(model,*pair,device,horizontal_flip=False)
            return {k:result[k] for k in ('R@1','AP')}
        result=run_gta_val_and_get_metrics(model,*pair,device)
        return {k:result[k] for k in ('R@1','AP','DIS@1','SDM@3')}
    roots={'u1652':args.u1652_dir or str(Path(args.data_root)/'U1652'),
           'sues200':args.sues200_dir or str(Path(args.data_root)/'SUES-200/SUES-200-512x512'),
           'gta':args.gta_dir or str(Path(args.data_root)/'GTA-UAV-LR/GTA-UAV-LR-baidu')}
    datasets=['u1652','sues200','gta'] if args.dataset=='all' else [args.dataset]
    for dataset in datasets:
        common=dict(img_size=[224,224],data_dir=roots[dataset],batch_size=args.batch_size,num_workers=args.num_workers)
        protocol={'input_size':224,'preprocessing':'canonical deterministic','augmentation':False}
        results={}
        if dataset=='u1652':
            loaders=build_1652_val_dataloaders(**common)
            for direction,pair in loaders.items():
                results[direction]=evaluate_pair(dataset,direction,pair)
            name='test_1652.json';protocol.update(split='test')
        elif dataset=='sues200':
            loaders=build_sues200_val_dataloaders(**common,heights=['150','200','250','300'])
            for height,directions in loaders.items():
                results[height]={}
                for direction,pair in directions.items():
                    results[height][direction]=evaluate_pair(dataset,height+'_'+direction,pair)
            name='test_sues200.json';protocol.update(split='Testing',zero_shot=True)
        else:
            loaders=build_gta_val_dataloaders(**common,split_type='cross-area',query_mode='D2S',mode='pos')
            results['D2S']=evaluate_pair(dataset,'D2S',loaders['D2S'])
            name='test_gta_cross_area_d2s.json';protocol.update(split='cross-area',query_mode='D2S')
            protocol.update(loaders['D2S'][0].dataset.protocol_audit)
            protocol.update(DIS1_unit='meter (m)',SDM3_scale='percentage')
        payload={'model_type':args.model_type,'checkpoint':str(Path(args.checkpoint).resolve()),'dataset':dataset,
                 'protocol':protocol,'descriptor':{'dim':model.descriptor_dim,'normalized':True,'dtype':'float32'},'results':results}
        if not args.certification_only:
            (output/name).write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
    if args.certification_only:
        (output/'metric_certification.json').write_text(json.dumps({'pass':all(r['pass'] for r in records),'records':records,'checkpoint':load_audit},indent=2))

if __name__=='__main__':main()
