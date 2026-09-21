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
    p.add_argument('--image-size',type=int,default=None);p.add_argument('--device',default='cuda');p.add_argument('--output-dir',required=True)
    p.add_argument('--gta-split',choices=['cross-area'],default='cross-area')
    p.add_argument('--gta-query-mode',choices=['D2S'],default='D2S')
    p.add_argument('--certification-only',action='store_true',help='Extract once and compare independent metrics, without issuing formal results')
    p.add_argument('--feature-cache-dir',help='Explicit certification cache directory, not included in result packages')
    p.add_argument('--reuse-certified-cache',action='store_true',help='Formal metrics from hash-verified previously certified descriptors')
    args=p.parse_args(argv)
    if args.model_type in ("student","middle") and not args.certification_only:
        from src.student.artifacts import require_u1652_eval_batch_size
        require_u1652_eval_batch_size(args.batch_size)
        if args.reuse_certified_cache:
            raise ValueError("U1652 reload verification requires fresh batch=32 extraction; cached batch provenance is unavailable")
    return args

@torch.no_grad()
def main(argv=None):
    args=parse_args(argv);device=torch.device(args.device)
    if args.model_type=="student":
        from src.student.artifacts import require_valid_run
        require_valid_run(Path(args.checkpoint).resolve().parent)
    model,load_audit=load_encoder(args.model_type,args.checkpoint,args.config,device,image_size=args.image_size or 224)
    signature=load_audit['precision_signature']
    image_size=signature['image_size']
    if args.image_size is not None and args.image_size!=image_size:
        raise RuntimeError('PRECISION_OR_RELOAD_CONTRACT_FAILURE: image size')
    if not args.certification_only and not load_audit['runtime_precision']['train_selection_signature_verified']:
        raise RuntimeError('PRECISION_OR_RELOAD_CONTRACT_FAILURE: checkpoint lacks train selection signature; legacy run is not V1 certified')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Evaluation output must be a fresh directory: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
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
            if previous.get('runtime_precision')!=load_audit['runtime_precision']:
                raise RuntimeError('Cache runtime precision mismatch; use a fresh certification cache directory')
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
    if not args.certification_only and 'u1652' not in datasets:datasets.insert(0,'u1652')
    for dataset in datasets:
        common=dict(img_size=[image_size,image_size],data_dir=roots[dataset],batch_size=args.batch_size,num_workers=args.num_workers)
        protocol={'input_size':image_size,'preprocessing':'canonical deterministic','augmentation':False}
        results={}
        if dataset=='u1652':
            if args.model_type=='teacher' and not args.certification_only and not args.reuse_certified_cache:
                from .u1652_canonical import evaluate_u1652_single_gpu_canonical
                results=evaluate_u1652_single_gpu_canonical(model, image_size=image_size,
                    device=device, data_dir=roots[dataset], num_workers=args.num_workers)
            else:
                loaders=build_1652_val_dataloaders(**common)
                if args.model_type=='teacher':
                    from .u1652_canonical import canonical_loader
                    loaders={d:tuple(canonical_loader(loader) for loader in pair) for d,pair in loaders.items()}
                for direction,pair in loaders.items():
                    results[direction]=evaluate_pair(dataset,direction,pair)
            name='test_1652.json';protocol.update(split='test')
            if not args.certification_only:
                from .precision_contract import assert_best_reload_metrics
                if load_audit['selection_metrics'] is None:raise RuntimeError('Missing train selection metrics')
                protocol['BEST_MODEL_RELOAD_CONSISTENCY']=assert_best_reload_metrics(results,load_audit['selection_metrics'])
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
                 'protocol':protocol,'descriptor':{'dim':model.descriptor_dim,'normalized':True,'dtype':'float32'},
                 'runtime_precision':load_audit['runtime_precision'],'precision_signature':signature,'results':results}
        if args.model_type=="student" and dataset=="u1652":
            payload["u1652_eval_batch_size"]=args.batch_size
        if not args.certification_only:
            (output/name).write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
    if args.certification_only:
        (output/'metric_certification.json').write_text(json.dumps({'pass':all(r['pass'] for r in records),'records':records,'checkpoint':load_audit},indent=2))

if __name__=='__main__':main()
