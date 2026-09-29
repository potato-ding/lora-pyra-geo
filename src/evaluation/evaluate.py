"""Unified formal Teacher/Middle/Student evaluation. No training dependencies."""
import argparse
import json
from pathlib import Path
import torch
from .model_loader import load_encoder as _load_checkpoint_encoder
from .metrics import getdist_1652_val_and_get_recall,run_sues_val_and_get_metrics,run_gta_val_and_get_metrics
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders,build_sues200_val_dataloaders,build_gta_val_dataloaders

def load_encoder(model_type, checkpoint, config=None, device="cuda", image_size=None):
    """Load a deployment model under its strict checkpoint contract."""
    return _load_checkpoint_encoder(model_type, checkpoint, config, device, image_size=image_size)


def verify_previous_u1652(path, load_audit, model_type, image_size):
    """Accept a prior formal U1652 result only for this exact checkpoint and resolution."""
    from .precision_contract import assert_best_reload_metrics
    previous = json.loads(path.read_text())
    if (previous.get('model_type') != model_type
            or previous.get('checkpoint_sha256') != load_audit['sha256']
            or previous.get('dataset') != 'u1652'
            or previous.get('protocol', {}).get('input_size') != image_size
            or previous.get('protocol', {}).get('BEST_MODEL_RELOAD_CONSISTENCY') != 'PASS'):
        raise RuntimeError("Existing U1652 result does not match this formal checkpoint")
    expected = load_audit['selection_metrics']
    if expected is None:
        raise RuntimeError("Missing train selection metrics")
    assert_best_reload_metrics(previous['results'], expected)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-type',required=True,help='teacher, middle, student; comparison adapters can be registered later')
    p.add_argument('--checkpoint',required=True);p.add_argument('--config')
    p.add_argument('--dataset',required=True,choices=['u1652','sues200','gta','anyvisloc','all'],help='all: U1652, SUES-200, GTA-UAV, AnyVisLoc')
    p.add_argument('--data-root',default='data');p.add_argument('--u1652-dir');p.add_argument('--sues200-dir');p.add_argument('--gta-dir');p.add_argument('--anyvisloc-dir')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--image-size',type=int,required=True,help='evaluation input size; must match our model checkpoint');p.add_argument('--device',default='cuda:0');p.add_argument('--output-dir',help='comparison-model output directory; formal models write beside their checkpoint')
    p.add_argument('--gta-split',choices=['cross-area'],default='cross-area')
    p.add_argument('--gta-query-mode',choices=['D2S'],default='D2S')
    p.add_argument('--certification-only',action='store_true',help='Extract once and compare independent metrics, without issuing formal results')
    p.add_argument('--feature-cache-dir',help='Explicit certification cache directory, not included in result packages')
    p.add_argument('--reuse-certified-cache',action='store_true',help='Formal metrics from hash-verified previously certified descriptors')
    args=p.parse_args(argv)
    if args.batch_size != 16:
        raise ValueError("Unified single-GPU evaluation requires batch_size=16")
    if args.image_size <= 0:
        raise ValueError("Evaluation image size must be positive")
    if args.model_type not in ("teacher", "middle", "student"):
        p.error(f"No evaluation model adapter registered for {args.model_type!r}")
    if args.reuse_certified_cache and not args.certification_only and args.model_type in ("student", "middle"):
        raise ValueError("U1652 reload verification requires fresh batch=16 extraction; cached batch provenance is unavailable")
    if args.dataset in ("anyvisloc", "all") and (args.certification_only or args.reuse_certified_cache):
        p.error("AnyVisLoc requires direct retrieval; descriptor certification cache is not supported")
    checkpoint_dir = Path(args.checkpoint).resolve().parent
    if args.certification_only:
        if args.output_dir is None:
            p.error("--certification-only requires a separate --output-dir")
    elif args.output_dir is None:
        args.output_dir = str(checkpoint_dir)
    elif Path(args.output_dir).resolve() != checkpoint_dir:
        p.error("Formal Teacher/Middle/Student results must be written beside best_model.pth")
    return args

@torch.no_grad()
def main(argv=None):
    args=parse_args(argv);device=torch.device(args.device)
    if device.type != 'cuda' or device.index not in (None, 0):
        raise ValueError('Formal evaluation requires logical cuda:0 on one visible GPU')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Formal evaluation requires exactly one visible GPU')
    if args.model_type=="student":
        from src.student.artifacts import require_valid_run
        require_valid_run(Path(args.checkpoint).resolve().parent)
    model,load_audit=load_encoder(args.model_type,args.checkpoint,args.config,device,image_size=args.image_size)
    if args.model_type=='middle' and not args.certification_only and load_audit.get('artifact_classification') != 'FORMAL_MIDDLE_CHECKPOINT':
        raise RuntimeError('LEGACY_CHECKPOINT: not certified for new formal Middle evaluation')
    if args.model_type=='student' and not args.certification_only and load_audit.get('artifact_classification') != 'FORMAL_STUDENT_CHECKPOINT':
        raise RuntimeError('LEGACY_CHECKPOINT: not certified for new formal Student evaluation')
    if args.model_type=='teacher' and not args.certification_only and load_audit.get('artifact_classification') != 'FORMAL_TEACHER_CHECKPOINT':
        raise RuntimeError('LEGACY_CHECKPOINT: not certified for new formal Teacher evaluation')
    signature=load_audit['precision_signature']
    image_size=signature['image_size']
    if args.image_size is not None and args.image_size!=image_size:
        raise RuntimeError('PRECISION_OR_RELOAD_CONTRACT_FAILURE: image size')
    if not args.certification_only and not load_audit['runtime_precision']['train_selection_signature_verified']:
        raise RuntimeError('PRECISION_OR_RELOAD_CONTRACT_FAILURE: checkpoint lacks train selection signature; legacy run is not V1 certified')
    output=Path(args.output_dir)
    if args.certification_only:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Certification output must be a fresh directory: "+str(output))
        output.mkdir(parents=True,exist_ok=True)
        (output/'checkpoint_load.json').write_text(json.dumps(load_audit,indent=2))
    else:
        if output.resolve() != Path(args.checkpoint).resolve().parent:
            raise RuntimeError("Formal results must share the checkpoint directory")
        if not output.is_dir():
            raise FileNotFoundError(output)
        previous_u1652 = output/'test_1652.json'
        if args.dataset != "u1652" and previous_u1652.exists():
            verify_previous_u1652(previous_u1652, load_audit, args.model_type, image_size)
            result_names = []
        else:
            result_names = ["test_1652.json"]
        if args.dataset in ("sues200", "all"):
            result_names.append("test_sues200.json")
        if args.dataset in ("gta", "all"):
            result_names.append("test_gta_cross_area_d2s.json")
        if args.dataset in ("anyvisloc", "all"):
            result_names.append("test_anyvisloc.json")
        for name in result_names:
            if (output/name).exists():
                raise FileExistsError("Evaluation result already exists: "+str(output/name))
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
            result=run_sues_val_and_get_metrics(model,*pair,device,horizontal_flip=True)
            return {k:result[k] for k in ('R@1','R@5','R@10','R@top1','AP')}
        result=run_gta_val_and_get_metrics(model,*pair,device)
        return {k:result[k] for k in ('R@1','R@5','R@10','R@top1','AP',
                                      'SDM@1','SDM@3','SDM@5','DIS@1','DIS@3','DIS@5')}
    roots={'u1652':args.u1652_dir or str(Path(args.data_root)/'U1652'),
           'sues200':args.sues200_dir or str(Path(args.data_root)/'SUES-200/SUES-200-512x512'),
           'gta':args.gta_dir or str(Path(args.data_root)/'GTA-UAV-LR/GTA-UAV-LR-baidu'),
           'anyvisloc':args.anyvisloc_dir or str(Path(args.data_root)/'test_anyvisloc_scenes01_02')}
    datasets=['u1652','sues200','gta','anyvisloc'] if args.dataset=='all' else [args.dataset]
    if not args.certification_only and 'u1652' not in datasets and not (output/'test_1652.json').exists():
        datasets.insert(0,'u1652')
    for dataset in datasets:
        common=dict(img_size=[image_size,image_size],data_dir=roots[dataset],batch_size=args.batch_size,num_workers=args.num_workers)
        protocol={'input_size':image_size,'preprocessing':'canonical deterministic','augmentation':False}
        results={}
        if dataset=='u1652':
            if args.model_type=='teacher' and not args.certification_only and not args.reuse_certified_cache:
                from .u1652_canonical import evaluate_u1652_single_gpu_canonical
                results=evaluate_u1652_single_gpu_canonical(model, image_size=image_size,
                    device=device, data_dir=roots[dataset], num_workers=args.num_workers,
                    batch_size=args.batch_size)
            elif args.model_type=='middle' and not args.certification_only and not args.reuse_certified_cache:
                from .middle_canonical import evaluate_middle_u1652_canonical
                results=evaluate_middle_u1652_canonical(model,image_size=image_size,
                    device=device,data_dir=roots[dataset],num_workers=args.num_workers,
                    batch_size=args.batch_size)
            elif args.model_type=='student' and not args.certification_only and not args.reuse_certified_cache:
                from .student_canonical import evaluate_student_u1652_canonical
                results=evaluate_student_u1652_canonical(model,image_size=image_size,
                    device=device,data_dir=roots[dataset],num_workers=args.num_workers)
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
            name='test_sues200.json';protocol.update(split='Testing',zero_shot=True,
                heights=['150m','200m','250m','300m'],directions=['D2S','S2D'],
                horizontal_flip=True,feature_fusion='sum_then_l2',similarity='cosine')
        elif dataset=='gta':
            loaders=build_gta_val_dataloaders(**common,split_type='cross-area',query_mode='D2S',mode='pos')
            results['D2S']=evaluate_pair(dataset,'D2S',loaders['D2S'])
            name='test_gta_cross_area_d2s.json';protocol.update(split='cross-area',query_mode='D2S')
            protocol.update(loaders['D2S'][0].dataset.protocol_audit)
            protocol.update(DIS_unit='meter (m)',SDM_scale='unitless [0,1]',
                            similarity='cosine',positive_pairs='pair_pos_sate_img_list',
                            postprocess_matching=False)
        elif dataset=='anyvisloc':
            from .anyvisloc import evaluate_anyvisloc, validate_anyvisloc_root
            validate_anyvisloc_root(roots[dataset])
            results=evaluate_anyvisloc(model,roots[dataset],image_size,device,batch_size=args.batch_size)
            name='test_anyvisloc.json'
            protocol.update(scene_subset=['Scene_01','Scene_02'],subset_identity='current_public_release_01_02',
                paper_table2_subset_verified=False,reference_mode='aerial',pose_priori='yp',
                patch_scale=1.0,retrieval_cover_percent=50,similarity='cosine',
                retrieval_ks=[1,3,5],pdm_lambda=6.0,pdm_alpha=0.9,
                recall_unit="fraction [0,1] (paper R@K reports percent)",pdm_unit="fraction [0,1]",
                gallery='per-query aerial map sliding-window tiles',
                preprocessing='model-native input size, RGB, ImageNet normalization',
                query_resize='Pillow BICUBIC',gallery_resize='OpenCV INTER_LINEAR',
                postprocess_matching=False)
        else:
            raise ValueError(dataset)
        payload={'model_type':args.model_type,'checkpoint':str(Path(args.checkpoint).resolve()),
                 'checkpoint_sha256':load_audit['sha256'],'dataset':dataset,
                 'protocol':protocol,'descriptor':{'dim':model.descriptor_dim,'normalized':True,'dtype':'float32'},
                 'runtime_precision':load_audit['runtime_precision'],'precision_signature':signature,'results':results}
        if args.model_type=="student" and dataset=="u1652":
            payload["u1652_eval_batch_size"]=args.batch_size
        if not args.certification_only:
            with (output/name).open('x') as result_file:
                result_file.write(json.dumps(payload,indent=2)+'\n')
            print(json.dumps(payload),flush=True)
    if args.certification_only:
        with (output/'metric_certification.json').open('x') as result_file:
            result_file.write(json.dumps({'pass':all(r['pass'] for r in records),'records':records,'checkpoint':load_audit},indent=2))

if __name__=='__main__':main()
