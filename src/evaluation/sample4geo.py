"""Sample4Geo model adapter and official AnyVisLoc evaluation.

External checkpoints have their own provenance, never a fabricated internal
train-selection/precision certification. Run one dataset per visible GPU.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'src/checkpoint/others/Sample4geo'
MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k_384'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Sample4GeoEncoder(nn.Module):
    descriptor_dim = 1024

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, images):
        # Shared extraction applies FP32 L2 to this single-forward descriptor.
        context = (torch.autocast('cuda', dtype=torch.float16)
                   if images.device.type == 'cuda' else nullcontext())
        with context:
            features = self.model(images)
        features = features.float()
        if features.ndim != 2 or features.shape[1] != self.descriptor_dim:
            raise RuntimeError(f'Invalid Sample4Geo descriptor shape: {features.shape}')
        if not torch.isfinite(features).all():
            raise RuntimeError('Nonfinite Sample4Geo descriptor')
        return features

    @torch.no_grad()
    def encode(self, images):
        # AnyVisLoc consumes already-normalized descriptors.
        return F.normalize(self(images), dim=-1)


def load_encoder(checkpoint, source, device):
    model_file = source / 'sample4geo/model.py'
    spec = importlib.util.spec_from_file_location('_official_sample4geo_model', model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TimmModel(MODEL_NAME, pretrained=False, img_size=384)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Deliberately strict: no silently ignored keys or fallback backbone weights.
    loaded = model.load_state_dict(state, strict=True)
    config = model.get_config()
    if tuple(config['mean']) != (0.485, 0.456, 0.406) or tuple(config['std']) != (0.229, 0.224, 0.225):
        raise RuntimeError('Dataset transforms do not match model normalization')
    encoder = Sample4GeoEncoder(model.to(device)).eval()
    encoder.requires_grad_(False)
    commit_file = source / 'UPSTREAM_COMMIT.txt'
    commit = commit_file.read_text().strip() if commit_file.exists() else subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    import timm
    audit = dict(model='Sample4Geo', backbone=MODEL_NAME, training_dataset='University-1652',
                 checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                 upstream='https://github.com/Skyy93/Sample4Geo', upstream_commit=commit,
                 model_source_sha256=sha256(model_file), evaluator_sha256=sha256(__file__),
                 image_size=384, state_keys=len(state), missing=loaded.missing_keys,
                 unexpected=loaded.unexpected_keys, parameters=sum(p.numel() for p in model.parameters()),
                 torch_version=torch.__version__, timm_version=timm.__version__,
                 parameters_dtype='float32', inference_autocast='cuda_float16',
                 descriptor_dtype='float32', similarity_dtype='float32',
                 normalization='single-forward L2 FP32', batch_size=16,
                 train_selection_certification='not_applicable_external_checkpoint',
                 shared_sources={str(p.relative_to(ROOT)): sha256(p) for p in [
                     ROOT/'src/utils/train_eval_utils.py', ROOT/'src/evaluation/anyvisloc.py',
                     ROOT/'src/dataset/teacher/val_dataloaders.py']})
    return encoder, audit


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=BASE/'weights/weights_e1_0.9515.pth')
    p.add_argument('--source',type=Path,default=BASE/'source')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--dataset',choices=['anyvisloc'],required=True)
    p.add_argument('--num-workers',type=int,default=4)
    args=p.parse_args(argv)
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one GPU')
    device=torch.device('cuda:0');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    prefix=args.dataset
    for suffix in ['.json','_load.json','_started.json','_progress.jsonl']:
        if (out/(prefix+suffix)).exists():raise FileExistsError(out/(prefix+suffix))
    model,audit=load_encoder(args.checkpoint.resolve(),args.source.resolve(),device)
    write_json(out/(prefix+'_load.json'),audit)
    write_json(out/(prefix+'_started.json'),dict(started_utc=datetime.now(timezone.utc).isoformat(),dataset=args.dataset))
    protocol=dict(input_size=384,training_dataset='University-1652',target_finetuning=False,batch_size=16)
    results={};counts={};start=time.perf_counter()
    def progress(row):
        with (out/(prefix+'_progress.jsonl')).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n');stream.flush()
    from src.evaluation.anyvisloc import evaluate_anyvisloc, official_protocol_metadata
    protocol.update(scene_subset=['Scene_01','Scene_02'],subset_identity='current_public_release_01_02',
        zero_shot=True,reference_mode='aerial',pose_priori='yp',patch_scale=1.0,retrieval_cover_percent=50,
        paper_table2_subset_verified=False,postprocess_matching=False,recall_ap_unit='Recall/PDM fraction [0,1]',
        query_channels='RGB',gallery_channels='RGB',
        normalization='ImageNet mean/std applied in input array order',
        query_resize='Pillow bicubic',gallery_resize='OpenCV INTER_LINEAR',retrieval_ks=[1,3,5],pdm_lambda=6.0,pdm_alpha=0.9)
    protocol.update(official_protocol_metadata())
    results=evaluate_anyvisloc(model,args.data_root/'test_anyvisloc_scenes01_02',384,device,batch_size=16,query_callback=progress)
    payload=dict(model_type='sample4geo',checkpoint_sha256=audit['checkpoint_sha256'],dataset=args.dataset,
        protocol=protocol,counts=counts,descriptor=dict(dim=1024,normalized=True,dtype='float32'),
        runtime_precision=model.anyvisloc_runtime,results=results,elapsed_seconds=time.perf_counter()-start,completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/(args.dataset+'.json'),payload)
    print('COMPLETE '+str(out/(args.dataset+'.json')),flush=True)


if __name__=='__main__':
    main()
