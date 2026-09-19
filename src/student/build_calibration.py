"""Deterministic TRAIN-only initial Student descriptors for one-time RMS matching."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .build_bank import TrainImages
from .model import StudentModel
from .artifacts import file_sha256
from src.evaluation.model_loader import EvaluationEncoder
from src.evaluation.precision_contract import apply_runtime_precision

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('middle-checkpoint','middle-run-config','student-pretrained','asset-output','train-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--device',default='cuda:0');p.add_argument('--num-workers',type=int,default=4)
    args=p.parse_args(argv);output=Path(args.asset_output).resolve()
    if output.exists() or Path(str(output)+'.json').exists():raise FileExistsError(output)
    middle=json.loads(Path(args.middle_run_config).read_text())
    if middle.get('sam',{}).get('enabled') or middle['data']['input_size']!=args.image_size:raise ValueError('New-chain Middle protocol mismatch')
    torch.manual_seed(args.seed)
    student=StudentModel(ckpt_path=args.student_pretrained).to(args.device)
    apply_runtime_precision(student,'student');encoder=EvaluationEncoder(student,512).eval()
    chunks=[];paths=[]
    with torch.no_grad():
        for domain in ('drone','satellite'):
            ds=TrainImages(domain,args.train_root,args.image_size)
            if len(ds)<384:raise ValueError('TRAIN calibration needs 384 images per view')
            loader=DataLoader(torch.utils.data.Subset(ds,range(384)),batch_size=32,shuffle=False,num_workers=args.num_workers)
            chunks.extend(encoder(batch[0].to(args.device)).float().cpu() for batch in loader)
            paths.extend(str(path) for path,_ in ds.rows[:384])
    calibration=torch.cat(chunks)
    assert calibration.dtype==torch.float32 and calibration.shape==(768,512) and torch.isfinite(calibration).all()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as handle:torch.save(calibration,handle)
    metadata=dict(split='train',seed=args.seed,img_size=args.image_size,dtype='float32',paths=paths,
        middle_checkpoint_sha256=file_sha256(args.middle_checkpoint),middle_config_sha256=file_sha256(args.middle_run_config),
        student_pretrained_sha256=file_sha256(args.student_pretrained),calibration_sha256=file_sha256(output))
    with Path(str(output)+'.json').open('x') as handle:json.dump(metadata,handle,indent=2)
if __name__=='__main__':main()
