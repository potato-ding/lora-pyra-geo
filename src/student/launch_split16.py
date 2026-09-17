"""Launch only the sealed GPU2 Split16 causal control; never overwrite a run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .split16 import load_config,PREFLIGHT
from .artifacts import ROOT,source_identity,file_sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=load_config(args.config)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    seal=json.loads(Path(cfg['sealed_provenance_file']).read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not seal['SPLIT16_READY'] or seal['SEALED_COMMIT']!=head:raise RuntimeError('Seal mismatch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise RuntimeError('Unclean source')
    if seal['source_sha256']!=source_identity() or seal['config_sha256']!=file_sha256(args.config):raise RuntimeError('Source/config SHA mismatch')
    memory=int(subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if memory>=100:raise RuntimeError('GPU2 occupied: STOP')
    run=Path(cfg['output_dir']).resolve()
    if run.exists() and any(run.iterdir()):raise FileExistsError(run)
    run.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k not in {'RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','LOCAL_WORLD_SIZE'} and not k.startswith('TORCHELASTIC_')}
    env.update(STUDENT_RESERVED_OUTPUT=str(run),STUDENT_SEALED_COMMIT=head,PYTHONUNBUFFERED='1')
    command=[sys.executable,'-u','-m','src.student.split16','train','--config',str(Path(args.config).resolve())]
    with (run/'train.log').open('xb') as sink:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
        print('SPLIT16_TRAIN_PID='+str(child.pid),flush=True)
        for line in iter(child.stdout.readline,b''):
            sink.write(line);sink.flush();sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
        code=child.wait()
    raise SystemExit(code)


if __name__=='__main__':main()
