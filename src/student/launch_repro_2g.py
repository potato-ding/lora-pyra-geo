"""Scoped 2G variant of the existing tee/torchrun launcher; DeepSpeed unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .repro_2g import load_config,NAME
from .artifacts import ROOT,source_identity,file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    cfg=load_config(args.config)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1':raise ValueError('Only physical GPU0,1 permitted')
    seal=json.loads(Path(cfg['sealed_provenance_file']).read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not seal['REPRO_2G_READY'] or seal['SEALED_COMMIT']!=head or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):
        raise RuntimeError('Sealed source/clean status gate failed')
    if seal['source_sha256']!=source_identity():raise RuntimeError('Source SHA mismatch')
    if file_sha256(args.config)!=seal['config_sha256']:raise RuntimeError('Config SHA mismatch')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(l.split(',')[0]):int(l.split(',')[1]) for l in raw.strip().splitlines()}
    if memory[0]>=100 or memory[1]>=100:raise RuntimeError('GPU0/1 occupied: STOP')
    run=Path(cfg['output_dir']).resolve()
    if run.exists() and any(run.iterdir()):raise FileExistsError(run)
    run.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,STUDENT_RESERVED_OUTPUT=str(run),STUDENT_SEALED_COMMIT=head,PYTHONUNBUFFERED='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             '-m','src.student.repro_2g','train','--config',str(Path(args.config).resolve())]
    with (run/'train.log').open('xb') as sink:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
        try:
            for line in iter(child.stdout.readline,b''):
                sink.write(line);sink.flush();sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
            code=child.wait()
        except BaseException:
            child.terminate();child.wait();raise
    raise SystemExit(code)


if __name__=='__main__':main()
