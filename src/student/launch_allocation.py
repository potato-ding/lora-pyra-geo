"""Reserve a fresh formal run and tee direct single-process output to train.log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
from .allocation_gbw import load_config,VARIANTS
from .artifacts import ROOT, source_identity, file_sha256

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args(argv)
    cfg=load_config(args.config)
    if args.validate_only:
        print(json.dumps(cfg,indent=2));return
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or len(visible.split(",")) != 1 or not visible.isdigit():
        raise ValueError("STU-1G-B32-R224-v1 requires exactly one visible GPU")
    if cfg.get('source_contract')!='CORE_SOURCE_CONTRACT_V2' and int(visible)!=VARIANTS[cfg['allocation_variant']][3]:raise ValueError('Fixed authorized GPU mapping required')
    memory=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(memory.strip())>=100:raise RuntimeError('Assigned GPU busy; no preemption')
    seal=None
    if cfg.get("sealed_provenance_file") and cfg.get("source_contract")!="CORE_SOURCE_CONTRACT_V2":
        seal=json.loads(Path(cfg["sealed_provenance_file"]).read_text())
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        dirty=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True)
        if not seal["B0_FINAL_SEAL"] or seal["SEALED_COMMIT"]!=head or dirty:
            raise RuntimeError("Sealed commit/clean-worktree gate failed")
        if seal["source_sha256"]!=source_identity():
            raise RuntimeError("Sealed source SHA mismatch")
        if file_sha256(args.config)!=seal["config_sha256"][str(Path(args.config).resolve())]:
            raise RuntimeError("Sealed config SHA mismatch")
    run=Path(cfg["output_dir"]).resolve()
    if run.exists() and any(run.iterdir()):raise FileExistsError(run)
    run.mkdir(parents=True,exist_ok=True)
    log=run/"train.log"
    env=dict(os.environ,STUDENT_RESERVED_OUTPUT=str(run),PYTHONUNBUFFERED="1")
    if seal:env["STUDENT_SEALED_COMMIT"]=seal["SEALED_COMMIT"]
    # Keep canonical DeepSpeed world-size-one behavior without a distributed launcher.
    with socket.socket() as socket_for_port:
        socket_for_port.bind(('127.0.0.1',0));port=socket_for_port.getsockname()[1]
    env.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',
               MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    command=[sys.executable,'-m','src.student.train_allocation','--config',str(Path(args.config).resolve())]
    with log.open("xb") as sink:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
        try:
            for line in iter(child.stdout.readline,b""):
                sink.write(line);sink.flush()
                sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
            code=child.wait()
        except BaseException:
            child.terminate();child.wait();raise
    raise SystemExit(code)

if __name__=="__main__":main()
