"""Reserve a fresh formal run and tee all torchrun output to train.log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .train import load_config
from .artifacts import ROOT, source_identity, file_sha256

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    args=p.parse_args(argv)
    cfg=load_config(args.config)
    seal=None
    if cfg.get("sealed_provenance_file"):
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
    command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",
             "-m","src.student.train","--config",str(Path(args.config).resolve())]
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
