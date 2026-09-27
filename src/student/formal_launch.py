"""Committed-source gate and direct one-process launch for formal R224 Students."""
import os
from pathlib import Path
import socket
import subprocess
import sys
from .artifacts import ROOT
from .core_config import validate_config


def launch(cfg, config_path):
    validate_config(cfg,check_assets=True)
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='dev':
        raise RuntimeError('Formal Student requires dev')
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if dirty:raise RuntimeError('Commit source/config/test changes before formal Student launch')
    config=Path(config_path).resolve()
    subprocess.run(['git','ls-files','--error-unmatch',str(config.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible!=str(cfg['assigned_gpu']):raise ValueError('Formal GPU assignment mismatch')
    memory=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(memory.strip())>=100:raise RuntimeError('Assigned GPU busy; no preemption')
    run=Path(cfg['output_dir']).resolve()
    if run.exists() and any(run.iterdir()):raise FileExistsError(run)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    env=dict(os.environ,STUDENT_RESERVED_OUTPUT=str(run),STUDENT_SEALED_COMMIT=commit,PYTHONUNBUFFERED='1')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    module='src.student.train_allocation'
    run.mkdir(parents=True,exist_ok=True)
    with (run/'train.log').open('xb') as sink:
        child=subprocess.Popen([sys.executable,'-m',module,'--config',str(config)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            for line in iter(child.stdout.readline,b''):
                sink.write(line);sink.flush();sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
            code=child.wait()
        except BaseException:
            child.terminate();child.wait();raise
    raise SystemExit(code)
