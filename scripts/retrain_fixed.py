"""One-shot 18-run (S0 deferred) training DAG followed by frozen-protocol evaluations.

No checkpoint resume or protocol migration. Failed commands stop the pipeline.
GPU acquisition waits for all requested physical devices to be idle.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/home/dingyi/miniforge3/envs/pyra_geo/bin/python')
RUN=ROOT/'runs/fixed_protocol_retrain_no_s0_20261005'

def tasks():
    jobs=[]
    def add(name,command,gpus,outputs):
        jobs.append(dict(name=name,command=command,gpus=gpus,outputs=[str(p) for p in outputs]))
    def train(stage,config,gpus):
        c=json.loads(config.read_text());out=Path(c['output_dir'])
        add('train_'+c['experiment_id'],['bash','scripts/train_'+stage+'.sh',str(config)],gpus,[out/'best_model.pth'])
        return c
    # S0 is deferred by explicit user instruction.
    for size in (224,256):
        train('teacher',ROOT/f'configs/teacher/t0_{size}.json',[0,1,2,3])
        for config in sorted((ROOT/'configs/middle_teacher').glob(f'*_{size}.json')):
            train('middle_teacher',config,[0,1])
        for method in (2,3):
            config=next((ROOT/'configs/middle_teacher').glob(f'm{method}_*_{size}.json'))
            c=json.loads(config.read_text());ck=Path(c['output_dir'])/'best_model.pth'
            asset=ROOT/f'src/checkpoint/student/R{size}/middle_assets/M{method}-{size}'
            add(f'top_M{method}_{size}',[str(PYTHON),'-m','src.student.build_top_only',
                '--middle-checkpoint',str(ck),'--middle-config',str(config),'--image-size',str(size),
                '--output',str(asset/'top128')],[0],[asset/'top128/manifest.json'])
            add(f'calibration_M{method}_{size}',[str(PYTHON),'-m','src.student.build_calibration',
                '--middle-checkpoint',str(ck),'--middle-run-config',str(config),'--image-size',str(size),
                '--student-pretrained',str(ROOT/'src/models/repvit/repvit_m1_5_distill_450e.pth'),
                '--train-root',str(ROOT/'data/U1652/train'),'--asset-output',str(asset/'top_calibration.pt')],
                [0],[asset/'top_calibration.pt',asset/'top_calibration.pt.json'])
        for config in sorted((ROOT/'configs/student').glob(f'*_{size}.json')):
            if not config.name.startswith('s0_'):train('student',config,[0])
    for stage,kind in [('teacher','teacher'),('middle_teacher','middle'),('student','student')]:
        for config in sorted((ROOT/'configs'/stage).glob('*.json')):
            if stage=='student' and config.name.startswith('s0_'):continue
            c=json.loads(config.read_text());out=Path(c['output_dir'])
            for dataset,name in [('u1652','test_1652.json'),('sues200','test_sues200.json'),('gta','test_gta_cross_area_d2s.json')]:
                add(f'eval_{c["experiment_id"]}_{dataset}',[str(PYTHON),'-m','src.evaluation.evaluate',
                    '--model-type',kind,'--checkpoint',str(out/'best_model.pth'),'--config',str(config),
                    '--image-size',str(c['img_size']),'--dataset',dataset],[0],[out/name])
    for model,folder in [('sample4geo','Sample4geo'),('dac','DAC'),('camp','CAMP'),('mean','MEAN')]:
        for dataset in ('u1652','sues200','gta'):
            add(f'comparison_{model}_{dataset}',[str(PYTHON),'-m','src.evaluation.comparison',
                '--model',model,'--dataset',dataset],[0],
                [ROOT/f'src/checkpoint/others/{folder}/results/fixed_protocol/{dataset}.json'])
    return jobs

def idle(gpus):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    states={int(a): (int(b),int(c)) for a,b,c in (line.split(',') for line in rows.splitlines())}
    return all(i in states and states[i][0]<400 and states[i][1]<5 for i in gpus)

def run():
    jobs=tasks();RUN.mkdir(parents=True,exist_ok=True)
    status=RUN/'status.json'
    if status.exists():raise FileExistsError('One-shot run already started')
    (RUN/'plan.json').write_text(json.dumps(jobs,indent=2))
    env=dict(os.environ,PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'],
             PYTHON_BIN=str(PYTHON),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    for index,job in enumerate(jobs):
        state=dict(index=index,total=len(jobs),job=job,pid=os.getpid())
        for output in job['outputs']:
            if Path(output).exists():raise FileExistsError(output)
        while True:
            status.write_text(json.dumps(dict(state,state='waiting_for_idle_gpus'),indent=2))
            if idle(job['gpus']):break
            time.sleep(30)
        env['CUDA_VISIBLE_DEVICES']=','.join(map(str,job['gpus']))
        log=RUN/(job['name']+'.log')
        with log.open('x') as stream:
            proc=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            status.write_text(json.dumps(dict(state,state='running',child_pid=proc.pid,log=str(log)),indent=2))
            code=proc.wait()
        missing=[p for p in job['outputs'] if not Path(p).is_file()]
        with (RUN/'completed.jsonl').open('a') as f:f.write(json.dumps(dict(state,exit_code=code,missing=missing))+'\n')
        if code or missing:
            status.write_text(json.dumps(dict(state,state='failed',exit_code=code,missing=missing,log=str(log)),indent=2))
            raise SystemExit(code or 1)
    status.write_text(json.dumps(dict(state='complete',total=len(jobs)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan-only',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.plan_only:
        jobs=tasks();print(json.dumps(jobs,indent=2))
    elif a.launch:
        RUN.mkdir(parents=True,exist_ok=True)
        if (RUN/'status.json').exists() or (RUN/'supervisor.pid').exists():raise FileExistsError(RUN)
        with (RUN/'supervisor.log').open('x') as f:
            proc=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,
                                  stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (RUN/'supervisor.pid').write_text(str(proc.pid));print(proc.pid)
    else:run()
