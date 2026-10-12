"""Sequential independent-process efficiency suite; never shares a model GPU."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/dingyi/lora-pyra-geo')


def entries():
    result = []
    for kind, tag, parent in [('middle', 'M3', 'middle_teacher'), ('student', 'S3', 'student')]:
        for size in (224, 256):
            result.append({'label': f'{kind.title()}-{tag}-{size}', 'model': kind,
                'checkpoint': str(ROOT/f'src/checkpoint/{parent}/R{size}/{tag}-{size}/best_model.pth'),
                'image_size': size})
    for kind, label in [('mean','MEAN'), ('dac','DAC'), ('camp','CAMP'),
                        ('sample4geo','Sample4Geo'), ('lpn','LPN'), ('muse','MuSe-Net'),
                        ('mccg','MCCG'), ('infogeo_rd_trial','InfoGeo-RD-trial'),
                        ('ccr','CCR'), ('spikevimformer','SpikeViMFormer-S'), ('dmnil','DMNIL')]:
        result.append({'label': label, 'model': kind})
    return result


def write(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--own-only', action='store_true')
    p.add_argument('--models', nargs='+', choices=[job['model'] for job in entries()],
                   help='Run only these models, in the supplied order')
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out/'status.json').exists():
        raise FileExistsError(out/'status.json')
    if a.own_only and a.models:
        p.error('--own-only and --models cannot be combined')
    if a.models:
        if len(set(a.models)) != len(a.models):
            p.error('--models must not contain duplicates')
        jobs = [job for kind in a.models for job in entries() if job['model'] == kind]
    else:
        jobs = entries()[:4] if a.own_only else entries()
    state = {'started_utc': datetime.now(timezone.utc).isoformat(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
             'smoke_only': a.smoke, 'entries': jobs, 'completed': [], 'failed': [], 'status': 'running'}
    for job in jobs:
        state['current'] = job['label']
        write(out/'status.json', state)
        cmd = [sys.executable, '-m', 'src.evaluation.efficiency', '--model', job['model'],
               '--label', job['label'], '--output', str(out/(job['label']+'.json'))]
        if 'checkpoint' in job:
            cmd += ['--checkpoint', job['checkpoint'], '--image-size', str(job['image_size'])]
        if a.smoke:
            cmd += ['--warmup', '3', '--iterations', '5', '--repeats', '1']
        with (out/(job['label']+'.log')).open('x') as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code:
            state['failed'].append({'label': job['label'], 'exit_code': code})
        else:
            state['completed'].append(job['label'])
            data = json.loads((out/(job['label']+'.json')).read_text())
            if data['gflops'] is None:
                state['failed'].append({'label': job['label'], 'reason': 'incomplete FLOPs coverage'})
        rows = []
        for label in state['completed']:
            data = json.loads((out/(label+'.json')).read_text())
            keys = ['label', 'resolution', 'params_m', 'gflops', 'latency_b1_ms', 'fps_b16', 'peak_allocated_b1_mib']
            row = {k: data[k] for k in keys}
            row['flop_status'] = data['flop_audit']['status']
            rows.append(row)
        if rows:
            with (out/'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write(out/'status.json', state)
        if state['failed'] and not a.smoke:
            break
    state.update(current=None, status='complete' if not state['failed'] else 'complete_with_failures',
                 completed_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
