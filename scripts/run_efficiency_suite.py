"""Sequential independent-process efficiency suite; never shares a model GPU."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def entries(include_optional=False):
    result = []
    for kind, tag, parent in [('middle', 'M3', 'middle_teacher'), ('student', 'S3', 'student')]:
        for size in (224, 256):
            result.append({'label': f'{kind.title()}-{tag}-{size}', 'model': kind,
                'checkpoint': str(ROOT/f'src/checkpoint/{parent}/R{size}/{tag}-{size}{"-M3" if kind == "student" else ""}/best_model.pth'),
                'image_size': size})
    for kind, label in [('mean','MEAN'), ('dac','DAC'), ('camp','CAMP'),
                        ('sample4geo','Sample4Geo'), ('lpn','LPN'), ('muse','MuSe-Net'),
                        ('mccg','MCCG'), ('infogeo_rd_trial','InfoGeo-RD-trial'),
                        ('ccr','CCR'), ('spikevimformer','SpikeViMFormer-S')]:
        result.append({'label': label, 'model': kind})
    if include_optional:
        result.append({"label": "DMNIL", "model": "dmnil"})
    return result



def select_entries(own_only=False, models=None):
    if own_only and models:
        raise ValueError("--own-only and --models cannot be combined")
    available = entries(include_optional=True)
    if models:
        if len(set(models)) != len(models):
            raise ValueError("--models must not contain duplicates")
        unknown = set(models) - {job["model"] for job in available}
        if unknown:
            raise ValueError("Unknown efficiency models: " + ", ".join(sorted(unknown)))
        return [job for kind in models for job in available if job["model"] == kind]
    return entries()[:4] if own_only else entries()


def comparison_input_paths(kind, root=None):
    """Mirror the existing adapters' flat and checkpoint-bundle layouts."""
    from src.evaluation.comparison_models import SPECS
    root = ROOT if root is None else Path(root)
    folder, weight_name, _ = SPECS[kind]
    base = root / "src/checkpoint/others" / folder
    extra = []
    if kind == "muse":
        bundles = sorted((base / "weights").glob("three_view*"))
        if len(bundles) != 1:
            raise FileNotFoundError("Expected exactly one uploaded MuSe-Net bundle: " + str(base))
        weight = bundles[0] / weight_name
        extra = [(bundles[0] / "opts.yaml", "file")]
    elif kind == "ccr":
        weight = base / "weights/extracted/model/CCR_Model_University" / weight_name
        extra = [(base / "source/compatibility_adapter.py", "file")]
    else:
        weight = base / "weights" / weight_name
    return [(weight, "file"), (base / "source", "directory"), *extra]


def validate_jobs(jobs, output_dir):
    """Reject missing inputs and occupied outputs before starting any measurement."""
    if len({job["label"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate efficiency job labels")
    missing = []
    for job in jobs:
        if "checkpoint" in job:
            paths = [(Path(job["checkpoint"]), "file")]
        else:
            paths = comparison_input_paths(job["model"])
        missing.extend(str(path) for path, kind in paths
                       if not (path.is_file() if kind == "file" else path.is_dir()))
    if missing:
        raise FileNotFoundError("Efficiency inputs missing: " + "; ".join(missing))
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Efficiency output must be a new or empty directory: " + str(output_dir))


def write(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--own-only', action='store_true')
    p.add_argument('--models', nargs='+', choices=sorted({job['model'] for job in entries(include_optional=True)}),
                   help='Run only these models, in the supplied order')
    p.add_argument('--check', action='store_true', help='Validate inputs and output paths without loading models or using a GPU')
    a = p.parse_args(argv)
    try:
        jobs = select_entries(a.own_only, a.models)
    except ValueError as error:
        p.error(str(error))
    out = a.output_dir
    validate_jobs(jobs, out)
    if a.check:
        print(json.dumps({'status': 'preflight_pass', 'entries': jobs, 'output_dir': str(out)}, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
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
    return 1 if state["failed"] else 0


if __name__ == '__main__':
    raise SystemExit(main())
