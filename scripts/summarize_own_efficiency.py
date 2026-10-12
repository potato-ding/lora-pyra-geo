"""Pool all declared rounds; preserve original measurements without selection."""
import csv
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
labels = ['Middle-M3-224', 'Middle-M3-256', 'Student-S3-224', 'Student-S3-256']
rows = []
records = []
for label in labels:
    paths = [root/(label+'.json')]
    if label == 'Middle-M3-256':
        paths.append(root/'Middle-M3-256_recheck.json')
    values = [json.loads(p.read_text()) for p in paths]
    first = values[0]
    for value in values:
        assert value['gflops'] is not None
        assert value['observed_gpu_floating_dtypes'] == ['torch.float32']
        assert max(v['max_abs_difference'] for v in value['parity_checks'].values()) == 0
        for key in ('model', 'resolution', 'params_m', 'gflops', 'script_sha256', 'protocol'):
            assert value[key] == first[key], key
        assert value['checkpoint_audit']['sha256'] == first['checkpoint_audit']['sha256']
    b1 = [r for value in values for r in value['b1_measurements']['rounds']]
    b16 = [r for value in values for r in value['throughput_measurements']['rounds']]
    means = [r['mean_ms'] for r in b1]
    timings = [t for r in b1 for t in r['iteration_ms']]
    fps = [r['synchronized_wall_images_per_second'] for r in b16]
    row = {'method': label, 'resolution': first['resolution'], 'params_m': first['params_m'],
           'gflops': first['gflops'], 'latency_b1_mean_ms': statistics.mean(means),
           'latency_round_std_ms': statistics.stdev(means),
           'latency_b1_median_ms': statistics.median(timings),
           'fps_b16': statistics.mean(fps), 'fps_round_std': statistics.stdev(fps),
           'peak_allocated_b1_mib': max(r['peak_allocated_mib'] for r in b1),
           'rounds': len(b1)}
    rows.append(row)
    records.append({'summary': row, 'raw_result_files': [str(p) for p in paths],
                    'checkpoint_sha256': first['checkpoint_audit']['sha256'],
                    'script_sha256': first['script_sha256'],
                    'all_rounds_included': True})
with (root/'summary_reviewed.csv').open('x', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root/'summary_reviewed.json').write_text(json.dumps(records, indent=2)+'\n')
for row in rows:
    print(json.dumps(row))
