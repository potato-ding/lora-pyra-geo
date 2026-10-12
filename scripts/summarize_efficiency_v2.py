"""Validate the deployment protocol and summarize all recorded formal rounds."""
import csv
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
status = json.loads((root/'status.json').read_text())
assert status['status'] == 'complete' and not status['failed'], status
rows, records = [], []
protocol = script_hash = None
for label in status['completed']:
    path = root/(label+'.json')
    value = json.loads(path.read_text())
    p = value['protocol']
    assert p['version'] == 'retrieval-efficiency-v2-deployment-wall'
    assert (p['warmup'], p['iterations_per_round'], p['rounds']) == (100, 500, 3)
    assert value['fps_batch_size'] == 16
    assert value['observed_gpu_floating_dtypes'] == ['torch.float32']
    assert value['gflops'] is not None and value['flop_audit']['status'] == 'operator_coverage_complete'
    assert not value['checkpoint_audit'].get('missing', []) and not value['checkpoint_audit'].get('unexpected', [])
    if protocol is None:
        protocol, script_hash = p, value['script_sha256']
    assert p == protocol and value['script_sha256'] == script_hash
    for b in (1, 2, 16):
        assert 'deployment_'+str(b) in value['parity_checks']
        assert 'cpu_offload_'+str(b) in value['parity_checks']
    if value['model'] == 'student':
        assert value['deployment_audit']['batchnorm_modules_after'] == 1
        assert value['deployment_audit']['remaining_batchnorm_names'] == ['model.neck']
    b1, b16 = value['b1_measurements']['rounds'], value['throughput_measurements']['rounds']
    means = [r['mean_ms'] for r in b1]
    fps = [r['synchronized_wall_images_per_second'] for r in b16]
    timings = [t for r in b1 for t in r['iteration_ms']]
    assert len(timings) == 1500
    row = {'method':label, 'resolution':value['resolution'], 'params_m':value['params_m'],
           'gflops':value['gflops'], 'latency_b1_mean_ms':statistics.mean(means),
           'latency_round_std_ms':statistics.stdev(means),
           'latency_b1_median_ms':statistics.median(timings),
           'fps_b16':statistics.mean(fps), 'fps_round_std':statistics.stdev(fps),
           'peak_allocated_b1_mib':max(r['peak_allocated_mib'] for r in b1),
           'max_deployment_descriptor_difference':max(v['max_abs_difference'] for k,v in value['parity_checks'].items() if k.startswith('deployment_')),
           'latency_cv_percent':100*statistics.stdev(means)/statistics.mean(means),
           'fps_cv_percent':100*statistics.stdev(fps)/statistics.mean(fps),
           'rounds':len(b1)}
    rows.append(row)
    records.append({'summary':row, 'raw_result_file':str(path),
                    'checkpoint_sha256':value['checkpoint_audit'].get('sha256', value['checkpoint_audit'].get('checkpoint_sha256')),
                    'script_sha256':script_hash, 'all_rounds_included':True})
with (root/'summary_reviewed.csv').open('x',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
with (root/'summary_reviewed.json').open('x') as f:
    json.dump(records,f,indent=2); f.write('\n')
for row in rows:
    print(json.dumps(row))
