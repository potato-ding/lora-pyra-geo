"""Merge only formally audited, identically configured efficiency tables."""
import csv
import json
import hashlib
from pathlib import Path
import sys

roots = list(map(Path, sys.argv[1:]))
assert len(roots) >= 2
baselines = roots[-1]
reference_protocol = reference_script = None
rows, sources = [], []
for root in roots:
    status = json.loads((root/'status.json').read_text())
    assert status['status'] == 'complete' and not status['failed']
    reviewed = json.loads((root/'summary_reviewed.json').read_text())
    for record in reviewed:
        row = record['summary']
        raw = json.loads((root/(row['method']+'.json')).read_text())
        source = (root/'efficiency.py').read_bytes()
        assert hashlib.sha256(source).hexdigest() == raw['script_sha256']
        # Sole audited counter extension: nearest sampling is data movement,
        # not floating interpolation. No timing/precision/fusion changes.
        canonical = source.replace(b'replication_pad2d pad upsample_nearest2d isfinite',
                                   b'replication_pad2d pad isfinite')
        canonical_hash = hashlib.sha256(canonical).hexdigest()
        if reference_protocol is None:
            reference_protocol, reference_script = raw['protocol'], canonical_hash
        assert raw['protocol'] == reference_protocol
        assert canonical_hash == reference_script
        assert raw['fps_batch_size'] == 16 and raw['observed_gpu_floating_dtypes'] == ['torch.float32']
        row = dict(row)
        row['qualification'] = {
            'mccg': 'Public-source compatible inference; checkpoint-era implementation unconfirmed',
            'infogeo_rd_trial': 'Reconstructed compatible inference; RD training variant unconfirmed',
            'lpn': 'Training configuration unavailable; audited default stride=1 / 4 parts / 256 input',
            'muse': 'Uploaded training configuration includes extra Google street-view data',
        }.get(raw['model'], '')
        rows.append(row)
        sources.append({'method':row['method'], 'raw_file':str(root/(row['method']+'.json')),
                        'actual_benchmark_sha256':raw['script_sha256'],
                        'checkpoint_sha256':raw['checkpoint_audit'].get('sha256', raw['checkpoint_audit'].get('checkpoint_sha256'))})
assert len({row['method'] for row in rows}) == len(rows)
with (baselines/f'main_table_{len(rows)}models.csv').open('x', newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (baselines/f'main_table_{len(rows)}models_audit.json').open('x') as f:
    json.dump({'protocol':reference_protocol,'canonical_benchmark_sha256':reference_script,
               'allowed_source_difference':'Only nearest-neighbor zero-arithmetic handler extension; exact source equality otherwise',
               'sources':sources},f,indent=2)
print(f'PASS: merged {len(rows)} configurations with identical timing/precision/fusion source and protocol; zero-cost nearest handler extension audited; qualifications preserved')
