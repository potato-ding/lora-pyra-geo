import json
from pathlib import Path
from unittest.mock import patch
import pytest
from src.student.formal_config import ROOT, load_formal_config

@pytest.mark.parametrize("i",[1,2,3,4])
def test_hrd_replica_only_changes_reference_and_destination(i):
    folder=ROOT/'configs/student'
    base=json.loads((folder/'s3_sam_adsd_hrd_shared_m3_256.json').read_text())
    path=folder/f's3_sam_adsd_hrd_random{i}_shared_m3_256.json'
    _,cfg=load_formal_config(path)
    assert {k for k in cfg if cfg[k]!=base[k]}=={'output_dir','subspace_reference'}
    assert Path(cfg['subspace_reference']['checkpoint']).parent.name==f'S3-256-M3-RMLP-LINEAR-RANDOM{i}-GBW'
    cfg['subspace_reference']['checkpoint']='/wrong/RANDOM/best_model.pth'
    with patch.object(Path,'read_text',return_value=json.dumps(cfg)):
        with pytest.raises(ValueError,match='matching RANDOM'):
            load_formal_config(path)
