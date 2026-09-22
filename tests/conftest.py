"""Explicit availability guard for external historical certification artifacts."""
import json
from pathlib import Path
import pytest

@pytest.fixture
def historical_p2_calibration():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/student/certified_r224/p2_top_rmlp_s0.json').read_text())
    path=Path(cfg['p2_calibration_path'])
    if not path.is_file():
        pytest.skip('Historical CERTIFIED_R224 calibration unavailable: '+str(path))
    # Existing production validation still checks its exact SHA; a present but
    # incorrect historical asset is a failure, never a reason to skip.
    return path
