"""Versioned explicit manifests for new runs; historical seals are immutable."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSION='CORE_SOURCE_CONTRACT_V2'

def source_identity(stage='m2s',config_path=None,gbw=False,sam=False):
    manifest_path=ROOT/'configs/source_contract_v2.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['version']!=VERSION:raise ValueError('source contract version')
    groups=['precision',stage,'paper_ablation']
    if gbw:groups.append('gbw_family')
    if sam:groups.append('sam_extension')
    paths=set(p for g in groups for p in manifest[g]);paths.add('configs/source_contract_v2.json')
    paths.add('configs/middle_teacher/fchain_margin_abv2_s0.json')
    result={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)}
    if config_path:result[str(config_path)]=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    return result
