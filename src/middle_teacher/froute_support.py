"""F-route validation only; historical component implementations remain untouched."""
import copy,json
from pathlib import Path
from src.middle_teacher.checkpoint import sha256
BASE=Path('src/checkpoint/middle_teacher/CERTIFIED_R224')
REFERENCES={'margin':'R1-P-MARGIN-S0','nrkd':'R1-P-NRKD-S0','adaptive_bridge_v2':'R2-P-ABV2-S0','retrieval_distribution_kd':'R2-P-RDD-L020-S0'}
def component_name(c):
    names=[k for k in c['distillation'] if k!='base_loss']
    assert len(names)==1 and names[0] in REFERENCES
    return names[0]
def validate_froute(c,validate_r0):
    ref=json.loads((BASE/'R0-F-S0/run_config.json').read_text())
    base=copy.deepcopy(c);base['distillation']=ref['distillation'];base['experiment']['name']=ref['experiment']['name'];base['checkpoint']['output_dir']=ref['checkpoint']['output_dir']
    assert base==ref,'unexpected R0-F base difference'
    validate_r0(base)
    comp=component_name(c)
    old=json.loads((BASE/REFERENCES[comp]/'run_config.json').read_text())
    assert c['distillation']==old['distillation'],'component identity drift'
def runtime_for(comp):
    if comp=='adaptive_bridge_v2':
        from src.middle_teacher.abv_runtime import ABVRuntime
        return ABVRuntime
    if comp=='retrieval_distribution_kd':
        from src.middle_teacher.rdd_runtime import RDDRuntime
        return RDDRuntime
    from src.middle_teacher.historical_kd_runtime import HistoricalKDRuntime
    return HistoricalKDRuntime
def fingerprints(config_path):
    names=['src/middle_teacher/froute_train.py','src/middle_teacher/froute_support.py','src/middle_teacher/r0_train.py','src/middle_teacher/abv_runtime.py','src/middle_teacher/rdd_runtime.py','src/middle_teacher/historical_kd_runtime.py','src/middle_teacher/losses/historical_retrieval_kd.py','src/middle_teacher/losses/adaptive_bridge_v2.py','src/middle_teacher/losses/retrieval_distribution_kd.py','src/middle_teacher/composer.py','src/evaluation/model_loader.py','src/middle_teacher/model.py','src/middle_teacher/optimizer.py','src/middle_teacher/runtime.py',config_path]
    return {n:sha256(n) for n in names}
