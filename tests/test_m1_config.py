import ast,copy,json,argparse
from pathlib import Path
from src.middle_teacher.config import load_config
from src.middle_teacher.core_config import validate_core_config
from src.middle_teacher.runtime import formal_deepspeed_config

CONFIG='configs/middle_teacher/m1-hrd-r224.json'
TEACHER='/home/dingyi/lora-pyra-geo/src/checkpoint/teacher/R224/T0-INFONCE-R224/best_model.pth'

def test_m1_exact_canonical_delta():
    original=load_config('configs/middle_teacher/core_v2/hrd.json')
    c=load_config(CONFIG);validate_core_config(c)
    expected=copy.deepcopy(original)
    expected['experiment']['name']='M1-HRD-R224'
    expected['checkpoint']['output_dir']='/home/dingyi/lora-pyra-geo/src/checkpoint/middle_teacher/R224/M1-HRD-R224'
    expected['checkpoint']['save_last']=False
    assert c==expected
    assert set(c['distillation'])=={'base_loss','margin'}
    assert c['distillation']['base_loss']=='pair_infonce'
    assert c['distillation']['margin']==dict(enabled=True,weight=.1,operator='ABS_MARGIN',negative_selection='teacher_top5_wrong_identity')
    assert not c['sam']['enabled'] and not c['sam']['adaptive']
    assert c['data']['input_size']==224 and c['experiment']['epochs']==10
    assert c['data']['world_size']==2 and c['data']['local_pair_batch']==16 and c['data']['global_pair_batch']==32
    assert c['trainability']['full_finetune_blocks']==list(range(12)) and not c['trainability']['lora_blocks']
    assert formal_deepspeed_config(c)['gradient_accumulation_steps']==1

def test_real_fchain_parser_accepts_m1_and_formal_teacher():
    # Execute only the actual parser construction, before distributed/CUDA setup.
    tree=ast.parse(Path('src/middle_teacher/fchain_train.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[]
    for node in main.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
        nodes.append(node)
    ns={'argparse':argparse}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'fchain_parser','exec'),ns)
    args=ns['parser'].parse_args(['--config',CONFIG,'--expected-gpus','0,1','--teacher-checkpoint',TEACHER])
    assert args.teacher_checkpoint==TEACHER and args.config==CONFIG
    validate_core_config(load_config(args.config))
