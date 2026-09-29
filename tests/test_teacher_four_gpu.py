"""The new four-GPU Teacher run is explicit and leaves legacy configs intact."""
import json
from pathlib import Path
import pytest
from src.training.teacher.formal_config import parse_args


def test_four_gpu_teacher_configs_have_one_method():
    configs = [json.loads(Path(f'configs/teacher/t0_{size}.json').read_text())
               for size in (224,256)]
    differences = {key for key in configs[0] if configs[0][key] != configs[1][key]}
    assert differences == {'img_size','experiment_id','output_dir'}
    for size,cfg in zip((224,256),configs):
        assert cfg['batch_size'] == 8
        assert cfg['val_batch_size'] == 16
        assert (cfg['lora_start_block'],cfg['lora_end_block']) == (20,36)
        assert (cfg['full_finetune_start_block'],cfg['full_finetune_end_block']) == (36,40)
        args = parse_args(['--config',f'configs/teacher/t0_{size}.json'])
        assert (args.img_size,args.batch_size,args.val_batch_size,args.epochs) == (size,8,16,10)
        assert args.batch_size * 4 * args.grad_accum_steps == 32


def test_four_gpu_teacher_rejects_old_batch_and_wrong_selection_batch(tmp_path):
    cfg = json.loads(Path('configs/teacher/t0_224.json').read_text())
    path = tmp_path/'teacher.json'
    cfg['batch_size'] = 4
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='batch_size'):
        parse_args(['--config',str(path)])
    cfg['batch_size'] = 8
    cfg['val_batch_size'] = 8
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='val_batch_size'):
        parse_args(['--config',str(path)])


def test_four_gpu_teacher_epoch_lr_and_layer_ranges_come_from_config(tmp_path):
    cfg = json.loads(Path('configs/teacher/t0_224.json').read_text())
    cfg.update(epochs=12,lr=2e-4)
    path = tmp_path/'teacher.json'
    path.write_text(json.dumps(cfg))
    args = parse_args(['--config',str(path)])
    assert (args.epochs,args.lr,args.lora_start_block) == (12,2e-4,20)


def test_teacher_selection_starts_at_six_and_ranges_do_not_overlap():
    import ast
    from types import SimpleNamespace
    from src.models.teacher.model import resolve_teacher_tuning_ranges
    tree=ast.parse(Path('src/training/teacher/train.py').read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='should_run_validation')
    module=ast.Module(body=[fn],type_ignores=[])
    namespace={}
    exec(compile(module,'teacher_selection_schedule','exec'),namespace)
    assert [epoch for epoch in range(1,11) if namespace['should_run_validation'](epoch,SimpleNamespace(epochs=10))]==[6,7,8,9,10]
    args=SimpleNamespace(lora_start_block=20,lora_end_block=36,
        full_finetune_start_block=36,full_finetune_end_block=40)
    ranges=resolve_teacher_tuning_ranges(args,40)
    lora=set(range(*ranges['lora_range']))
    full=set(range(*ranges['full_range']))
    assert lora==set(range(20,36)) and full==set(range(36,40))
    assert not lora & full


def test_teacher_v3_saves_exact_two_artifacts_in_config_output(tmp_path,monkeypatch):
    import torch
    from src.models.teacher import model as teacher_module
    from src.training.teacher.formal_checkpoint import (SCHEMA,checkpoint_metadata,
        save_best_checkpoint,validate_training_artifacts)
    from src.training.teacher.formal_reload import load_teacher_v3

    class TinyTeacher(torch.nn.Module):
        def __init__(self,args):
            super().__init__()
            self.proj=torch.nn.Linear(4,4096,bias=False).bfloat16()
        def forward(self,x):
            return torch.nn.functional.normalize(self.proj(x.bfloat16()).float(),dim=-1)

    monkeypatch.setattr(teacher_module,'TeacherModel',TinyTeacher)
    for size in (224,256):
        run=tmp_path/f'R{size}'
        run.mkdir()
        (run/'train.log').write_text('test log\n')
        cfg=json.loads(Path(f'configs/teacher/t0_{size}.json').read_text())
        cfg['output_dir']=str(run)
        cfg_path=tmp_path/f'config-{size}.json'
        cfg_path.write_text(json.dumps(cfg))
        args=parse_args(['--config',str(cfg_path)])
        model=TinyTeacher(args)
        refs={'D2S':{'R@1':25.,'R@5':40.,'mAP':30.},
              'S2D':{'R@1':35.,'R@5':45.,'mAP':38.},'epoch':6}
        save_best_checkpoint(model,args,refs,run,4)
        validate_training_artifacts(run)
        assert {p.name for p in run.iterdir()}=={'best_model.pth','train.log'}
        payload=torch.load(run/'best_model.pth',map_location='cpu',weights_only=True)
        assert payload['artifact_schema']==SCHEMA
        assert checkpoint_metadata(payload)['eval_batch_size']==16
        assert payload['precision_signature']['selection_batch_size']==16
        assert payload['config']['output_dir']==str(run)
        encoder,audit=load_teacher_v3(run/'best_model.pth',device='cpu',image_size=size)
        assert audit['checkpoint_metadata']['best_epoch']==6
        assert set(encoder.model.state_dict())==set(model.state_dict())
        with pytest.raises(ValueError,match='destination mismatch'):
            save_best_checkpoint(model,args,refs,tmp_path,4)


def test_teacher_selection_uses_single_process_u1652_and_matching_image_size(monkeypatch):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from src.training.teacher import formal_selection as selection
    from contextlib import nullcontext
    seen=[]
    dataset=TensorDataset(torch.zeros(33,3,4,4))
    original=DataLoader(dataset,batch_size=2)
    def loaders(*,data_dir,img_size,batch_size,num_workers,distributed):
        seen.append(('build',img_size,batch_size,distributed))
        return {direction:(original,original) for direction in ('D2S','S2D')}
    def evaluate(model,query,gallery,device,task_name,precomputed_features):
        seen.append((task_name,query.batch_size,gallery.batch_size))
        return (1.,2.,3.,4.)
    monkeypatch.setattr(selection,'build_1652_val_dataloaders',loaders)
    monkeypatch.setattr(selection,'getdist_1652_val_and_get_recall',evaluate)
    monkeypatch.setattr(selection,'single_process_evaluation',nullcontext)
    monkeypatch.setattr(selection,'formal_selection_signature',lambda *args: {})
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.zeros(1))
    for size in (224,256):
        seen.clear()
        results=selection.certified_teacher_selection(Tiny(),image_size=size,device='cpu')
        assert seen==[('build',[size,size],16,False),('D2S',16,16),('S2D',16,16)]
        assert set(results)=={'D2S','S2D'}
