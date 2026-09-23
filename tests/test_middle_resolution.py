import copy,json
from pathlib import Path
import pytest
import torch
from src.middle_teacher.core_config import validate_core_config,validate_teacher_identity
from src.middle_teacher.artifacts import MiddleCheckpointController,checkpoint_metadata
ROOT=Path(__file__).resolve().parents[1]

def config(method='m0-infonce',size=224):
    return json.loads((ROOT/f'configs/middle_teacher/{method}-r{size}.json').read_text())

@pytest.mark.parametrize('size',[224,384])
@pytest.mark.parametrize('method',['m0-infonce','m2-hrd-sem'])
def test_config_and_only_resolution_differences(size,method):
    c=config(method,size);validate_core_config(c)
    base=config(method)
    c['experiment']['name']=base['experiment']['name'];c['checkpoint']['output_dir']=base['checkpoint']['output_dir'];c['data']['input_size']=224
    assert c==base

@pytest.mark.parametrize('field,value',[('epochs',11),('batch',8),('weight',.2),('optimizer',.002),('selection','loss'),('gather',False),('size',448)])
def test_illegal_mutations(field,value):
    c=config('m2-hrd-sem',384)
    if field=='epochs':c['experiment']['epochs']=value
    elif field=='batch':c['data']['local_pair_batch']=value
    elif field=='weight':c['distillation']['margin']['weight']=value
    elif field=='optimizer':c['optimizer']['base_lr']=value
    elif field=='selection':c['checkpoint']['best_metric']=value
    elif field=='gather':c['data']['cross_gpu_gather']=value
    else:c['data']['input_size']=value
    with pytest.raises(ValueError):validate_core_config(c)

@pytest.mark.parametrize('size',[224,384])
def test_teacher_identity(size):
    m=dict(experiment_id=f'T0-INFONCE-R{size}',image_size=size,selection_mode='SINGLE_GPU_CANONICAL',selection_world_size=1,selection_rank=0)
    assert validate_teacher_identity(m,size)==m
    with pytest.raises(ValueError):validate_teacher_identity(m,384 if size==224 else 224)
    m['experiment_id']='OTHER'
    with pytest.raises(ValueError):validate_teacher_identity(m,size)

@pytest.mark.parametrize('size',[224,384])
def test_evaluator_uses_requested_transform(size,monkeypatch):
    from src.evaluation import middle_canonical as mod
    from torch.utils.data import DataLoader,TensorDataset
    seen={}
    def builder(**kw):
        seen.update(kw);loader=DataLoader(TensorDataset(torch.eye(4),torch.arange(4),torch.arange(4)),batch_size=2)
        return {d:(loader,loader) for d in ('D2S','S2D')}
    monkeypatch.setattr(mod,'build_1652_val_dataloaders',builder)
    x=mod.evaluate_middle_u1652_canonical(torch.nn.Identity(),image_size=size,device='cpu')
    assert seen['img_size']==[size,size] and seen['batch_size']==32 and seen['distributed'] is False
    assert x['D2S']['R@1']==100 and x['S2D']['AP']==100

@pytest.mark.parametrize('size',[224,384])
def test_artifact_and_cross_resolution_reload(size,tmp_path,monkeypatch):
    import src.middle_teacher.model as model_module
    from src.evaluation.model_loader import load_encoder
    c=config('m2-hrd-sem',size);m=torch.nn.Linear(4,768,bias=False).bfloat16()
    m.distillation_teacher_identity=dict(checkpoint=f'/teacher/R{size}/best_model.pth',sha256='fixture',checkpoint_metadata=dict(experiment_id=f'T0-INFONCE-R{size}',image_size=size,selection_mode='SINGLE_GPU_CANONICAL',selection_world_size=1,selection_rank=0))
    ctl=MiddleCheckpointController(tmp_path,c)
    metrics={d+'_'+k:v for d in ('D2S','S2D') for k,v in [('R1',80.),('R5',90.),('AP',75.)]};metrics['R1_sum']=160.
    assert ctl.save_best_if_improved(m,1,1,metrics)
    assert not ctl.save_best_if_improved(m,2,2,metrics)
    p=tmp_path/'best_model.pth';x=torch.load(p,weights_only=False);meta=checkpoint_metadata(x)
    assert meta['image_size']==x['precision_signature']['image_size']==size
    assert meta['teacher']==m.distillation_teacher_identity
    monkeypatch.setattr(model_module,'build_middle_teacher',lambda *a,**kw:torch.nn.Linear(4,768,bias=False))
    enc,audit=load_encoder('middle',p,device='cpu',image_size=size)
    assert torch.equal(enc.model.weight,m.weight)
    with pytest.raises(ValueError,match='cross-resolution'):load_encoder('middle',p,device='cpu',image_size=384 if size==224 else 224)
    bad=copy.deepcopy(x);bad['config']['data']['input_size']=384 if size==224 else 224
    with pytest.raises(ValueError):checkpoint_metadata(bad)

@pytest.mark.parametrize('patch_count',[196,576])
def test_abv_dynamic_patches_and_backward(patch_count):
    from src.middle_teacher.losses.adaptive_bridge_v2 import AdaptiveBridgeV2Bank,adaptive_bridge_v2_loss
    c=config('m2-hrd-sem')['distillation']['adaptive_bridge_v2'];c.update(teacher_dim=8,middle_dim=4,bridge_hidden_dim=16)
    bank=AdaptiveBridgeV2Bank(c);target=torch.randn(4,4,requires_grad=True)
    cls=tuple(torch.randn(4,8) for _ in range(2));patch=tuple(torch.randn(4,patch_count,8) for _ in range(2))
    loss,audit=adaptive_bridge_v2_loss(cls,patch,target,bank,c);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(target.grad).all()
    assert audit['teacher_patch_shapes']==[[4,patch_count,8],[4,patch_count,8]]

def test_p0_middle_states_match():
    from src.middle_teacher.model import build_middle_teacher
    from src.middle_teacher.abv_runtime import build_stage3_model
    c0=config(size=384);c2=config('m2-hrd-sem',384)
    if not Path(c0['initialization']['path']).exists():pytest.skip('P0 asset unavailable')
    torch.manual_seed(0);a=build_middle_teacher(c0)
    torch.manual_seed(0);b=build_stage3_model(c2)
    assert a.backbone.state_dict().keys()==b.backbone.state_dict().keys()
    assert all(torch.equal(p,b.backbone.state_dict()[n]) for n,p in a.backbone.state_dict().items())
    assert torch.equal(a.logit_scale,b.logit_scale)

def test_r224_math_and_training_metadata_unchanged():
    import subprocess,ast
    rev='d2c920b6d92df59bfaea6591e4338f03c42fa0b5'
    for name in ['src/middle_teacher/losses/hard_rank_distillation.py','src/middle_teacher/losses/adaptive_bridge_v2.py','src/middle_teacher/optimizer.py','src/middle_teacher/runtime.py','src/middle_teacher/model.py']:
        old=subprocess.check_output(['git','show',rev+':'+name],cwd=ROOT)
        assert old==(ROOT/name).read_bytes()
    s=(ROOT/'src/middle_teacher/fchain_train.py').read_text()
    assert "'img_size':config['data']['input_size']" in s
    assert "'img_size':224" not in s


def test_real_middle_block10_geometry_and_selection_transform():
    from src.middle_teacher.abv_runtime import build_stage3_model
    from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
    c=config('m2-hrd-sem',384)
    if not Path(c['initialization']['path']).exists():pytest.skip('P0 asset unavailable')
    torch.manual_seed(0);model=build_stage3_model(c).eval();shapes=[]
    handle=model.backbone.model.blocks[10].register_forward_hook(lambda m,a,out:shapes.append(tuple(out.shape)))
    with torch.no_grad():out=model(torch.zeros(1,3,384,384),return_layer_features=True)
    handle.remove()
    assert model.backbone.model.n_storage_tokens==4
    assert shapes==[(1,581,768)]
    assert out['middle_features'][0].shape==(1,768)
    assert out['final_descriptor'].shape==(1,768)
    loaders=build_1652_val_dataloaders(data_dir='data/U1652',img_size=[384,384],batch_size=32,num_workers=0,distributed=False)
    for pair in loaders.values():
        for loader in pair:assert tuple(loader.dataset[0][0].shape)==(3,384,384)


def _two_rank_shape_loss_worker(rank,init):
    import torch.distributed as dist
    from src.utils.gather_features_and_labels_and_views import GatherLayer
    from src.middle_teacher.fchain_train import r0_pair_loss
    from src.middle_teacher.losses.hard_rank_distillation import hard_rank_losses
    from src.middle_teacher.losses.adaptive_bridge_v2 import AdaptiveBridgeV2Bank,adaptive_bridge_v2_loss
    from src.middle_teacher.composer import DistillationComposer
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2)
    try:
        for method in ['m0-infonce','m2-hrd-sem']:
            torch.manual_seed(rank)
            local=torch.randn(32,8,requires_grad=True)
            desc=torch.nn.functional.normalize(local,dim=1)
            md=torch.cat(GatherLayer.apply(desc[:16]),0);ms=torch.cat(GatherLayer.apply(desc[16:]),0)
            base,*_=r0_pair_loss(md,ms,torch.tensor(1.))
            loss=base
            if method=='m2-hrd-sem':
                c=config(method,384)['distillation'];c['adaptive_bridge_v2'].update(teacher_dim=8,middle_dim=8,bridge_hidden_dim=16)
                raw=hard_rank_losses(md,ms,md.detach(),ms.detach(),torch.arange(32),c)
                bank=AdaptiveBridgeV2Bank(c['adaptive_bridge_v2'])
                raw['adaptive_bridge_v2']=adaptive_bridge_v2_loss(tuple(torch.randn(32,8) for _ in range(2)),tuple(torch.randn(32,576,8) for _ in range(2)),local,bank,c['adaptive_bridge_v2'])
                out=DistillationComposer(c).compose(base,{k:(lambda _,v=v:v) for k,v in raw.items()})
                loss=out['total_loss']
                torch.testing.assert_close(loss,base+.1*raw['margin'][0]+.05*raw['adaptive_bridge_v2'][0])
            loss.backward()
            assert md.shape==(32,8) and torch.isfinite(loss) and torch.isfinite(local.grad).all()
    finally:dist.destroy_process_group()


def test_two_rank_synthetic_m0_m2(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_two_rank_shape_loss_worker,args=('file://'+str(tmp_path/'init'),),nprocs=2,join=True)
