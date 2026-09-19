import copy
import pytest
import torch
from torch import nn
from src.evaluation.precision_contract import *

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Linear(4,4);self.neck=nn.BatchNorm1d(4)
        self.lora_A=nn.Linear(4,2,bias=False);self.lora_B=nn.Linear(2,4,bias=False)
    def forward(self,x):return descriptor_postprocess(self.neck(self.backbone(x)))

@pytest.mark.parametrize('kind',['teacher','middle','student'])
def test_precision_signature(kind):
    model=Tiny().bfloat16().eval();sig=selection_signature(model,kind)
    other=Tiny();other.load_state_dict(model.state_dict(),strict=True)
    apply_runtime_precision(other,kind,sig)
    assert_precision_signature(inspect_precision_signature(other,kind),sig)
    assert sig['bn_buffer_dtypes']['neck.running_mean']=='bfloat16'
    assert sig['bn_buffer_dtypes']['neck.num_batches_tracked']=='int64'
    with pytest.raises(RuntimeError):assert_precision_signature(inspect_precision_signature(other.float(),kind),sig)

def test_checkpoint_reload_precision_signature(tmp_path):
    model=Tiny().bfloat16().eval();sig=selection_signature(model,'student')
    file=tmp_path/'best.pth';torch.save(dict(model={n:t.float() if t.is_floating_point() else t for n,t in model.state_dict().items()},precision_signature=sig),file)
    state=torch.load(file,weights_only=True);copy=Tiny();copy.load_state_dict(state['model'],strict=True)
    apply_runtime_precision(copy,'student',state['precision_signature']);copy.eval()
    x=torch.randn(8,4).bfloat16();assert torch.equal(copy(x),model(x))

def test_bn_buffers_strict_reload():
    model=Tiny().bfloat16();state=model.state_dict();state.pop('neck.running_mean')
    with pytest.raises(RuntimeError):Tiny().load_state_dict(state,strict=True)

def test_best_reload_metric_mismatch_is_fatal():
    a={d:{k:1.0 for k in ('R@1','R@5','AP')} for d in ('D2S','S2D')}
    b=copy.deepcopy(a);b['S2D']['AP']=.99
    with pytest.raises(RuntimeError,match='PRECISION_OR_RELOAD'):assert_best_reload_metrics(a,b)


def test_pca_assets_never_quantized():
    from src.student.dual_stst import DualSTSTSupervision
    from src.student.part1 import PartISupervision
    # Non-BF16-representable real-valued fixtures catch round-trip corruption.
    module=PartISupervision.__new__(PartISupervision);nn.Module.__init__(module)
    module.projector_top=nn.Linear(512,128)
    for name,shape in [('teacher_mean',(768,)),('top32_basis',(768,128)),('random32_basis',(768,32)),('random_b_basis',(768,32))]:
        module.register_buffer(name,torch.randn(shape),persistent=False)
    original={n:t.clone() for n,t in module.named_buffers()}
    for dtype in [torch.bfloat16,torch.float16,torch.float32,torch.bfloat16]:
        module.to('cpu').to(dtype=dtype)
        for name,tensor in module.named_buffers():
            assert tensor.dtype==torch.float32
            assert torch.equal(tensor,original[name])
    assert module.projector_top.weight.dtype==torch.bfloat16


def test_pca_assets_formal_setup_and_calibration():
    import json
    from pathlib import Path
    from src.student.part1 import PartISupervision
    from src.student.artifacts import file_sha256
    from src.student.allocation_gbw import prepare_top
    from src.student.part2_integration import prepare_precision_groups
    from src.student.train import StudentTrainingModel
    from src.student.optimizer import build_student_optimizer
    cfg=json.loads(Path('configs/student/certified_r224/p2_5_fixed_s0.json').read_text())
    if not Path(cfg['stst_asset']).is_file():pytest.skip('Local fixture assets absent')
    sup=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],file_sha256(cfg['middle_checkpoint']),128,'single32')
    before={n:t.clone() for n,t in sup.named_buffers()}
    calibration=torch.load(cfg['p2_calibration_path'],weights_only=True,map_location='cpu')
    exact=calibration.clone();assert calibration.dtype==torch.float32
    sup.to('cpu').bfloat16();prepare_top(sup,cfg)
    model=StudentTrainingModel(nn.Linear(2,512),sup);optimizer=build_student_optimizer(model)
    prepare_precision_groups(model,optimizer,cfg)
    for name,tensor in sup.named_buffers():
        assert tensor.dtype==torch.float32 and torch.equal(tensor.cpu(),before[name])
    assert torch.equal(calibration,exact)


@pytest.mark.parametrize('kind',['middle','student'])
def test_real_model_checkpoint_reload(kind,tmp_path):
    import json
    from pathlib import Path
    from src.evaluation.model_loader import load_encoder,EvaluationEncoder
    if kind=='student':
        from src.student.model import StudentModel
        model=StudentModel(ckpt_path=None);config=None;dim=512
    else:
        from src.middle_teacher.model import build_middle_teacher
        config='configs/middle_teacher/core_v2/baseline.json'
        model=build_middle_teacher(json.loads(Path(config).read_text()),load_foundation=True);dim=768
    model.bfloat16().eval();sig=selection_signature(model,kind)
    path=tmp_path/'best.pth'
    torch.save(dict(model=model.state_dict(),precision_signature=sig),path)
    encoder,audit=load_encoder(kind,path,config,device='cpu')
    assert audit['precision_signature']==sig and not audit['missing'] and not audit['unexpected']
    with torch.no_grad():
        for size in (224,384,448):
            x=torch.randn(1,3,size,size)
            before=EvaluationEncoder(model,dim)(x);after=encoder(x)
            assert before.shape==(1,dim) and before.dtype==torch.float32
            assert torch.equal(before,after)


def test_teacher_real_lora_precision_signature():
    from src.models.teacher.peft_lora import LoRALayer
    layer=LoRALayer(nn.Linear(16,16),r=4,alpha=8,dropout=0).bfloat16().eval()
    sig=selection_signature(layer,'teacher')
    other=LoRALayer(nn.Linear(16,16),r=4,alpha=8,dropout=0)
    other.load_state_dict(layer.state_dict(),strict=True);apply_runtime_precision(other,'teacher',sig)
    other.eval();x=torch.randn(8,16).bfloat16()
    assert sig['lora_dtypes'] and torch.equal(layer(x),other(x))


def test_teacher_nested_map_metric_schema():
    metrics={d:{'R@1':1.,'R@5':2.,'mAP':3.} for d in ('D2S','S2D')}
    converted=flat_selection_metrics(metrics)
    assert converted=={d:{'R@1':1.,'R@5':2.,'AP':3.} for d in ('D2S','S2D')}
