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
    from src.student.formal_supervision import SubspaceSupervision
    # Non-BF16-representable real-valued fixtures catch round-trip corruption.
    module=SubspaceSupervision.__new__(SubspaceSupervision);nn.Module.__init__(module)
    module.projector_top=nn.Linear(512,128)
    for name,shape in [('teacher_mean',(768,)),('top128_basis',(768,128)),('random32_basis',(768,32))]:
        module.register_buffer(name,torch.randn(shape),persistent=False)
    original={n:t.clone() for n,t in module.named_buffers()}
    for dtype in [torch.bfloat16,torch.float16,torch.float32,torch.bfloat16]:
        module.to('cpu').to(dtype=dtype)
        for name,tensor in module.named_buffers():
            assert tensor.dtype==torch.float32
            assert torch.equal(tensor,original[name])
    assert module.projector_top.weight.dtype==torch.bfloat16




@pytest.mark.parametrize("kind", ["middle", "student"])
def test_real_model_checkpoint_reload(kind, tmp_path):
    import json
    from pathlib import Path
    from src.evaluation.model_loader import load_encoder, EvaluationEncoder

    if kind == "middle":
        from src.middle_teacher.formal_config import to_runtime_config
        from src.middle_teacher.model import build_middle_teacher
        from src.middle_teacher.artifacts import MiddleCheckpointController
        config_path = Path("configs/middle_teacher/m0_infonce_224.json")
        public = json.loads(config_path.read_text())
        runtime = to_runtime_config(config_path)
        runtime["checkpoint"]["output_dir"] = str(tmp_path)
        model = build_middle_teacher(runtime, load_foundation=True).bfloat16().eval()
        signature = selection_signature(model, kind, 224, selection_batch_size=16)
        controller = MiddleCheckpointController(tmp_path, runtime, public_config=public)
        metrics = {direction + "_" + metric: 1.0 for direction in ("D2S", "S2D")
                   for metric in ("R1", "R5", "AP")}
        controller.save_best_if_improved(model, 6, 1, metrics)
        path = tmp_path / "best_model.pth"
        dimension = 768
    else:
        from src.student.model import StudentModel
        from src.student.formal_checkpoint import SCHEMA, fingerprint, validate_checkpoint
        config_path = Path("configs/student/s0_infonce_224.json")
        public = json.loads(config_path.read_text())
        model = StudentModel(ckpt_path=None).bfloat16().eval()
        signature = selection_signature(model, kind, 224, selection_batch_size=16)
        metrics = {direction: {"R@1": 1.0, "R@5": 1.0, "AP": 1.0}
                   for direction in ("D2S", "S2D")}
        payload = {
            "artifact_schema": SCHEMA,
            "model": {name: tensor.detach().cpu().clone()
                      for name, tensor in model.state_dict().items()},
            "public_config": public,
            "metadata": {
                "experiment_id": public["experiment_id"], "image_size": 224,
                "config_sha256": fingerprint(public), "best_epoch": 11,
                "best_score": 2.0, "selection_world_size": 1,
                "selection_batch_size": 16,
                "asset_sha256": {"student_pretrained": "a" * 64},
                "source_identity": {"fixture": "a" * 64},
            },
            "precision_signature": signature,
            "selection_metrics": metrics,
            "training_auxiliary": None,
        }
        validate_checkpoint(payload)
        path = tmp_path / "best_model.pth"
        torch.save(payload, path)
        dimension = 512
        config_path = None
    encoder, audit = load_encoder(kind, path, config_path, device="cpu")
    assert audit["precision_signature"] == signature
    assert not audit["missing"] and not audit["unexpected"]
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        before = EvaluationEncoder(model, dimension)(x)
        after = encoder(x)
        assert before.shape == (1, dimension) and before.dtype == torch.float32
        assert torch.equal(before, after)


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
