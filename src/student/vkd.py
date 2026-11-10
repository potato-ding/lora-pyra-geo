"""VkD adaptation of roymiles/VkD train.py:184 and engine.py:87-92.

Keep the pretrained retrieval encoder; distill pre-L2 BN features through an
FP32 semi-orthogonal projection to a LayerNorm M3 descriptor with SmoothL1.
No PCA, Random32, GBW, logits KD or additional trainable normalization.
"""
import json
import torch
from torch import nn
import torch.nn.functional as F
from .part2 import FP32Module
from .formal_engine import StudentTrainingModel
from .formal_methods import task_objective
from .adsd import combine_task_and_kd

class VkDHead(FP32Module):
    def __init__(self):
        super().__init__()
        self.projector=nn.utils.parametrizations.orthogonal(
            nn.Linear(512,768,bias=False,dtype=torch.float32),
            orthogonal_map="householder",use_trivialization=True)

    def forward(self, features, targets):
        with torch.autocast(device_type=features.device.type,enabled=False):
            prediction=self.projector(features.float())
            target=F.layer_norm(targets.detach().float(),(768,),eps=1e-5)
            loss=F.smooth_l1_loss(prediction,target,beta=1.,reduction="mean")
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite VkD loss")
            return loss

    @torch.no_grad()
    def orthogonality_error(self):
        w=self.projector.weight.float()
        return float((w.T@w-torch.eye(512,device=w.device)).abs().max())


def forward_features(student, images):
    captured=[]
    hook=student.neck.register_forward_hook(lambda module,args,out: captured.append(out))
    try:
        descriptor=student(images)
    finally:
        hook.remove()
    if len(captured)!=1 or captured[0].shape != descriptor.shape:
        raise RuntimeError("VkD requires exactly one shared BN512 output")
    return descriptor,captured[0].float()

class VkDTrainingModel(StudentTrainingModel):
    def forward(self,images):
        return forward_features(self.student,images)


def objective(student,head,outputs,targets,criterion,cfg,epoch,*,search_only=False):
    descriptors,features=outputs
    task=descriptors.new_zeros(()) if search_only else task_objective(
        student,descriptors,criterion,cfg["local_pair_batch"])
    kd=head(features,targets)
    loss,weight=combine_task_and_kd(task,kd,cfg["distillation"]["weight"],epoch,
        cfg["distillation"]["warmup_epochs"])
    return loss,None,{"InfoNCE":task.detach(),"VkD":kd.detach(),
        "effective_KD_loss":(weight*kd).detach(),"effective_kd_weight":weight}


def sam_backward(engine,middle,images,criterion,cfg,epoch,named):
    from .distill_sam import student_sam_step,isolated_search_state
    if middle.training or any(p.requires_grad for p in middle.parameters()):
        raise RuntimeError("VkD provider must be frozen")
    with torch.no_grad():
        targets=middle(images.to(getattr(middle,"input_dtype",torch.bfloat16))).detach().float()
    student_images=images.to(next(engine.module.student.parameters()).dtype)
    class Search(VkDTrainingModel):
        def forward(self,x):
            loss,_,parts=objective(self.student,self.supervision,super().forward(x),targets,
                criterion,cfg,epoch,search_only=True)
            return loss,{"first_pass_kd_weight":parts["effective_kd_weight"]}
    search=Search(engine.module.student,engine.module.supervision)
    state,search_named=isolated_search_state(search,named)
    result=student_sam_step(engine,named,
        lambda:torch.func.functional_call(search,state,(student_images,),strict=True),
        lambda:objective(engine.module.student,engine.module.supervision,engine(student_images),
            targets,criterion,cfg,epoch),search_named=search_named)
    error=engine.module.supervision.orthogonality_error()
    if not error < 1e-4:raise RuntimeError("VkD projection lost orthogonality")
    result[3].update(FIRST_PASS_TOP_INCLUDED=False,FIRST_PASS_RANDOM_INCLUDED=False,
        FIRST_PASS_VKD_INCLUDED=True,GBW_PRESENT=False,VKD_ORTHOGONALITY_ERROR=error)
    count=getattr(sam_backward,"_audit_count",0)
    if count<3:
        print("STUDENT_SAM_AUDIT="+json.dumps(result[3]),flush=True)
        sam_backward._audit_count=count+1
    return result


def protect_householder_optimizer(model,optimizer):
    """Do not decay the fixed +/-1 Householder diagonal via AdamW.

    PyTorch casts it to int in forward; any decay would turn +/-1 into zero.
    Keep Student groups untouched; place the constrained coordinate tensor in
    an explicit zero-decay FP32 group before scheduler/DeepSpeed initialization.
    """
    original=model.supervision.projector.parametrizations.weight.original
    found=0
    for group in optimizer.param_groups:
        found+=sum(p is original for p in group["params"])
        group["params"]=[p for p in group["params"] if p is not original]
    if found!=1:raise RuntimeError("VkD projector optimizer ownership mismatch")
    optimizer.add_param_group({"params":[original],"name":"vkd_orthogonal",
        "lr":optimizer.param_groups[0]["lr"],"weight_decay":0.})
