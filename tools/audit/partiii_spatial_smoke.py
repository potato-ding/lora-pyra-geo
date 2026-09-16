"""One real TRAIN backward; no optimizer, model save, or training-mode BN."""
from tools.audit.partiii_spatial_interface_audit import (
    OUT, TEACHER, STUDENT, TEACHER_SHA, STUDENT_SHA, read, write, runtime,
    models, state_sha, inputs, sha,
)


def main():
    runtime(2)
    import torch
    from src.student.spatial_kd import SameImageSpatialKD, SpatialKDTrainingContainer, spatial_tokens
    from src.student.model import StudentModel
    workers=[read('worker_0.json'),read('worker_1.json')]
    assert all(w['pass_status'] and w['spatial_features_finite'] for w in workers)
    assert all(w['teacher_inventory']['grid_order_pass'] and w['student_inventory']['grid_order_pass'] for w in workers)
    exact=read('exact_grid_match.json');primary=exact['primary'];assert primary
    grid=exact['teacher_grid'];prefix=read('teacher_spatial_inventory.json')['teacher_prefix_token_count']
    teacher,student,ta,sa=models()
    teacher.eval();student.eval()
    before=[state_sha(teacher),state_sha(student)]
    for p in student.backbone.parameters():p.requires_grad_(True)
    kd=SameImageSpatialKD(primary['C'],grid).cuda()
    wrapped=SpatialKDTrainingContainer(student,kd).eval()
    manifest=read('spatial_audit_manifest.json')['records']
    # Two images of each view, not a cross-view correspondence loss.
    batch=[r for r in manifest if r['identity_index']<2]
    x,checks=inputs(batch)
    captured={}
    def th(m,args,out):captured['teacher']=spatial_tokens(out,prefix_count=prefix,grid=grid)
    def sh(m,args,out):captured['student']=out
    handles=[teacher.backbone.model.norm.register_forward_hook(th),
             student.get_submodule(primary['module_name']).register_forward_hook(sh)]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        teacher(x)
        descriptor_before=student(x).detach().clone()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        descriptor_with_projector=wrapped(x)
        sf=captured['student'];tf=captured['teacher']
        d=[i for i,r in enumerate(batch) if r['view']=='drone']
        s=[i for i,r in enumerate(batch) if r['view']=='satellite']
        di=[batch[i]['image_path'] for i in d];si=[batch[i]['image_path'] for i in s]
        result=kd(sf[d],tf[d],sf[s],tf[s],drone_image_ids=di,teacher_drone_image_ids=di,
                  satellite_image_ids=si,teacher_satellite_image_ids=si)
    assert all(torch.isfinite(v) for v in result.values())
    result['loss'].backward()
    def grads(module):
        named=[(n,p.grad) for n,p in module.named_parameters() if p.grad is not None]
        assert named and all(torch.isfinite(g).all() for n,g in named)
        total=sum(float(g.float().abs().sum()) for n,g in named)
        assert total>0
        return dict(tensors=len(named),absolute_sum=total,all_finite=True,nonzero=True)
    stage_grad=grads(student.get_submodule(primary['module_name']))
    projector_grad=grads(kd)
    teacher_no_grad=all(p.grad is None for p in teacher.parameters())
    assert teacher_no_grad
    assert torch.equal(descriptor_before,descriptor_with_projector.detach())
    after=[state_sha(teacher),state_sha(student)]
    assert before==after
    bare_state=wrapped.deployment_state_dict()
    assert set(bare_state)==set(student.state_dict())
    fresh=StudentModel(ckpt_path=None).cuda().eval()
    strict=fresh.load_state_dict(bare_state,strict=True)
    assert not strict.missing_keys and not strict.unexpected_keys
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): fresh_descriptor=fresh(x)
    assert torch.equal(fresh_descriptor,descriptor_before)
    assert sum(p.numel() for p in fresh.parameters())==sum(p.numel() for p in student.parameters())
    for handle in handles:handle.remove()
    teacher.zero_grad(set_to_none=True);student.zero_grad(set_to_none=True);kd.zero_grad(set_to_none=True)
    assert all(p.grad is None for m in [teacher,student,kd] for p in m.parameters())
    assert sha(TEACHER)==TEACHER_SHA and sha(STUDENT)==STUDENT_SHA
    report=dict(implemented=True,backward_smoke_pass=True,gpu=2,backward_calls=1,optimizer_steps=0,
        scheduler_steps=0,model_weights_saved=False,student_eval_mode=True,teacher_eval_mode=True,
        batch=batch,geometric_checks=checks,student_feature_shape=list(sf.shape),teacher_feature_shape=list(tf.shape),
        prefix_count=prefix,projector='Conv2d(%d,768,kernel_size=1,bias=True)'%primary['C'],
        losses={k:float(v.detach()) for k,v in result.items()},teacher_grad_none=teacher_no_grad,
        student_selected_stage_gradient=stage_grad,projector_gradient=projector_grad,
        state_hash_before=before,state_hash_after=after,state_unchanged=True,all_grads_cleared=True,
        bare_student_strict_reload_pass=True,global_descriptor_exact_match=True,
        deployment_parameter_count_unchanged=True,deployment_macs_unchanged='Identical bare Student class, state keys and forward path',
        no_cross_view_position_matching=True,lambda_spatial=None,integrated_into_existing_trainers=False)
    write(OUT/'spatial_kd_implementation_check.json',report)
    print('SPATIAL_KD_BACKWARD_SMOKE_PASS=True',flush=True)


if __name__=='__main__':main()
