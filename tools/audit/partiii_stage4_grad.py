"""GPU3-only TRAIN_INIT_GRAPH audit. No optimizer, selector, or saved tensors."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tarfile
from tools.audit import partiii_group0 as g0

OUT=g0.BASE/'PARTIII-STAGE4-GRAD-AUDIT-V1'

def write(name,value):
    from src.student.artifacts import write_json
    write_json(OUT/name,value)

def main():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(OUT)
    g0.setup(3)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from src.student.model import StudentModel
    from src.student.part1 import PartISupervision
    from src.student.part2_integration import prepare_top
    from src.student.runtime import _seed_all
    from src.student.objective import PairInfoNCE
    from src.student.spatial_kd import SameImageSpatialKD
    from src.student.spatial_group0 import Stage4TeacherPooling,centered_relations
    from src.student.spatial_group1 import pretrained_check
    cfg=json.loads(g0.CONFIG.read_text());manifest=g0.read('preflight_manifest.json')
    assert g0.sha(g0.TEACHER)==g0.TSHA and g0.sha(g0.PRETRAIN)==g0.PSHA
    _seed_all(0)
    student=StudentModel(ckpt_path=str(g0.PRETRAIN)).cuda()
    pretrained=pretrained_check(student,g0.PRETRAIN)
    supervision=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],g0.TSHA,128,'single32').cuda()
    teacher=g0.teacher_load();prepare_top(supervision,cfg)
    student.bfloat16();supervision.bfloat16()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(g0.SEED);point=SameImageSpatialKD(512,(7,7)).cuda()
    initial={k:v.detach().clone() for k,v in student.state_dict().items()}
    heads={k:v.detach().clone() for k,v in supervision.state_dict().items()}
    initial_hash=g0.state_hash(student);heads_hash=g0.state_hash(supervision)
    teacher_hash=g0.state_hash(teacher);point_hash=g0.state_hash(point)
    reference=g0.read('gpu0_pointwise_gradient_audit.json')
    assert initial_hash==reference['student_initial_hash'] and heads_hash==reference['heads_initial_hash']
    boundary=student.backbone.out_indices[-2]+1
    endpoint=student.backbone.out_indices[-1]
    # Membership from actual module parameter identities, not name prefix guesses.
    stage_ids={id(p) for m in list(student.backbone.features.children())[boundary:] for p in m.parameters()}
    names,parameters=zip(*student.backbone.named_parameters())
    offset=0;ranges=[];layout=[]
    for name,p in zip(names,parameters):
        if id(p) in stage_ids:ranges.append((offset,offset+p.numel()))
        layout.append(dict(name=name,shape=list(p.shape),numel=p.numel(),in_stage4=id(p) in stage_ids))
        offset+=p.numel()
    selections={'ALL_BACKBONE':np.arange(offset),'STAGE4_AND_DOWNSTREAM':np.concatenate([np.arange(a,b) for a,b in ranges])}
    cache={};ht=g0.capture_teacher(teacher,cache)
    hs=student.backbone.features[endpoint].register_forward_hook(lambda m,a,o:cache.update(stage4=o))
    hb=student.backbone.features[boundary].register_forward_hook(lambda m,a,o:cache.update(stage4_start_shape=list(o.shape)))
    OUT.mkdir(parents=True)
    write('stage4_grad_config.json',dict(graph='TRAIN_INIT_GRAPH',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=g0.ROOT,text=True).strip(),
        manifest_path=str(g0.OUT/'preflight_manifest.json'),manifest_sha256=g0.sha(g0.OUT/'preflight_manifest.json'),
        batch_count=16,pair_batch=32,seed=g0.SEED,student_init_seed=0,pretrained=pretrained,
        student_initial_hash=initial_hash,heads_initial_hash=heads_hash,group0_initial_states_match=True,
        teacher_sha256=g0.TSHA,pretrained_sha256=g0.PSHA,stage_start=boundary,stage_endpoint=endpoint,
        parameter_layout=layout,stage_ranges=ranges,pooling='raw final norm 14x14 -> mean2x2 stride2 -> FP32 token L2',
        global_kd_strength=.2,spatial_weights_applied=False,optimizer_steps=0,checkpoint_selection=False))
    pool=Stage4TeacherPooling();rows=[];inputs=[]
    for batch in manifest['batches']:
        student.load_state_dict(initial,strict=True);supervision.load_state_dict(heads,strict=True)
        student.train();supervision.train();x,ids,input_sha=g0.training_input(batch)
        expected=reference['input_checksums'][batch['batch']]
        assert input_sha==expected['input_sha256']
        inputs.append(dict(batch=batch['batch'],input_sha256=input_sha))
        shapes={};handles=[]
        for n,m in student.named_modules():
            if isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)):
                def capture(m,a,n=n):shapes.setdefault(n,[]).append(len(a[0]))
                handles.append(m.register_forward_pre_hook(capture))
        with torch.no_grad():td=teacher(x.bfloat16())
        descriptor=student(x.bfloat16())
        for h in handles:h.remove()
        assert len(shapes)==171 and all(v==[64] for v in shapes.values())
        s=cache['stage4'];assert tuple(s.shape)==(64,512,7,7)
        assert cache['stage4_start_shape']==[64,512,7,7]
        t=pool(cache['teacher'],normalize=False)
        ret=PairInfoNCE(.1)(descriptor[:32],descriptor[32:],student.logit_scale.exp())
        dual,ka=supervision(descriptor,td,32);kd=.2*dual;base=ret+kd
        assert torch.allclose(dual,ka['top_loss']+ka['random_loss'])
        pt=point(s[:32].float(),t[:32],s[32:].float(),t[32:],drone_image_ids=ids[:32],teacher_drone_image_ids=ids[:32],
            satellite_image_ids=ids[32:],teacher_satellite_image_ids=ids[32:])
        rel_per=1-F.cosine_similarity(centered_relations(s.flatten(2).transpose(1,2)),centered_relations(t),dim=1)
        rd=rel_per[:32].mean();rs=rel_per[32:].mean()
        sources=dict(ret=ret,kd=kd,base=base,point=pt['loss'],point_drone=pt['drone_loss'],point_satellite=pt['satellite_loss'],
                     relation=.5*(rd+rs),relation_drone=rd,relation_satellite=rs)
        gradients={}
        for name,loss in sources.items():
            assert torch.isfinite(loss)
            gs=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            vector=torch.cat([(torch.zeros_like(p) if grad is None else grad).detach().float().reshape(-1).cpu() for p,grad in zip(parameters,gs)]).numpy()
            assert np.isfinite(vector).all();gradients[name]=vector
            del gs
        for scope,index in selections.items():
            g={k:v[index] for k,v in gradients.items()};norms={k:float(np.linalg.norm(v.astype('float64'))) for k,v in g.items()}
            assert min(norms.values())>0
            row=dict(batch=batch['batch'],parameter_set=scope,**{k+'_norm':v for k,v in norms.items()},
                     **{k+'_loss':float(v.detach()) for k,v in sources.items()},point_relation_cos=g0.cosine(g['point'],g['relation']))
            for kind in ('point','relation'):
                for target in ('base','ret','kd'):row[kind+'_'+target+'_cos']=g0.cosine(g[kind],g[target])
                row[kind+'_drone_sat_cos']=g0.cosine(g[kind+'_drone'],g[kind+'_satellite'])
                row[kind+'_drone_sat_norm_ratio']=norms[kind+'_drone']/norms[kind+'_satellite']
            rows.append(row)
        assert all(p.grad is None for p in teacher.parameters())
        print('STAGE4_AUDIT_BATCH='+str(batch['batch']+1)+'/16 FINITE=True',flush=True)
        del descriptor,td,s,t,ret,dual,ka,kd,base,pt,rd,rs,rel_per,sources,gradients,g,x
        cache.clear()
    student.load_state_dict(initial);supervision.load_state_dict(heads)
    assert g0.state_hash(student)==initial_hash and g0.state_hash(supervision)==heads_hash
    assert g0.state_hash(teacher)==teacher_hash and g0.state_hash(point)==point_hash
    ht.remove();hs.remove();hb.remove()
    summary={scope:{k:g0.stats([r[k] for r in rows if r['parameter_set']==scope]) for k in rows[0] if k not in ('batch','parameter_set')} for scope in selections}
    a=summary['ALL_BACKBONE'];target=min(a['kd_norm']['median'],.25*a['ret_norm']['median'])
    for kind,label in (('point','pointwise'),('relation','relational')):
        raw=target/a[kind+'_norm']['median'];warning=not(.01<=raw<=1)
        write('stage4_'+label+'_gradient.json',dict(pass_status=True,summary={s:{k:v for k,v in values.items() if k.startswith(kind+'_') or k in ('ret_norm','kd_norm','base_norm')} for s,values in summary.items()},
            lambda_raw=raw,lambda_value=None if warning else float(format(raw,'.3g')),scale_warning=warning,
            target_gradient=target,lambda_basis='ALL_BACKBONE',state_restored=True,optimizer_steps=0))
    write('stage4_matched_gradient.json',dict(pass_status=True,point_relation_cos={s:v['point_relation_cos'] for s,v in summary.items()},
        inputs=inputs,input_group0_match=True,group0_initial_state_match=True,N64_forward=True,no_cross_view_position_matching=True))
    with (OUT/'stage4_per_batch.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'README.txt').write_text('Stage4 TRAIN_INIT_GRAPH gradient audit only. Same 16 Group-0 TRAIN batches and augmentations.\n'
        'No optimizer step, no selection, no formal training, no test benchmark. Raw pooled Teacher tokens normalized after pooling.\n'
        'STAGE4_AND_DOWNSTREAM defined from actual modules after the penultimate backbone output boundary; all backbone params only.\n'
        'Losses and gradients are unweighted spatial diagnostics; KD is full strength 0.2*(Top128+Random32).\n'
        'Lambda out of [0.01,1] is left null with scale warning. No automatic training.\n')
    names=['stage4_grad_config.json','stage4_pointwise_gradient.json','stage4_relational_gradient.json','stage4_matched_gradient.json','stage4_per_batch.csv','README.txt']
    archive=OUT/(OUT.name+'_RESULTS.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:
        for name in names:tar.add(OUT/name,arcname=name,recursive=False)
    with tarfile.open(archive) as tar:assert sorted(tar.getnames())==sorted(names) and all(m.isfile() for m in tar.getmembers())
    print(json.dumps(dict(AUDIT_PASS=True,PACKAGE_PATH=str(archive),PACKAGE_SHA256=g0.sha(archive))),flush=True)

if __name__=='__main__':main()
