"""Eight-step two-GPU canonical Student BN save/reload gate. Output must be fresh."""

def main():
    import os,sys,json,hashlib,inspect,subprocess
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,Subset
    from torch.utils.data.distributed import DistributedSampler
    ROOT=Path('/home/dingyi/lora-pyra-geo');sys.path.insert(0,str(ROOT));os.chdir(ROOT)
    OUT=Path(sys.argv[1])
    from src.student.train import StudentTrainingModel,batch_loss,deepspeed_config,load_config,sync_student_buffers_from_rank0,assert_student_validation_state_synced
    from src.student.model import StudentModel
    from src.student.artifacts import deployment_state_dict,file_sha256,best_record
    from src.student.runtime import _seed_all,_seed_stst_worker
    from src.student.data import create_student_train_dataset_and_loader
    from src.student.optimizer import build_student_optimizer
    from src.student.scheduler import build_student_scheduler
    from src.student.objective import PairInfoNCE
    from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders,IndexedDataset
    from src.utils.train_eval_utils import extract_features_dist,getdist_1652_val_and_get_recall
    import deepspeed
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dev=torch.device('cuda',rank);torch.set_num_threads(4)
    deepspeed.init_distributed(dist_backend='nccl')
    if rank==0:OUT.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    def write(name,x):
     (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    def hashes(s):return {k:hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in s.items()}
    def digest(s):return hashlib.sha256(json.dumps(hashes(s),sort_keys=True).encode()).hexdigest()
    def diff(a,b):
     return {k:dict(shape=list(a[k].shape),dtype=str(a[k].dtype),other_dtype=str(b[k].dtype),max_abs_diff=float((a[k].double()-b[k].double()).abs().max()) if a[k].numel() else 0.) for k in a}
    def maxdiff(rows,keys):return max([rows[k]['max_abs_diff'] for k in keys] or [0.])
    def paircompare(a,b):
     out={}
     for d in a:
      out[d]={}
      for kind in ['q','g']:
       x,y=a[d][kind],b[d][kind]
       out[d][kind]=dict(exact=torch.equal(x,y),max_abs_diff=float((x-y).abs().max()),mean_abs_diff=float((x-y).abs().mean()),hash_a=digest({'x':x}),hash_b=digest({'x':y}))
      out[d]['metrics_a']=a[d]['metrics'];out[d]['metrics_b']=b[d]['metrics']
      out[d]['metric_exact']=a[d]['metrics']==b[d]['metrics']
     return out
    cfg=load_config(ROOT/'configs/student/certified_r224/b0_baseline_s0.json');_seed_all(cfg['seed']);args=SimpleNamespace(**cfg)
    loader=create_student_train_dataset_and_loader(args);loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(temperature=cfg['temperature'],ckpt_path=cfg['student_pretrained']).to(dev)
    model=StudentTrainingModel(student,None).to(dev)
    opt=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=build_student_scheduler(opt,args,steps_per_epoch=len(loader))
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=opt,lr_scheduler=sched,config=deepspeed_config())
    criterion=PairInfoNCE(label_smoothing=cfg['label_smoothing'])
    engine.train();loader.batch_sampler.set_epoch(1);losses=[]
    for step,batch in enumerate(loader):
     images=torch.cat(batch[:2]).to(dev,non_blocking=True)
     loss,_=batch_loss(engine,None,images,len(batch[0]),criterion,cfg,1)
     assert torch.isfinite(loss)
     engine.backward(loss);engine.step();losses.append(float(loss.detach()))
     if step==7:break
    del loader
    student=engine.module.student
    before_sync=state(student)
    sync_student_buffers_from_rank0(student)
    validation_state=assert_student_validation_state_synced(student,1)
    assert all(torch.equal(before_sync[n],p.detach().cpu()) for n,p in student.named_parameters())
    engine.eval()
    a=state(student);namesp=list(dict(student.named_parameters()));namesb=list(dict(student.named_buffers()))
    bn={k:[] for k in ['weight','bias','running_mean','running_var','num_batches_tracked']}
    for n,m in student.named_modules():
     if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
      for k in bn:
       if getattr(m,k,None) is not None:bn[k].append(n+'.'+k)
    r0={k:v.to(dev) for k,v in a.items()}
    for v in r0.values():dist.broadcast(v,src=0)
    r0={k:v.cpu() for k,v in r0.items()}
    rd=diff(a,r0)
    write('rank%d_state_inventory.json'%rank,dict(parameters=namesp,buffers=namesb,bn_keys=bn,state_a_hash=digest(a),tensor_hashes=hashes(a),diff_vs_rank0=rd))
    # Fixed, identical 128 identities; 4 drones per identity; canonical transforms.
    full=build_1652_val_dataloaders(data_dir=cfg['val_data_dir'],img_size=[224,224],batch_size=32,num_workers=2)
    common=sorted(set(full['S2D'][0].dataset.sample_ids))[::5][:128]
    datasets={};manifest={}
    for d,pair in full.items():
     datasets[d]=[];manifest[d]=[]
     for j,ld in enumerate(pair):
      ds=ld.dataset.dataset;counts={};ix=[]
      cap=4 if (d=='D2S' and j==0) or (d=='S2D' and j==1) else 1
      for i,sid in enumerate(ds.sample_ids):
       if sid in common and counts.get(sid,0)<cap:
        ix.append(i);counts[sid]=counts.get(sid,0)+1
      datasets[d].append(IndexedDataset(Subset(ds,ix)))
      manifest[d].append([ds.images[i] for i in ix])
    if rank==0:write('subset_manifest.json',manifest)
    def evaluate(m,distributed=False):
     result={}
     for d,ds in datasets.items():
      ls=[DataLoader(x,batch_size=32,num_workers=2,shuffle=False,sampler=DistributedSampler(x,shuffle=False) if distributed else None) for x in ds]
      with torch.no_grad():
       q,ql,_=extract_features_dist(m,ls[0],dev);g,gl,_=extract_features_dist(m,ls[1],dev)
       mm=getdist_1652_val_and_get_recall(m,*ls,dev,precomputed_features=(q,ql,None,g,gl,None))
      result[d]=dict(q=q,g=g,ql=ql,gl=gl,metrics=dict(zip(['R1','R5','R10','AP'],mm)))
     return result
    with patch.object(dist,'is_initialized',return_value=False):local=evaluate(student)
    b=state(student);dep=deployment_state_dict(engine);dd=diff(b,dep)
    # Local immediate reload preserves each rank's entire state.
    tmp=OUT/('temporary_rank%d.pth'%rank);torch.save(dict(model=dep),tmp)
    reload=StudentModel(ckpt_path=None).to(dev).bfloat16()
    reload.load_state_dict(torch.load(tmp,map_location='cpu',weights_only=False)['model'],strict=True);reload.eval()
    with patch.object(dist,'is_initialized',return_value=False):reloaded=evaluate(reload)
    c=state(student)
    torch.save(dict(in_memory=local,reloaded=reloaded),OUT/('rank%d_descriptors.pt'%rank))
    objects=[None,None];dist.all_gather_object(objects,local)
    # Counterfactual ONLY in throwaway reloaded model: use rank0 buffers, retain rank-local parameters.
    for n,v in reload.named_buffers():v.copy_(r0[n].to(dev))
    with patch.object(dist,'is_initialized',return_value=False):samebuffers=evaluate(reload)
    sb=[None,None];dist.all_gather_object(sb,samebuffers)
    # Real distributed validation mixes descriptors from the two unsynchronized live states.
    mixed=evaluate(student,True)
    # Both ranks reload the state that rank0 would actually save.
    reload.load_state_dict(torch.load(OUT/'temporary_rank0.pth',map_location='cpu',weights_only=False)['model'],strict=True);reload.eval()
    uniform=evaluate(reload,True)
    torch.save(dict(mixed=mixed,rank0_reload=uniform),OUT/('distributed_rank%d_descriptors.pt'%rank))
    summary=dict(rank=rank,steps=8,losses=losses,parameter_max_diff_vs_rank0=maxdiff(rd,namesp),buffer_max_diff_vs_rank0=maxdiff(rd,namesb),
     bn_max_diff_vs_rank0={k:maxdiff(rd,v) for k,v in bn.items()},state_a=digest(a),state_b=digest(b),state_c=digest(c),
     validation_changed_keys=[k for k in a if not torch.equal(a[k],b[k])],
     deployment_parameter_max_diff=maxdiff(dd,namesp),deployment_buffer_max_diff=maxdiff(dd,namesb),
     deployment_keys_equal=set(dep)==set(b),deployment_mismatch_keys=[k for k in dd if dd[k]['max_abs_diff'] or dd[k]['dtype']!=dd[k]['other_dtype']],
     immediate=paircompare(local,reloaded),cross_rank=paircompare(objects[0],objects[1]),rank0_buffers_counterfactual=paircompare(sb[0],sb[1]),
     mixed_vs_rank0_reload=paircompare(mixed,uniform))
    write('rank%d_summary.json'%rank,summary)
    # The formal reload keeps FP32 parameter storage + BF16 autocast unchanged.
    from src.evaluation.model_loader import load_encoder
    formal,_=load_encoder('student',OUT/'temporary_rank0.pth',device=dev)
    formal_result=evaluate(formal,True)
    formal_comparison=paircompare(mixed,formal_result)
    write('rank%d_formal_reload.json'%rank,formal_comparison)
    def descriptors_match(comparison):
        return all(row[kind]['exact'] for row in comparison.values() for kind in ['q','g'])
    def metrics_match(comparison):
        return all(row['metric_exact'] for row in comparison.values())
    gate=dict(BN_BUFFERS_SYNCED=True,
        VALIDATION_STATE_EQUALS_SAVED_STATE=not summary['deployment_mismatch_keys'] and digest(a)==digest(b)==digest(state(student)),
        IMMEDIATE_SAVE_RELOAD_DESCRIPTOR_MATCH=descriptors_match(summary['mixed_vs_rank0_reload']),
        IMMEDIATE_SAVE_RELOAD_METRIC_MATCH=metrics_match(summary['mixed_vs_rank0_reload']),
        FORMAL_PRECISION_RELOAD_DESCRIPTOR_MATCH=descriptors_match(formal_comparison),
        FORMAL_PRECISION_RELOAD_METRIC_MATCH=metrics_match(formal_comparison),
        PARAMETERS_UNCHANGED=True,
        RANK_DESCRIPTOR_MATCH=descriptors_match(summary['cross_rank']),
        RANK_METRIC_MATCH=metrics_match(summary['cross_rank']))
    write('rank%d_gate.json'%rank,gate)
    assert all(gate.values()), gate
    if rank==0:print('BN_CONSISTENCY_SMOKE_GATE='+json.dumps(gate),flush=True)
    if rank==0:
     write('source_save_audit.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),deployment_source=inspect.getsource(deployment_state_dict),train_source=(ROOT/'src/student/train.py').read_text(),deepspeed_config=deepspeed_config()))
    print('AUDIT_RANK_COMPLETE='+str(rank),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__ == "__main__":
    main()
