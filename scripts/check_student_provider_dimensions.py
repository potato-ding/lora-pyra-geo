from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def smoke(cfg,assets):
    import torch,json,tempfile,deepspeed,copy
    from types import SimpleNamespace
    from src.student.formal_config import ROOT
    from src.student.formal_train import select_student,collect_student_source_identity
    from src.student.data import create_student_train_dataset_and_loader
    from src.student.shared_subspaces import configure_shared_data
    from src.student.model import StudentModel
    from src.student.component_ablation import prepare,sam_backward
    from src.student.formal_engine import StudentTrainingModel,deepspeed_config
    from src.student.formal_top import prepare_student_precision_groups
    from src.student.optimizer import build_student_optimizer
    from src.student.scheduler import build_student_scheduler,assert_student_scheduler_groups
    from src.student.distill_sam import collect_optimizer_trainable_params,assert_engine_ownership
    from src.student.objective import PairInfoNCE
    from src.evaluation.model_loader import load_encoder
    device=torch.device("cuda",0);torch.cuda.set_device(0);deepspeed.init_distributed(dist_backend="nccl")
    assert torch.distributed.get_world_size()==1 and torch.cuda.device_count()==1
    torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    args=SimpleNamespace(img_size=256,batch_size=32,train_data_dir=str(ROOT/cfg["data_dir"]/"train"),num_workers=2,seed=0,epochs=30,warmup_epochs=.1,min_lr_ratio=.01)
    loader=create_student_train_dataset_and_loader(args);configure_shared_data(loader,0)
    student=StudentModel(str(ROOT/cfg["pretrained_checkpoint"]),temperature=.07).to(device)
    from src.student.provider_dimensions import load_provider
    middle=load_provider(cfg,device)
    cal=torch.load(cfg["top_calibration"],map_location="cpu",weights_only=True);sup,gate,seed=prepare(cfg,assets,cal)
    sup=sup.to(device);gate=gate.to(device) if gate is not None else None
    model=StudentTrainingModel(student,sup).to(device);opt=build_student_optimizer(model,cfg["lr"],cfg["weight_decay"])
    prepare_student_precision_groups(model,opt);schedule=build_student_scheduler(opt,args,steps_per_epoch=len(loader))
    named=collect_optimizer_trainable_params(model,opt,middle.parameters())
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=opt,lr_scheduler=schedule,config=deepspeed_config({"batch_size":32,"world_size":1,"grad_accum_steps":1,"precision":"bfloat16"}))
    assert_engine_ownership(engine,named)
    gate_opt=torch.optim.AdamW([gate.d],lr=1e-4,weight_decay=0.) if gate is not None else None
    gate_schedule=build_student_scheduler(gate_opt,args,steps_per_epoch=len(loader)) if gate_opt else None
    engine.train();loader.batch_sampler.set_epoch(0);iterator=iter(loader)
    for step in range(1,4):
        batch=next(iterator);images=torch.cat(batch[:2]).to(device);assert images.shape==(64,3,256,256)
        if gate_opt:gate_opt.zero_grad(set_to_none=True)
        loss,gl,parts,audit=sam_backward(engine,middle,images,PairInfoNCE(.07),cfg,1,gate,named,identities=batch[2].to(device))
        assert audit["PERTURB_RESTORE_EXACT"] and audit["FIRST_PASS_GRADIENT_LEAK"]=="NONE"
        assert all(p.grad is None for p in middle.parameters()) and (gate is None or gate.d.grad is None)
        before=engine.global_steps;sc=schedule.last_epoch;engine.step();assert_student_scheduler_groups(schedule)
        assert engine.global_steps==before+1 and schedule.last_epoch==sc+1
        if gl is not None:gl.backward();assert torch.isfinite(gate.d.grad);gate_opt.step();gate_schedule.step()
        print("DIMENSION_GPU_SMOKE="+json.dumps({"variant":cfg["ablation"]["variant"],"step":step,"loss":float(loss.detach()),"audit":audit}),flush=True)
    import src.student.formal_train as trainer
    evaluator=trainer.evaluate_student_u1652_canonical
    try:
        trainer.evaluate_student_u1652_canonical=lambda *a,**kw:{"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
        with tempfile.TemporaryDirectory(prefix="dimension_synthetic_reload_") as tmp:
            select_student(engine,cfg,Path(tmp),11,float("-inf"),assets,sup,gate,seed,collect_student_source_identity(cfg))
            reloaded,_=load_encoder("student",Path(tmp)/"best_model.pth",device=device,image_size=256)
            student.eval()
            with torch.no_grad():assert torch.equal(student(images[:2].to(torch.bfloat16)),reloaded(images[:2].to(torch.bfloat16)))
            print("DIMENSION_UNIFIED_RELOAD_MAX_DIFF=0; SYNTHETIC_FIXTURE_DELETED",flush=True)
    finally:trainer.evaluate_student_u1652_canonical=evaluator
    torch.distributed.destroy_process_group();print("DIMENSION_GPU_SMOKE=PASS",cfg["ablation"]["variant"],flush=True)


import sys
from src.student.formal_config import load_formal_config
from src.student.formal_train import validate_assets
cfg=load_formal_config(sys.argv[1])[1]
smoke(cfg,validate_assets(cfg))
