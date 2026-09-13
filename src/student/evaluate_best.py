"""Formal best-only Student wrapper using the unchanged unified evaluator."""
import argparse
import json
from pathlib import Path
import tempfile
from .artifacts import file_sha256,write_json,package_results,validate_training_complete,ROOT,U1652_EVAL_BATCH_SIZE,require_u1652_eval_batch_size,require_valid_run

NAMES={"u1652":("test_1652.json","test_1652_best.json"),
       "sues200":("test_sues200.json","test_sues200_all_best.json"),
       "gta":("test_gta_cross_area_d2s.json","test_gta_cross_area_d2s_best.json")}

def publish_result(run,dataset,payload,checkpoint_sha):
    run=Path(run)
    require_valid_run(run)
    if dataset=="u1652":
        require_u1652_eval_batch_size(payload.get("u1652_eval_batch_size"))
    if payload["model_type"]!="student" or Path(payload["checkpoint"]).resolve()!=run/"best_model.pth":
        raise ValueError("Only Student best checkpoint results accepted")
    destination=run/NAMES[dataset][1]
    if destination.exists():raise FileExistsError(destination)
    payload=dict(payload,checkpoint_sha256=checkpoint_sha)
    write_json(destination,payload)
    return destination

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True)
    p.add_argument("--dataset",choices=["u1652","sues200","gta","all"],default="all")
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=U1652_EVAL_BATCH_SIZE)
    p.add_argument("--num-workers",type=int,default=8)
    p.add_argument("--package-only",action="store_true")
    args=p.parse_args(argv)
    if args.dataset in ("u1652","all") and not args.package_only:
        require_u1652_eval_batch_size(args.batch_size)
    run=Path(args.run_dir).resolve()
    validate_training_complete(run)
    if args.package_only:
        print(json.dumps(package_results(run),indent=2));return
    from src.evaluation.evaluate import main as unified_main
    checkpoint=run/"best_model.pth";initial=file_sha256(checkpoint)
    datasets=list(NAMES) if args.dataset=="all" else [args.dataset]
    for dataset in datasets:
        if (run/NAMES[dataset][1]).exists():raise FileExistsError(run/NAMES[dataset][1])
    for dataset in datasets:
        with tempfile.TemporaryDirectory(prefix="student_formal_eval_") as temporary:
            unified_main(["--model-type","student","--checkpoint",str(checkpoint),
                "--dataset",dataset,"--data-root",str(ROOT/"data"),"--device",args.device,"--batch-size",str(args.batch_size),
                "--num-workers",str(args.num_workers),"--output-dir",temporary])
            if file_sha256(checkpoint)!=initial:raise RuntimeError("Checkpoint changed during evaluation")
            payload=json.loads((Path(temporary)/NAMES[dataset][0]).read_text())
            publish_result(run,dataset,payload,initial)
    if all((run/name[1]).exists() for name in NAMES.values()):
        print(json.dumps(package_results(run),indent=2))
if __name__=="__main__":main()
