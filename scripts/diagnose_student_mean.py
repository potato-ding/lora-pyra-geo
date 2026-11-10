"""Native-resolution, fused-model launch/kernel diagnostic; not main-table data."""
import argparse,json,os,sys,time,statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.profiler import profile,ProfilerActivity
from src.evaluation.efficiency import RetrievalForward,fp32_context,sample_inputs,prepare_deployment,ParameterUse
from src.evaluation.model_loader import load_encoder
from src.evaluation.comparison_models import load_comparison, SPECS
from scripts.run_efficiency_suite import comparison_input_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["student", *SPECS])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Explicit Student checkpoint; defaults to S3-M3")
    parser.add_argument("--image-size", type=int, choices=(224, 256), help="Student input size; defaults to 224")
    parser.add_argument("--check", action="store_true", help="Validate paths without loading models or using a GPU")
    args = parser.parse_args(argv)
    if args.kind != "student" and (args.checkpoint is not None or args.image_size is not None):
        parser.error("--checkpoint and --image-size apply only to Student")
    if args.kind == "student":
        args.image_size = args.image_size or 224
        args.checkpoint = args.checkpoint or ROOT / f"src/checkpoint/student/R{args.image_size}/S3-{args.image_size}-M3/best_model.pth"
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
    else:
        args.image_size = SPECS[args.kind][2]
        paths = comparison_input_paths(args.kind, ROOT)
        args.checkpoint = paths[0][0]
        missing = [str(path) for path, kind in paths
                   if not (path.is_file() if kind == "file" else path.is_dir())]
        if missing:
            raise FileNotFoundError("Comparison inputs missing: " + "; ".join(missing))
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise FileExistsError(args.output_dir)
    result = args.output_dir / (args.kind + ".json")
    if result.exists():
        raise FileExistsError(result)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.check:
        print(json.dumps({"status": "preflight_pass", "model": args.kind, "checkpoint": str(args.checkpoint), "image_size": args.image_size}))
        return
    kind, out = args.kind, args.output_dir
    torch.set_num_threads(2);torch.manual_seed(20261011)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if kind=='student':
     enc,audit=load_encoder(kind,args.checkpoint,device='cuda:0',image_size=args.image_size);size=args.image_size
    else: enc,audit,size=load_comparison(kind,'cuda:0')
    enc.float().eval().requires_grad_(False);m=RetrievalForward(enc,kind).eval();x,_=sample_inputs(size,16)

    def wall(fn,batch,iterations=100):
     for _ in range(50):fn()
     torch.cuda.synchronize();rounds=[]
     for _ in range(3):
      vals=[];t0=time.perf_counter()
      for _ in range(iterations):
       if batch==1:torch.cuda.synchronize();start=time.perf_counter()
       y=fn()
       if batch==1:torch.cuda.synchronize();vals.append((time.perf_counter()-start)*1000)
       del y
      torch.cuda.synchronize();total=time.perf_counter()-t0
      rounds.append({'mean_ms':statistics.mean(vals) if vals else total*1000/iterations,'fps':batch*iterations/total})
     return {'rounds':rounds,'mean_ms':statistics.mean(r['mean_ms'] for r in rounds),'fps':statistics.mean(r['fps'] for r in rounds)}

    def kernels(batch):
     for _ in range(20):m(x[:batch])
     torch.cuda.synchronize()
     with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
      for _ in range(10):m(x[:batch])
      torch.cuda.synchronize()
     events=[e for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA]
     ops=sorted(p.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)
     return {'cuda_activities_per_forward':len(events)/10,
             'gpu_activity_sum_ms_per_forward':sum(e.device_time_total for e in events)/10/1000,
             'top_gpu_ops':[{'name':e.key,'count_per_forward':e.count/10,'self_cuda_ms_per_forward':e.self_device_time_total/10/1000} for e in ops[:12]]}

    with torch.inference_mode(),fp32_context():
     enc.encode(x[:1]);deployment=prepare_deployment(m)
     watch=ParameterUse(m)
     with watch:m(x[:1]);m(x)
     for label,entries in [('parameter',list(m.named_parameters())),('buffer',list(m.named_buffers()))]:
      for name,tensor in entries:
       if (label,name) not in watch.used:tensor.data=tensor.data.cpu()
     del watch,enc
     eager=wall(lambda:m(x[:1]),1)
     b16=wall(lambda:m(x),16)
     gpu1=kernels(1);gpu16=kernels(16)
     reference=m(x[:1]);side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(side):
      for _ in range(20):m(x[:1])
     torch.cuda.current_stream().wait_stream(side)
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):captured=m(x[:1])
     graph.replay();torch.cuda.synchronize()
     torch.testing.assert_close(captured,reference,atol=1e-6,rtol=1e-5)
     delta=float((captured-reference).abs().max())
     graph_time=wall(graph.replay,1)
    result={'model':kind,'resolution':size,'checkpoint_audit':audit,'deployment':deployment,
     'eager_wall_b1':eager,'eager_wall_b16':b16,'profiler_b1':gpu1,'profiler_b16':gpu16,
     'graph_wall_b1':graph_time,'graph_parity_max_abs':delta,
     'warning':'Diagnostic only: profiling adds overhead; activity sums are not unprofiled latency. CUDA Graph is a different execution backend and is not main-table data.'}
    out.mkdir(parents=True,exist_ok=True)
    with (out/(kind+'.json')).open('x') as result_file:
     result_file.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['model','resolution','eager_wall_b1','eager_wall_b16','profiler_b1','profiler_b16','graph_wall_b1','graph_parity_max_abs']}),flush=True)


if __name__ == "__main__":
    main()
