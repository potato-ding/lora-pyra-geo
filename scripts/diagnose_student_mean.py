"""Native-resolution, fused-model launch/kernel diagnostic; not main-table data."""
import json,os,sys,time,statistics
from pathlib import Path
import torch
from torch import nn
from torch.profiler import profile,ProfilerActivity
from src.evaluation.efficiency import RetrievalForward,fp32_context,sample_inputs,prepare_deployment,ParameterUse
from src.evaluation.model_loader import load_encoder
from src.evaluation.comparison_models import load_comparison

kind=sys.argv[1];out=Path(sys.argv[2]);root=Path('/home/dingyi/lora-pyra-geo')
torch.set_num_threads(2);torch.manual_seed(20261011)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
if kind=='student':
 enc,audit=load_encoder(kind,root/'src/checkpoint/student/R224/S3-224/best_model.pth',device='cuda:0',image_size=224);size=224
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
(out/(kind+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['model','resolution','eager_wall_b1','eager_wall_b16','profiler_b1','profiler_b16','graph_wall_b1','graph_parity_max_abs']}),flush=True)
