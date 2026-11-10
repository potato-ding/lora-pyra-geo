import sys,json,hashlib
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.student.formal_config import ROOT
from src.student.subspace_utils import tensor_sha256
from src.student.artifacts import file_sha256
from scripts.build_student_dimension_assets import extend_top
from src.student.build_bank import TrainImages
from src.student.subspace import identity_centroids
from src.evaluation.model_loader import load_encoder
torch.set_num_threads(4)
provider=sys.argv[1]
root=ROOT/'src/checkpoint/student/R256'
old=root/('teacher_assets/T0-256' if provider=='T0' else 'middle_assets/M0-256')
out=old/'dimension256_64'
assert not out.exists()
checkpoint=ROOT/f'src/checkpoint/{"teacher" if provider=="T0" else "middle_teacher/R256"}/{provider}-256/best_model.pth'
if provider=='T0':
 rows=torch.load(old/'train_identity_centroids.pt',weights_only=True,map_location='cpu')
else:
 model,_=load_encoder('middle',checkpoint,device='cuda',image_size=256)
 chunks=[]
 for domain in ('drone','satellite'):
  ds=TrainImages(domain,ROOT/'data/U1652/train',256)
  features=[];labels=[]
  with torch.inference_mode():
   for step,(images,ids) in enumerate(DataLoader(ds,batch_size=32,num_workers=4,shuffle=False)):
    features.append(model(images.cuda()).cpu());labels.append(ids)
    if step%200==0:print(domain,step,flush=True)
  centroids,ids=identity_centroids(torch.cat(features),torch.cat(labels));assert len(ids)==701
  chunks.append(centroids)
 rows=torch.cat(chunks)
mean=torch.load(old/'top128/teacher_mean.pt',weights_only=True)
prefix=torch.load(old/'top128/top128_basis.pt',weights_only=True)
assert rows.shape==(1402,mean.numel())
top=extend_top(rows,mean,prefix)[:,:256].contiguous()
d2=root/'middle_assets/M3-256/dimension_extension'
if provider=='M0':
 random=torch.load(d2/'random_basis.pt',weights_only=True)
else:
 gen=torch.Generator().manual_seed(3581722385785991807)
 q=torch.linalg.qr(torch.randn(4096,32,generator=gen,dtype=torch.float64),mode='reduced').Q
 def sign(q):
  piv=q.abs().argmax(0)
  return q*torch.where(q[piv,torch.arange(q.shape[1])]<0,-1.,1.)
 q=sign(q).float()
 extra=torch.randn(4096,32,generator=gen,dtype=torch.float64)
 p=q.double();extra-=p@torch.linalg.solve(p.T@p,p.T@extra)
 random=torch.cat([q,sign(torch.linalg.qr(extra,mode='reduced').Q).float()],1).contiguous()
out.mkdir()
tensors={'teacher_mean':mean,'top_basis':top,'random_basis':random}
for n,t in tensors.items():
 assert t.dtype==torch.float32 and t.isfinite().all()
 if n!='teacher_mean':assert (t.T@t-torch.eye(t.shape[1])).abs().max()<1e-5
 torch.save(t,out/(n+'.pt'))
manifest=dict(schema='PROVIDER256_64_V1',provider=provider,resolution=256,split='train',rows=1402,identities_per_view=701,
 checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
 old_manifest_sha256=file_sha256(old/'top128/manifest.json'),
 random_seed=3581722385785991807,top_protocol='original Top128 plus constrained TRAIN residual PCA extension',
 random_protocol='same seed Gaussian QR32 plus Gaussian orthogonal completion64',
 tensor_sha256={n:tensor_sha256(t) for n,t in tensors.items()})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+chr(10))
print('ASSET_PASS',provider,out,flush=True)
