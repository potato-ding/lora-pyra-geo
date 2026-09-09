import torch
import torch.nn.functional as F

def _direction(middle,teacher,anchor_ids,candidate_ids,top_k,temperature):
    positive=anchor_ids[:,None].eq(candidate_ids[None,:]); masked=teacher.masked_fill(positive,float("-inf")); k=min(int(top_k),masked.shape[1]-1); idx=masked.topk(k,dim=1).indices
    t=torch.gather(teacher,1,idx); s=torch.gather(middle,1,idx); p=teacher.masked_select(positive).reshape(middle.shape[0],1); tp=torch.softmax(torch.cat((p,t),1)/float(temperature),1); sp=F.log_softmax(torch.cat((middle.masked_select(positive).reshape(middle.shape[0],1),s),1)/float(temperature),1); return F.kl_div(sp,tp,reduction="batchmean")
def nrkd(middle_drone,middle_satellite,teacher_drone,teacher_satellite,drone_ids,satellite_ids,top_k=8,temperature=0.2):
    sd=F.normalize(middle_drone.float(),dim=1); ss=F.normalize(middle_satellite.float(),dim=1); td=F.normalize(teacher_drone.detach().float(),dim=1); ts=F.normalize(teacher_satellite.detach().float(),dim=1); sm=sd@ss.t(); tm=td@ts.t(); return 0.5*(_direction(sm,tm,drone_ids,satellite_ids,top_k,temperature)+_direction(sm.t(),tm.t(),satellite_ids,drone_ids,top_k,temperature))
