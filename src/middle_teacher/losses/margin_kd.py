import torch
def _direction(middle,teacher,anchor_ids,candidate_ids,top_k=5):
    n=middle.size(0); pos=anchor_ids[:,None].eq(candidate_ids[None,:]); diag=torch.arange(n,device=middle.device); masked=teacher.masked_fill(pos,float("-inf")); idx=masked.topk(min(top_k,n-1),dim=1).indices; sn=torch.gather(middle,1,idx); tn=torch.gather(teacher,1,idx); sm=middle[diag,diag,None]-sn; tm=teacher[diag,diag,None]-tn; return torch.abs(sm-tm).mean()
def margin_kd(middle_drone,middle_satellite,teacher_drone,teacher_satellite,drone_ids,satellite_ids):
    sm=middle_drone.float()@middle_satellite.float().t(); tm=teacher_drone.detach().float()@teacher_satellite.detach().float().t(); return 0.5*(_direction(sm,tm,drone_ids,satellite_ids)+_direction(sm.t(),tm.t(),satellite_ids,drone_ids))
