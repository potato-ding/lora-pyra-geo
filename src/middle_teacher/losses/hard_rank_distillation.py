"""Teacher-selected Top5 wrong-identity absolute margin distillation."""
import torch

def margin_direction(student, teacher, query_ids, gallery_ids, top_k=5):
    n = student.size(0)
    same = query_ids[:, None].eq(gallery_ids[None, :])
    diagonal = torch.arange(n, device=student.device)
    if not bool(same[diagonal, diagonal].all()):
        raise RuntimeError('paired positive diagonal identity mismatch')
    if not bool(((~same).sum(1) == n-1).all()):
        raise RuntimeError('margin requires unique identity per global pair')
    indices = teacher.masked_fill(same, float('-inf')).topk(min(top_k,n-1), dim=1, sorted=True).indices
    sm = student[diagonal, diagonal].unsqueeze(1) - torch.gather(student,1,indices)
    tm = teacher[diagonal, diagonal].unsqueeze(1) - torch.gather(teacher,1,indices)
    return (sm-tm).abs().mean(), dict(indices=indices, student_margin=sm,
                                     teacher_margin=tm, valid_mask=~same)


def hard_rank_losses(md,ms,td,ts,ids,config):
    if not config.get('margin',{}).get('enabled'):return {}
    sm=md.float()@ms.float().t()
    tm=td.detach().float()@ts.detach().float().t()
    a,ad=margin_direction(sm,tm,ids,ids)
    b,bd=margin_direction(sm.t(),tm.t(),ids,ids)
    return {'margin':((a+b)*.5,dict(D2S=ad,S2D=bd))}
