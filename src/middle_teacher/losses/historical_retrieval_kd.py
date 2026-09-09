"""Historical negative-only NRKD and ABS_MARGIN (b9cbcb2).

Kept separate from the retained positive-plus-negative NRKD variant.
KL is KL(teacher || middle), with batchmean reduction and no T**2.
"""
import torch
import torch.nn.functional as F


def nrkd_direction(student, teacher, query_ids, gallery_ids, top_k=8, temperature=0.2):
    positive = query_ids[:, None].eq(gallery_ids[None, :])
    if torch.any((~positive).sum(1) < top_k):
        raise RuntimeError('insufficient wrong-identity negatives')
    indices = teacher.masked_fill(positive, float('-inf')).topk(top_k, dim=1).indices
    t = torch.gather(teacher, 1, indices)
    s = torch.gather(student, 1, indices)
    probability = F.softmax(t / temperature, dim=1).detach()
    log_probability = F.log_softmax(s / temperature, dim=1)
    loss = F.kl_div(log_probability, probability, reduction='batchmean')
    return loss, dict(indices=indices, teacher_probability=probability,
                      student_log_probability=log_probability, valid_mask=~positive)


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


def historical_losses(md, ms, td, ts, ids, config):
    # Margin historically consumes model-normalized descriptors directly;
    # NRKD explicitly normalizes again. Do not collapse these two contracts.
    result = {}
    if config.get('nrkd', {}).get('enabled'):
        c = config['nrkd']
        sm = F.normalize(md.float(),dim=1) @ F.normalize(ms.float(),dim=1).t()
        tm = F.normalize(td.detach().float(),dim=1) @ F.normalize(ts.detach().float(),dim=1).t()
        a, ad = nrkd_direction(sm,tm,ids,ids,c['top_k'],c['temperature'])
        b, bd = nrkd_direction(sm.t(),tm.t(),ids,ids,c['top_k'],c['temperature'])
        result['nrkd'] = ((a+b)*0.5, dict(D2S=ad,S2D=bd))
    if config.get('margin', {}).get('enabled'):
        sm = md.float() @ ms.float().t()
        tm = td.detach().float() @ ts.detach().float().t()
        a, ad = margin_direction(sm,tm,ids,ids)
        b, bd = margin_direction(sm.t(),tm.t(),ids,ids)
        result['margin'] = ((a+b)*0.5, dict(D2S=ad,S2D=bd))
    return result
