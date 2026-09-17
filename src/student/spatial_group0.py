"""Independent Part-III helpers; no existing trainer imports or enables these."""
import torch
from torch import nn
from torch.nn import functional as F
from .spatial_kd import SameImageSpatialKD

DEFAULT_SPATIAL_FLAGS = dict(spatial_kd=False, relational_spatial_kd=False,
    multistage_spatial_kd=False, shift_spatial_kd=False, stable_region_kd=False)


class Stage3PointwiseSpatialKD(SameImageSpatialKD):
    """Shared FP32 linear spatial projector; same-image FP32 cosine loss."""
    def __init__(self):super().__init__(256,(14,14),768)
    def view_loss(self, student, teacher, **kwargs):
        with torch.autocast(device_type=student.device.type,enabled=False):
            return super().view_loss(student.float(),teacher,**kwargs)


def centered_relations(tokens):
    if tokens.ndim!=3 or tokens.shape[1]<2:raise ValueError('Expected B,N,C tokens')
    with torch.autocast(device_type=tokens.device.type,enabled=False):
        normalized=F.normalize(tokens.float(),dim=-1)
        gram=normalized@normalized.transpose(1,2)
        mask=~torch.eye(tokens.shape[1],dtype=torch.bool,device=tokens.device)
        off=gram[:,mask]
        return off-off.mean(dim=1,keepdim=True)


class CenteredSpatialRelationKD(nn.Module):
    """Projector-free, per-image off-diagonal centered relation cosine."""
    def view_loss(self,student,teacher,*,student_image_ids,teacher_image_ids):
        if student.ndim!=4 or tuple(student.shape[1:])!=(256,14,14):raise ValueError('Stage3 exact grid required')
        if tuple(teacher.shape)!=(len(student),196,768):raise ValueError('Teacher spatial shape')
        if len(student_image_ids)!=len(student) or tuple(student_image_ids)!=tuple(teacher_image_ids):
            raise ValueError('Same-image order required')
        a=centered_relations(student.flatten(2).transpose(1,2))
        b=centered_relations(teacher.detach())
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('Nonfinite relation')
        return (1-F.cosine_similarity(a,b,dim=1)).mean()
    def forward(self,sd,td,ss,ts,*,drone_image_ids,teacher_drone_image_ids,satellite_image_ids,teacher_satellite_image_ids):
        if set(drone_image_ids)&set(satellite_image_ids):raise ValueError('Distinct full image IDs required')
        d=self.view_loss(sd,td,student_image_ids=drone_image_ids,teacher_image_ids=teacher_drone_image_ids)
        s=self.view_loss(ss,ts,student_image_ids=satellite_image_ids,teacher_image_ids=teacher_satellite_image_ids)
        return dict(loss=.5*(d+s),drone_loss=d,satellite_loss=s)


class Stage4TeacherPooling(nn.Module):
    """Fixed 2x2 mean pooling of final-normalized raw tokens, then FP32 L2."""
    def forward(self,tokens,normalize=True):
        if tokens.ndim!=3 or tuple(tokens.shape[1:])!=(196,768):raise ValueError('Teacher14 required')
        grid=tokens.detach().float().transpose(1,2).reshape(-1,768,14,14)
        pooled=F.avg_pool2d(grid,kernel_size=2,stride=2).flatten(2).transpose(1,2)
        return F.normalize(pooled,dim=-1) if normalize else pooled


class PatchAlignedShiftMapper:
    """(dx,dy) in image pixels: right/down positive. No wrap-around."""
    def __init__(self,dx,dy):
        if type(dx)!=int or type(dy)!=int or (dx,dy) not in ((16,0),(-16,0),(0,16),(0,-16),(16,16),(16,-16),(-16,16),(-16,-16)):
            raise ValueError('Only nonzero 16-pixel cardinal/diagonal shifts')
        self.dx,self.dy=dx,dy
    def indices(self,device='cpu'):
        y,x=torch.meshgrid(torch.arange(14,device=device),torch.arange(14,device=device),indexing='ij')
        xx=x+self.dx//16;yy=y+self.dy//16
        valid=(xx>=0)&(xx<14)&(yy>=0)&(yy<14)
        return (y[valid]*14+x[valid]).long(),(yy[valid]*14+xx[valid]).long()
    def translate(self,images,padding=0.):
        if images.shape[-2:]!=(224,224):raise ValueError('224 image required')
        dx,dy=self.dx,self.dy
        sx=slice(max(0,-dx),min(224,224-dx));sy=slice(max(0,-dy),min(224,224-dy))
        tx=slice(max(0,dx),min(224,224+dx));ty=slice(max(0,dy),min(224,224+dy))
        out=torch.full_like(images,padding);out[...,ty,tx]=images[...,sy,sx]
        return out
    def align(self,original,shifted):
        if original.shape!=shifted.shape or original.ndim!=3 or original.shape[1]!=196:
            raise ValueError('Aligned B,196,C token sequences required')
        a,b=self.indices(original.device)
        return original[:,a],shifted[:,b]


class TeacherStableWeight(nn.Module):
    def forward(self,original,shifted):
        if original.shape!=shifted.shape or original.ndim!=3:raise ValueError('Matching overlap shapes')
        cosine=F.cosine_similarity(original.detach().float(),shifted.detach().float(),dim=-1)
        confidence=((1+cosine)/2).clamp(0,1)  # Numeric range guard, not a hard mask.
        mean=confidence.mean(dim=1,keepdim=True)
        if not torch.isfinite(confidence).all() or (mean<=0).any():raise ValueError('Undefined all-zero stable weights')
        return confidence,confidence/mean
    @staticmethod
    def weighted_loss(per_position_loss,weights):
        if per_position_loss.shape!=weights.shape:raise ValueError('Overlap shapes differ')
        return ((per_position_loss*weights.detach()).sum(-1)/weights.detach().sum(-1)).mean()
    @staticmethod
    def relation_pair_weights(weights):return weights.detach().unsqueeze(-1)*weights.detach().unsqueeze(-2)
