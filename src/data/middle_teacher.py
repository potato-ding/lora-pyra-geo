
"""Minimal University-1652 paired data path for Middle Teacher training."""
from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader, get_worker_info
from src.dataset.teacher.datasets import CrossViewPairSampler
from src.dataset.transforms import get_train_transforms

def _read(path): return np.array(Image.open(path).convert("RGB"))
class University1652MiddlePairDataset(Dataset):
    def __init__(self,data_dir,satellite_transform,drone_transform,flip_probability=0.5):
        self.satellite_transform=satellite_transform; self.drone_transform=drone_transform
        self.flip_probability=float(flip_probability); self.pairs=[]; self.pair_pids=[]
        satellite_root=os.path.join(data_dir,"satellite"); drone_root=os.path.join(data_dir,"drone")
        for pid in sorted(os.listdir(satellite_root)):
            sat_dir=os.path.join(satellite_root,pid); drone_dir=os.path.join(drone_root,pid)
            if not os.path.isdir(sat_dir) or not os.path.isdir(drone_dir): continue
            sat=sorted(os.path.join(sat_dir,n) for n in os.listdir(sat_dir) if n.lower().endswith((".jpg",".jpeg",".png")))
            drones=sorted(os.path.join(drone_dir,n) for n in os.listdir(drone_dir) if n.lower().endswith((".jpg",".jpeg",".png")))
            if not sat or not drones: continue
            label=len(set(self.pair_pids))
            for drone in drones: self.pairs.append((pid,label,sat[0],drone)); self.pair_pids.append(pid)
        if not self.pairs: raise RuntimeError(f"no paired data under {data_dir}")
    def __len__(self): return len(self.pairs)
    def __getitem__(self,index):
        pid,label,satellite_path,drone_path=self.pairs[index]; satellite=_read(satellite_path); drone=_read(drone_path)
        if random.random()<self.flip_probability:
            satellite=np.ascontiguousarray(np.flip(satellite,axis=1)); drone=np.ascontiguousarray(np.flip(drone,axis=1))
        return self.drone_transform(image=drone)["image"],self.satellite_transform(image=satellite)["image"],label,pid
def _seed_middle_worker(_worker_id):
    seed=int(torch.initial_seed()%(2**32)); random.seed(seed); np.random.seed(seed); worker=get_worker_info()
    if worker is not None:
        worker.dataset.satellite_transform.set_random_seed(seed+1); worker.dataset.drone_transform.set_random_seed(seed+2)
def create_middle_teacher_train_dataset_and_loader(config):
    data=config["data"]; _,satellite_transform,drone_transform=get_train_transforms(img_size=[data["input_size"]]*2,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    dataset=University1652MiddlePairDataset(data.get("train_dir","data/U1652/train"),satellite_transform,drone_transform)
    rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0; generator=torch.Generator(); generator.manual_seed(int(config["seed"])+100003*rank)
    sampler=CrossViewPairSampler(dataset,batch_size=int(data["local_pair_batch"]),shuffle=True,seed=int(config["seed"]))
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=int(data.get("num_workers",8)),pin_memory=True,worker_init_fn=_seed_middle_worker,generator=generator)
    return dataset,loader
