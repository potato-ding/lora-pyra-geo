"""Deterministic TRAIN images shared by formal Top128 and calibration builders."""
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from src.dataset.transforms import get_test_transforms
ROOT=Path(__file__).resolve().parents[2]

class TrainImages(Dataset):
    def __init__(self, domain, train_root=None, image_size=224):
        self.root = Path(train_root or ROOT / "data/U1652/train") / domain
        self.transform = get_test_transforms([image_size, image_size])
        self.ids = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.rows = []
        for pid in self.ids:
            paths = sorted(p for p in (self.root/pid).iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if not paths or (domain == "satellite" and len(paths) != 1):
                raise ValueError("Invalid TRAIN identity " + pid)
            self.rows.extend((p, int(pid)) for p in paths)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        path, label = self.rows[index]
        with Image.open(path) as im:
            array = np.array(im.convert("RGB"))
        return self.transform(image=array)["image"], label
