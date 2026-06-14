import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.student_model import StudentModel


def main():
    torch.manual_seed(0)

    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 384, 384)

    with torch.no_grad():
        default_desc = model(x)
        _, _, f3, f4 = model.backbone(x)

    assert f4.shape[1] == 512

    print(f"default desc shape: {tuple(default_desc.shape)}")
    print(f"f3 shape: {tuple(f3.shape)}")
    print(f"f4 shape: {tuple(f4.shape)}")


if __name__ == "__main__":
    main()
