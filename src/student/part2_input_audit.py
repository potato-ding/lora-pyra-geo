"""Read-only TRAIN input-range diagnostic; no optimizer, checkpoint, or evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .artifacts import ROOT, file_sha256, write_json
from .data import create_student_train_dataset_and_loader, CrossViewPairSampler
from .model import StudentModel
from .runtime import _seed_all, _seed_stst_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if "_AUDITS" not in out.parts:
        raise ValueError("Diagnostic output must be in _AUDITS, never a formal run")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "input_distribution.json").exists():
        raise FileExistsError(out)
    cfg_path = ROOT / "configs/student/certified_r224/p1_t128_r32_s0.json"
    cfg = json.loads(cfg_path.read_text())
    protected = {key: file_sha256(cfg[key]) for key in
                 ("student_pretrained", "middle_checkpoint", "stst_asset", "original_stst_asset")}
    _seed_all(0)
    torch.set_num_threads(4)
    loader0 = create_student_train_dataset_and_loader(SimpleNamespace(**cfg))
    ds = loader0.dataset
    sampler = CrossViewPairSampler(ds, batch_size=32, shuffle=True, seed=0)
    sampler.set_epoch(1)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=8, pin_memory=True,
                        worker_init_fn=_seed_stst_worker)
    student = StudentModel(ckpt_path=cfg["student_pretrained"]).to("cuda:0").bfloat16()
    student.train()  # Match the tensor seen by the training-time head, including batch statistics.
    before = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    buffers = {k: v.detach().clone() for k, v in student.named_buffers()}
    samples, manifest = [], []
    selected = iter(sampler)
    with torch.no_grad():
        for step, batch in enumerate(loader):
            indices = next(selected)
            images = torch.cat(batch[:2]).to("cuda:0", dtype=torch.bfloat16)
            assert images.shape == (64, 3, 224, 224)
            z = student(images)
            assert z.shape == (64, 512) and z.dtype == torch.float32
            assert torch.isfinite(z).all()
            assert torch.allclose(z.norm(dim=1), torch.ones(64, device=z.device), atol=2e-6, rtol=0)
            samples.append(z.cpu())
            manifest.append([dict(pid=ds.samples[i][0], satellite=ds.samples[i][2],
                                  drone=ds.samples[i][3]) for i in indices])
            # TRAIN BN uses current batch statistics. Restore running buffers after
            # every diagnostic forward; persisted and in-memory state stay intact.
            for k, v in student.named_buffers():
                v.copy_(buffers[k])
            if step == 11:
                break
    assert len(samples) == 12
    assert all(torch.equal(v, student.state_dict()[k].cpu()) for k, v in before.items())
    z = torch.cat(samples)
    flat = z.flatten().double()
    quantiles = torch.quantile(flat, torch.tensor([.001, .01, .99, .999], dtype=torch.float64))
    abs_q = float(torch.quantile(flat.abs(), .999))
    radius = math.ceil(abs_q * 1.20 * 100) / 100
    coverage = float((flat.abs() <= radius).double().mean())
    assert coverage >= .998
    report = dict(projector_input_is_l2_normalized=True, input_min=float(flat.min()),
                  input_max=float(flat.max()), input_mean=float(flat.mean()),
                  input_std=float(flat.std(unbiased=False)),
                  **dict(zip(["input_p0_1", "input_p1", "input_p99", "input_p99_9"],
                             map(float, quantiles))),
                  grid_range=[-radius, radius], grid_rule="ceil(1.20 * q99.9(abs(input)) * 100) / 100",
                  observed_coverage=coverage, abs_q99_9=abs_q, batches=12, images=768,
                  elements=flat.numel(), seed=0, sample_epoch=1, pair_batch=32, image_size=224,
                  split="University-1652 TRAIN", model_mode="train, no_grad, restore BN buffers after every batch",
                  forward_dtype="BF16", descriptor_dtype="FP32",
                  state_unchanged=True, optimizer_steps=0, backward_calls=0,
                  config_sha256=file_sha256(cfg_path), protected_sha256=protected,
                  source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in
                                 [Path(__file__), ROOT/"src/student/model.py", ROOT/"src/student/data.py"]})
    assert all(file_sha256(cfg[k]) == h for k, h in protected.items())
    write_json(out / "input_distribution.json", report)
    write_json(out / "input_batch_manifest.json", manifest)
    # Small diagnostic descriptors only; never a deployment/training checkpoint.
    torch.save(z, out / "diagnostic_inputs.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
