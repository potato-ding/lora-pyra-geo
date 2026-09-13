"""Isolated formal U1652 entry; optional subset is smoke-only and never published."""
import argparse
import hashlib
import json
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-dir")
    args = parser.parse_args()
    if torch.distributed.is_initialized():
        raise RuntimeError("Canonical evaluator must be standalone")
    from src.evaluation import evaluate as unified
    with ExitStack() as stack:
        if args.audit_dir:
            from torch.utils.data import DataLoader, Subset
            from src.dataset.teacher.val_dataloaders import IndexedDataset
            from src.utils.train_eval_utils import extract_features_dist
            audit = Path(args.audit_dir)
            audit.mkdir(parents=True, exist_ok=False)
            original_builder = unified.build_1652_val_dataloaders
            original_metric = unified.getdist_1652_val_and_get_recall
            manifest, descriptors = {}, {}
            def subset_builder(**kwargs):
                full = original_builder(**kwargs)
                identities = set(sorted(set(full["S2D"][0].dataset.sample_ids))[::5][:64])
                pairs = {}
                for direction, loaders in full.items():
                    pairs[direction] = []
                    for index, loader in enumerate(loaders):
                        ds = loader.dataset.dataset
                        cap = 4 if (direction, index) in (("D2S", 0), ("S2D", 1)) else 1
                        counts, selected = {}, []
                        for i, identity in enumerate(ds.sample_ids):
                            if identity in identities and counts.get(identity, 0) < cap:
                                selected.append(i)
                                counts[identity] = counts.get(identity, 0)+1
                        manifest[direction+"_"+str(index)] = [ds.images[i] for i in selected]
                        pairs[direction].append(DataLoader(IndexedDataset(Subset(ds, selected)),
                            batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True))
                (audit/"subset_manifest.json").write_text(json.dumps(manifest, indent=2))
                return pairs
            def capture_metric(model, query, gallery, device):
                key = "D2S" if not descriptors else "S2D"
                with torch.no_grad():
                    q, ql, qv = extract_features_dist(model, query, device)
                    g, gl, gv = extract_features_dist(model, gallery, device)
                    result = original_metric(model, query, gallery, device,
                        precomputed_features=(q, ql, qv, g, gl, gv))
                descriptors[key] = dict(query=q.cpu(), gallery=g.cpu())
                return result
            stack.enter_context(patch.object(unified, "build_1652_val_dataloaders", subset_builder))
            stack.enter_context(patch.object(unified, "getdist_1652_val_and_get_recall", capture_metric))
        unified.main(["--model-type", "student", "--checkpoint", args.checkpoint,
                      "--dataset", "u1652", "--u1652-dir", args.data_dir,
                      "--batch-size", "32", "--num-workers", str(args.num_workers),
                      "--device", args.device, "--output-dir", args.output_dir])
        if args.audit_dir:
            torch.save(descriptors, audit/"descriptors.pt")
            hashes = {d: {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
                          for k,v in values.items()} for d,values in descriptors.items()}
            (audit/"descriptor_hashes.json").write_text(json.dumps(hashes, indent=2))
            result = Path(args.output_dir)/"test_1652.json"
            payload = json.loads(result.read_text())
            payload["protocol"].update(audit_subset_only=True, formal_result=False)
            result.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
