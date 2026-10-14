"""Split routing and official GTA query/gallery semantics, without model training."""
import json
from types import SimpleNamespace
import pytest
import torch
from src.evaluation.protocols import GTA_PROTOCOL, gta_eval_splits, gta_protocol, gta_result_name
from src.dataset.teacher.val_dataloaders import build_gta_val_dataloaders


@pytest.mark.parametrize("split", ["cross-area", "same-area"])
def test_gta_split_reads_its_own_queries_and_full_gallery(tmp_path, split):
    satellite = tmp_path / "satellite"
    satellite.mkdir()
    for name in ["4_0_0_0.png", "4_0_0_1.png", "4_0_0_2.png"]:
        (satellite / name).touch()
    def entry(name, positives):
        return dict(drone_img_dir="drone/images", drone_img_name=name,
                    drone_loc_x_y=[1., 2.], pair_pos_sate_img_list=positives)
    for selected in ["cross-area", "same-area"]:
        rows = [entry(selected + ".png", ["4_0_0_1.png"]), entry("no-positive.png", [])]
        (tmp_path / (selected + "-drone2sate-test.json")).write_text(json.dumps(rows))
    query, gallery = build_gta_val_dataloaders([256,256], str(tmp_path), split_type=split,
                                             batch_size=16, num_workers=0)["D2S"]
    assert len(query.dataset) == 1 and len(gallery.dataset) == 3
    assert query.dataset.img_paths[0].endswith(split + ".png")
    positive_id = gallery.dataset.img_paths.index(str(satellite / "4_0_0_1.png"))
    assert query.dataset.labels == [[positive_id]]
    assert query.dataset.protocol_audit["gta_protocol"] == gta_protocol(split)
    assert query.dataset.protocol_audit["raw_json_queries"] == 2
    assert query.batch_size == gallery.batch_size == 16


def test_split_identity_keeps_cross_area_storage_unchanged():
    assert gta_eval_splits("both") == ("cross-area", "same-area")
    assert gta_protocol("cross-area") == GTA_PROTOCOL
    assert gta_result_name("cross-area") == "test_gta_cross_area_d2s.json"
    assert gta_result_name("same-area") == "test_gta_same_area_d2s.json"
    with pytest.raises(ValueError):
        gta_eval_splits("cross-view")


@pytest.mark.parametrize("selection", ["both", "same-area"])
def test_formal_entry_writes_separate_results_and_preserves_cross_area(tmp_path, monkeypatch, selection):
    from src.evaluation import evaluate as module
    from src.student import artifacts
    from src.evaluation.protocols import RETRIEVAL_PROTOCOL
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.touch()
    (tmp_path / "test_1652.json").write_text("{}")
    cross = tmp_path / gta_result_name("cross-area")
    if selection == "same-area":
        cross.write_text("existing cross-area result")
    monkeypatch.setattr(artifacts, "require_valid_run", lambda path: None)
    monkeypatch.setattr(module, "verify_previous_u1652", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"selection_dataset_protocol": RETRIEVAL_PROTOCOL})
    audit = dict(sha256="checkpoint-sha", precision_signature=dict(image_size=256),
                 runtime_precision=dict(train_selection_signature_verified=True),
                 artifact_classification="FORMAL_STUDENT_CHECKPOINT")
    monkeypatch.setattr(module, "load_encoder", lambda *a, **k: (SimpleNamespace(descriptor_dim=512), audit))
    seen = []
    def loaders(**kwargs):
        split = kwargs["split_type"]
        seen.append(split)
        return {"D2S": (SimpleNamespace(dataset=SimpleNamespace(protocol_audit={"gta_protocol":gta_protocol(split)})), object())}
    monkeypatch.setattr(module, "build_gta_val_dataloaders", loaders)
    metrics = {k:1. for k in ("R@1","R@5","R@10","R@top1","AP","SDM@1","SDM@3","SDM@5","DIS@1","DIS@3","DIS@5")}
    monkeypatch.setattr(module, "run_gta_val_and_get_metrics", lambda *a: metrics)
    module.main(["--model-type","student","--checkpoint",str(checkpoint),"--image-size","256",
                 "--dataset","gta","--gta-split",selection])
    assert seen == list(gta_eval_splits(selection))
    for split in seen:
        result = json.loads((tmp_path / gta_result_name(split)).read_text())
        assert result["dataset"] == "gta" and result["protocol"]["split"] == split
        assert result["checkpoint_sha256"] == "checkpoint-sha"
        assert result["results"]["D2S"] == metrics
    if selection == "same-area":
        assert cross.read_text() == "existing cross-area result"
    with pytest.raises(FileExistsError):
        module.main(["--model-type","student","--checkpoint",str(checkpoint),"--image-size","256",
                     "--dataset","gta","--gta-split",selection])


@pytest.mark.parametrize("selection", ["both", "same-area"])
def test_comparison_entry_routes_both_splits_without_overwriting(tmp_path, monkeypatch, selection):
    import sys
    from src.evaluation import comparison as module
    name = next(iter(module.SPECS))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_comparison", lambda *args: (object(), {"sha256":"test-sha"}, 256))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sys, "argv", ["comparison", "--model",name,"--dataset","gta","--gta-split",selection])
    out = tmp_path / "src/checkpoint/others" / module.SPECS[name][0] / "results/fixed_protocol"
    out.mkdir(parents=True)
    if selection == "same-area":
        (out / "gta.json").write_text("existing cross-area result")
    seen = []
    def loaders(**kwargs):
        split = kwargs["split_type"]
        seen.append(split)
        return {"D2S": (SimpleNamespace(dataset=SimpleNamespace(protocol_audit={"gta_protocol":gta_protocol(split)})), object())}
    monkeypatch.setattr(module, "build_gta_val_dataloaders", loaders)
    monkeypatch.setattr(module, "run_gta_val_and_get_metrics", lambda *a: {"R@1":50.})
    module.main()
    assert seen == list(gta_eval_splits(selection))
    result = json.loads((out / "gta_same_area.json").read_text())
    assert result["protocol"]["split"] == "same-area"
    assert result["results"]["D2S"]["R@1"] == 50.
    if selection == "same-area":
        assert (out / "gta.json").read_text() == "existing cross-area result"
    else:
        assert json.loads((out / "gta.json").read_text())["protocol"]["split"] == "cross-area"
