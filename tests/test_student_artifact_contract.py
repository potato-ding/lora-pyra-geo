"""Contract tests use synthetic metadata/byte fixtures, never model evaluation."""
import json
import subprocess
import sys
import tarfile
from pathlib import Path
import pytest
from src.student import artifacts as a
from src.student.evaluate_best import publish_result

def metrics(score):
    return {"D2S":{"R@1":score,"R@5":score+1,"AP":score-.5},
            "S2D":{"R@1":score,"R@5":score+1,"AP":score-.5}}

def fixture_run(tmp_path):
    run=tmp_path/"SYNTHETIC_CONTRACT_FIXTURE";run.mkdir()
    cfg=dict(epochs=30,method="baseline",seed=0,git_commit="fixture",
        student_pretrained_path="fixture-only",student_pretrained_sha256="fixture")
    a.write_json(run/"run_config.json",cfg)
    history=[dict(epoch=i,metrics=metrics(80 if i>=9 else i)) for i in range(1,31)]
    a.write_json(run/"epoch_metrics.json",history)
    a.write_json(run/"best_metrics.json",a.best_record(9,history[8]["metrics"]))
    for name in ("best_model.pth","last_model.pth","train.log"):
        (run/name).write_bytes(b"SYNTHETIC TEST BYTES, NOT A MODEL")
    return run,history

def test_selection_requires_complete_history_and_first_strict_maximum(tmp_path):
    run,history=fixture_run(tmp_path)
    a.validate_training_complete(run)
    a.write_json(run/"best_metrics.json",a.best_record(10,history[9]["metrics"]))
    with pytest.raises(ValueError):a.validate_training_complete(run)
    a.write_json(run/"best_metrics.json",a.best_record(9,history[8]["metrics"]))
    a.write_json(run/"epoch_metrics.json",history[:-1])
    with pytest.raises(ValueError):a.validate_training_complete(run)

def test_slim_archive_allowlist_and_actual_external_hash(tmp_path):
    run,history=fixture_run(tmp_path)
    sha=a.file_sha256(run/"best_model.pth")
    for name in a.RESULT_FILES:
        a.write_json(run/name,dict(checkpoint_sha256=sha))
    result=a.package_results(run)
    archive=Path(result["PACKAGE"])
    assert a.file_sha256(archive)==result["PACKAGE_SHA256"]
    assert archive.stat().st_size==result["PACKAGE_SIZE"]
    with tarfile.open(archive) as f:
        assert set(f.getnames())==set(a.SLIM_FILES+("RESULT_MANIFEST.txt",))
        assert not any(n.endswith((".pth",".pt")) for n in f.getnames())
        embedded=json.load(f.extractfile("RESULT_MANIFEST.txt"))
        assert embedded["PACKAGE_SHA256"]=="RECORDED_IN_EXTERNAL_FINAL_MANIFEST"
    with pytest.raises(FileExistsError):a.package_results(run)

def test_result_identity_and_no_overwrite(tmp_path):
    run,_=fixture_run(tmp_path)
    payload=dict(model_type="student",checkpoint=str(run/"last_model.pth"),u1652_eval_batch_size=32)
    with pytest.raises(ValueError):publish_result(run,"u1652",payload,"fixture")
    payload["checkpoint"]=str(run/"best_model.pth")
    target=publish_result(run,"u1652",payload,"fixture")
    assert target.name=="test_1652_best.json"
    with pytest.raises(FileExistsError):publish_result(run,"u1652",payload,"fixture")

def test_launcher_tees_stderr_and_preserves_nonempty_run(tmp_path,monkeypatch,capfdbinary):
    from src.student import launch
    run=tmp_path/"reserved"
    monkeypatch.setattr(launch,"load_config",lambda _:dict(output_dir=str(run)))
    original=subprocess.Popen
    def fake_child(command,**kwargs):
        return original([sys.executable,"-c",
            "import sys; print('stdout-line'); print('stderr-line',file=sys.stderr); sys.exit(7)"],**kwargs)
    monkeypatch.setattr(launch.subprocess,"Popen",fake_child)
    with pytest.raises(SystemExit) as e:launch.main(["--config","fixture.json"])
    assert e.value.code==7
    data=(run/"train.log").read_bytes()
    assert b"stdout-line" in data and b"stderr-line" in data
    with pytest.raises(FileExistsError):launch.main(["--config","fixture.json"])
    assert (run/"train.log").read_bytes()==data


def test_approved_selection_metadata_and_short_warmup():
    import torch
    from types import SimpleNamespace
    from src.student.scheduler import build_student_scheduler
    contract=a.selection_metadata()
    assert contract == dict(checkpoint_selection_dataset="University-1652",
        checkpoint_selection_split="test",checkpoint_selection_metric="D2S_R1 + S2D_R1",
        checkpoint_selection_rule="strict_greater_than",checkpoint_selection_frequency="every_epoch",
        SUES_USED_FOR_SELECTION=False,GTA_USED_FOR_SELECTION=False)
    best=a.best_record(1,metrics(80))
    assert all(best[k]==v for k,v in contract.items())
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))],lr=1e-4)
    scheduler=build_student_scheduler(optimizer,SimpleNamespace(epochs=30,warmup_epochs=.1,min_lr_ratio=.01),steps_per_epoch=1182)
    fn=scheduler.lr_lambdas[0]
    assert fn(0)==pytest.approx(1/118)
    assert fn(117)==pytest.approx(1)
    assert fn(118)==pytest.approx(1)
    assert fn(30*1182)==pytest.approx(.01)
