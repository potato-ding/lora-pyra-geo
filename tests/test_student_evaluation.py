import json
from types import SimpleNamespace

from src.training.student_test import write_results


def _args(tmp_path, dataset, output_json=None):
    return SimpleNamespace(
        checkpoint=str(tmp_path / "best_model.pth"),
        dataset=dataset,
        img_size=224,
        batch_size=32,
        output_json=output_json,
        gta_split="cross-area",
        gta_query_mode="both",
        sues_height="all",
        sues_horizontal_flip=False,
    )


def test_student_results_use_dataset_specific_default_filename(tmp_path):
    args = _args(tmp_path, "1652")
    results = {"1652": {"D2S": {"R@1": 42.0}}}

    write_results(args, results)

    output_path = tmp_path / "student_test_1652.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["checkpoint"] == args.checkpoint
    assert payload["dataset"] == "1652"
    assert payload["results"] == results


def test_student_results_honor_explicit_output_path_and_dataset_options(tmp_path):
    output_path = tmp_path / "reports" / "gta.json"
    args = _args(tmp_path, "GTA-UAV", str(output_path))
    results = {"GTA-UAV": {"D2S": {"R@1": 31.0}}}

    write_results(args, results)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["gta_split"] == "cross-area"
    assert payload["gta_query_mode"] == "both"
    assert payload["results"] == results
