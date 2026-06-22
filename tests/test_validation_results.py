import json

from src.utils.validation_results import save_validation_results


def test_save_validation_results_keeps_aggregate_and_per_epoch_files(tmp_path):
    history = [
        {
            "epoch": 5,
            "R@1_sum": 101.5,
            "D2S": {"R@1": 50.0},
            "S2D": {"R@1": 51.5},
            "is_best": True,
        },
        {
            "epoch": 10,
            "R@1_sum": 103.0,
            "D2S": {"R@1": 51.0},
            "S2D": {"R@1": 52.0},
            "is_best": True,
        },
    ]

    save_validation_results(str(tmp_path), history)

    aggregate_path = tmp_path / "validation_results.json"
    epoch_path = tmp_path / "validation_results" / "epoch_0010.json"
    assert aggregate_path.is_file()
    assert epoch_path.is_file()

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    epoch_result = json.loads(epoch_path.read_text(encoding="utf-8"))
    assert aggregate["latest_epoch"] == 10
    assert aggregate["validation_results"] == history
    assert epoch_result == history[-1]


def test_save_validation_results_does_nothing_for_empty_history(tmp_path):
    save_validation_results(str(tmp_path), [])

    assert not (tmp_path / "validation_results.json").exists()
