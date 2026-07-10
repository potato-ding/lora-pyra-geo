import io
import sys
from types import SimpleNamespace

from src.utils.run_logging import (
    DEFAULT_TRAIN_LOG_FILENAME,
    TeeStream,
    resolve_shared_output_dir,
    setup_rank0_run_log,
)


def test_tee_stream_mirrors_terminal_and_log_file():
    terminal = io.StringIO()
    log_file = io.StringIO()
    stream = TeeStream(terminal, log_file)

    assert stream.write("hello\n") == len("hello\n")
    stream.flush()

    assert terminal.getvalue() == "hello\n"
    assert log_file.getvalue() == "hello\n"


def test_setup_rank0_run_log_captures_stdout_and_stderr(tmp_path, monkeypatch):
    original_stdout = io.StringIO()
    original_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", original_stdout)
    monkeypatch.setattr(sys, "stderr", original_stderr)

    capture = setup_rank0_run_log(str(tmp_path), is_main_process=True)
    print("train message")
    print("error message", file=sys.stderr)
    capture.close()

    log_text = (tmp_path / DEFAULT_TRAIN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "[RunLog] started_at=" in log_text
    assert "train message" in log_text
    assert "error message" in log_text
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_resolve_shared_output_dir_uses_default_on_single_process(tmp_path):
    args = SimpleNamespace(output_dir=None)
    expected_output_dir = tmp_path / "0709_1643"

    output_dir = resolve_shared_output_dir(
        args,
        lambda _args: str(expected_output_dir),
        is_main_process=True,
    )

    assert output_dir == str(expected_output_dir)
    assert args.output_dir == output_dir
    assert expected_output_dir.is_dir()
