"""Experiment-local terminal log capture helpers."""

import atexit
import os
import sys
from datetime import datetime

import torch.distributed as dist


DEFAULT_TRAIN_LOG_FILENAME = "train.log"


class TeeStream:
    """Write text to the original terminal stream and a log file."""

    def __init__(self, terminal_stream, log_file):
        self._terminal_stream = terminal_stream
        self._log_file = log_file

    def write(self, text):
        written = self._terminal_stream.write(text)
        self._log_file.write(text)
        return written

    def flush(self):
        self._terminal_stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._terminal_stream.isatty()

    def __getattr__(self, name):
        return getattr(self._terminal_stream, name)


class RunLogCapture:
    """Own the temporary stdout/stderr tee for one training process."""

    def __init__(self, log_path, log_file, stdout, stderr):
        self.log_path = log_path
        self._log_file = log_file
        self._stdout = stdout
        self._stderr = stderr
        self._tee_stdout = TeeStream(stdout, log_file)
        self._tee_stderr = TeeStream(stderr, log_file)
        self._closed = False

    def install(self):
        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._tee_stdout.flush()
        if sys.stdout is self._tee_stdout:
            sys.stdout = self._stdout
        if sys.stderr is self._tee_stderr:
            sys.stderr = self._stderr
        self._log_file.close()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def resolve_shared_output_dir(args, default_dir_factory, is_main_process):
    """Resolve one experiment directory and broadcast it to every rank."""

    output_dir = None
    if is_main_process:
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            output_dir = default_dir_factory(args)
        os.makedirs(output_dir, exist_ok=True)

    if is_distributed():
        shared_output_dir = [output_dir]
        dist.broadcast_object_list(shared_output_dir, src=0)
        output_dir = shared_output_dir[0]

    if not isinstance(output_dir, str) or not output_dir:
        raise RuntimeError("Unable to resolve a shared training output directory")

    args.output_dir = output_dir
    return output_dir


def setup_rank0_run_log(output_dir, is_main_process, filename=DEFAULT_TRAIN_LOG_FILENAME):
    """Mirror rank-zero stdout and stderr to an experiment-local log file."""

    if not is_main_process:
        return None

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    capture = RunLogCapture(log_path, log_file, sys.stdout, sys.stderr)
    capture.install()
    atexit.register(capture.close)

    print("=" * 80)
    print(f"[RunLog] started_at={datetime.now().isoformat(timespec='seconds')}")
    print(f"[RunLog] path={log_path}")
    print(f"[RunLog] command={' '.join(sys.argv)}")
    print("=" * 80)
    return capture


__all__ = [
    "DEFAULT_TRAIN_LOG_FILENAME",
    "RunLogCapture",
    "TeeStream",
    "resolve_shared_output_dir",
    "setup_rank0_run_log",
]
