import os
from datetime import datetime


def get_save_pth(args):
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        return output_dir

    run_timestamp = getattr(args, "run_timestamp", None)
    if not run_timestamp:
        run_timestamp = datetime.now().strftime("%m%d_%H%M")
        setattr(args, "run_timestamp", run_timestamp)

    return os.path.join(
        getattr(args, "output_root", "src/checkpoint/teacher"),
        run_timestamp,
    )


def get_student_save_pth(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(
        getattr(args, "output_root", "src/checkpoint/student"),
        timestamp,
    )
