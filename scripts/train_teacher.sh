#!/usr/bin/env bash
set -euo pipefail
# Supply the audited T0 CLI and its DeepSpeed configuration explicitly.
exec python -m src.training.teacher.train "$@"
