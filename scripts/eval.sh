#!/usr/bin/env bash
# Run a checkpoint evaluation on one physical GPU.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=""
ARGS=()
while (($#)); do
    case "$1" in
        --gpu)
            if (($# < 2)); then echo "Missing value for --gpu" >&2; exit 2; fi
            GPU="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
cd "$ROOT"
if [[ "$GPU" == "" && "${ARGS[*]}" == *"--help"* ]]; then
    exec python -m src.evaluation.evaluate "${ARGS[@]}"
fi
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    echo "Specify one GPU with --gpu <index>" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU"
exec python -m src.evaluation.evaluate "${ARGS[@]}"
