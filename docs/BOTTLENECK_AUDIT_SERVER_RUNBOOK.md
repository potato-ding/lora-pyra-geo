# Bottleneck Audit Server Runbook

This runbook executes the diagnostics on the Linux training server. Local development does not validate real checkpoints, datasets, CUDA precision, or metric parity.

## 1. Commit locally (manual)

Review first; Codex does not commit or push:

```bash
git status --short
git diff --check
git diff
git add src/models/student_model.py src/diagnostics scripts/bottleneck_audit docs/BOTTLENECK_AUDIT_SERVER_RUNBOOK.md tests
git commit -m "add bottleneck audit diagnostics"
git push origin clear_dev
```

## 2. Synchronize the server

```bash
cd /home/dingyi/lora-pyra-geo
git status --short
git pull --ff-only origin clear_dev
git rev-parse HEAD
```

## 3. Activate the environment

```bash
source /home/dingyi/miniforge3/etc/profile.d/conda.sh
conda activate pyra_geo
which python
python --version
```

## 4. Configure paths and preflight

Defaults come from the formal repository layout. Override only when the server differs:

```bash
export B0_CKPT=src/checkpoint/student/B0-2GPU-3090/best_model.pth
export T0_CKPT=src/checkpoint/teacher/T0-3090/best_model.pth
export U1652_ROOT=data/U1652
export SUES_ROOT=data/SUES-200/SUES-200-512x512
export GTA_ROOT=data/GTA-UAV-LR/GTA-UAV-LR-baidu
export OUTPUT_ROOT=src/diagnostics/results/bottleneck_audit
export PYTHON_BIN=python
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_P1=29501
export MASTER_PORT_P2=29502
export MASTER_PORT_P3=29503

test -s "$B0_CKPT" && test -s "$T0_CKPT"
test -d "$U1652_ROOT" && test -d "$SUES_ROOT" && test -d "$GTA_ROOT"
python -m src.diagnostics.teacher_advantage --help
python -m src.diagnostics.student_representation --help
python -m src.diagnostics.train_frozen_probe --help
nvidia-smi
```

## 5. Dry run

This prints commands and does not run diagnosis or training:

```bash
DRY_RUN=1 bash scripts/bottleneck_audit/run_all_8gpu.sh
```

No `SUCCESS` is created in dry-run mode.

## 6. Start all eight GPUs

Use tmux so an SSH disconnect does not terminate the jobs:

```bash
tmux new -s bottleneck_audit
cd /home/dingyi/lora-pyra-geo
source /home/dingyi/miniforge3/etc/profile.d/conda.sh
conda activate pyra_geo
bash scripts/bottleneck_audit/run_all_8gpu.sh
```

GPU assignment is fixed by default: GPU 0 runs U1652+SUES G1; GPU 1 runs GTA G1 and then G2; GPU 2,3 run P1 on port 29501; GPU 4,5 run P2 on port 29502; GPU 6,7 run P3 on port 29503. Each probe checks its configured port before launching and fails explicitly if the port is occupied.

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t bottleneck_audit
```

## 7. Start one task only

```bash
bash scripts/bottleneck_audit/run_gap_u1652_sues_gpu0.sh
bash scripts/bottleneck_audit/run_gap_gta_gpu1.sh
bash scripts/bottleneck_audit/run_representation_gpu1.sh
bash scripts/bottleneck_audit/run_probe_p1_gpu23.sh
bash scripts/bottleneck_audit/run_probe_p2_gpu45.sh
bash scripts/bottleneck_audit/run_probe_p3_gpu67.sh
```

Environment overrides such as `GAP_GPU=7` or `P1_GPUS=0,1` may be placed before an individual command.

## 8. Inspect logs and GPUs

```bash
tail -f "$OUTPUT_ROOT/logs/gap_u1652_sues.log"
tail -f "$OUTPUT_ROOT/logs/gap_gta.log"
tail -f "$OUTPUT_ROOT/logs/representation.log"
tail -f "$OUTPUT_ROOT/logs/probe_p1.log"
tail -f "$OUTPUT_ROOT/logs/probe_p2.log"
tail -f "$OUTPUT_ROOT/logs/probe_p3.log"
watch -n 2 nvidia-smi
```

Each G1 protocol also has its own `run.log` beside `summary.json` and `representative_queries.csv`.

## 9. Interrupt and rerun

List exact processes, then terminate only the intended PID:

```bash
pgrep -af 'src.diagnostics|scripts/bottleneck_audit'
kill <PID>
```

After an interruption, rerun the corresponding individual script above. A probe rerun starts its formal 30-epoch protocol from epoch 1; preserve or move partial output first if it is needed for debugging. G1 and G2 are deterministic evaluation tasks and can simply be rerun. Never treat partial files as complete.

## 10. Validate completeness

Only this validator may create `FINAL_MANIFEST.txt` and `SUCCESS`:

```bash
bash scripts/bottleneck_audit/validate_results.sh
test -s "$OUTPUT_ROOT/FINAL_MANIFEST.txt"
test -s "$OUTPUT_ROOT/SUCCESS"
```

Any missing/empty formal artifact, failed report build, or failed parity check returns nonzero and prevents `SUCCESS`.

## 11. Package results

```bash
tar -czf bottleneck_audit_results.tar.gz \
  "$OUTPUT_ROOT" \
  src/checkpoint/student/diagnostic_probes/P1-f3-linear \
  src/checkpoint/student/diagnostic_probes/P2-f4-linear \
  src/checkpoint/student/diagnostic_probes/P3-f4-mlp
ls -lh bottleneck_audit_results.tar.gz
```

Download from Windows PowerShell:

```powershell
scp dingyi@192.168.10.68:/home/dingyi/lora-pyra-geo/bottleneck_audit_results.tar.gz .
```
