# Lora-Pyra-Geo

UAV–satellite cross-view geo-localization with a three-stage research pipeline:
**DINOv3 ViT-7B Teacher → DINOv3 ViT-B Middle Teacher → RepViT-M1.5 Student**.

The repository contains the current implementations used to evaluate different
components of the Teacher-to-Middle pipeline, including SRMD, as well as the
Dual-STST Middle-to-Student interface. The R224 chain is undergoing renewed
component-necessity validation: a historical SRMD combination is not claimed to
be the final paper configuration.

## Method and objectives

Teacher task adaptation → Middle knowledge transfer → residual-aware Middle
optimization → Dual-STST Student distillation.

The retrieval baseline is **PairInfoNCE**, paired UAV/satellite cross-view
contrastive learning. Teacher, Middle and Student use this base objective;
distillation components contribute additional losses. Source and explicit run
configurations define component equations and weights.

## Repository structure

```text
src/
  training/teacher/     Teacher training, selection wrapper and PairInfoNCE
  middle_teacher/      Middle models, composer, bridges, residual methods, SAM
  student/             RepViT baseline, Dual-STST and TRAIN-only subspace tools
  evaluation/          Unified evaluator, strict loaders and metric certification
  models/              Backbone adapters and tuning policies
  data/                Middle paired-data pipeline
  dataset/             Teacher/Student data and benchmark loaders
  utils/               Shared metrics, distributed extraction and run utilities
configs/
  teacher/             Teacher protocol reference JSONs
  middle_teacher/      Seventeen retained component recipes
  student/             Baseline and Dual-STST configurations
scripts/               Launch wrappers and result summarization
tests/                 CPU-compatible unit and contract checks
analysis/              Compact historical evidence, not model assets
```

## Environment and external assets

Recorded runtime: Python 3.10, PyTorch 2.5.1+cu121,
torchvision 0.20.1+cu121, DeepSpeed 0.19.0. From the repository root:

```bash
conda env create -f environment.yml
conda activate pyra_geo
# Alternatively, in a Python 3.10 environment:
python -m pip install -r requirements.txt
```

DeepSpeed supplies the training runtime; Student uses torchrun to launch it.
A compatible NVIDIA driver and an appropriately configured CUDA toolkit/compiler
are external prerequisites for CUDA/DeepSpeed extension builds. These files do
not install a system driver or change system CUDA. Preserve the validated CUDA
environment when reproducing a run. Import/unit checks in the existing environment
do not certify a fresh installation.

**DINOv3 upstream source is not vendored.** Obtain its licensed source separately
and place it at `src/models/dinov3_main/`, containing `dinov3/hub/backbones.py`.
Keep the upstream version used by the run. The current local source snapshot has
no recorded upstream Git commit; exact upstream provenance remains a portability
limitation. Supply official pretrained files under `src/models/dinov3-pth/`:

```text
dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

Student requires a separately obtained RepViT-M1.5 pretrained checkpoint.
Dual-STST also requires a strict-compatible Middle checkpoint, config, and a
SHA256-matched TRAIN-only 32D subspace bank. Missing assets fail closed. No weights,
bank, dataset or upstream source download is performed by these examples.

## Datasets

Prepare datasets independently, following their licenses and official splits.
Default evaluator layout:

```text
data/
  U1652/                       train/ and test/ view directories
  SUES-200/SUES-200-512x512/     Testing/ with all four heights
  GTA-UAV-LR/GTA-UAV-LR-baidu/  official cross-area metadata and images
```

Use `--u1652-dir`, `--sues200-dir` and `--gta-dir` to override evaluation paths.
See `src/dataset/teacher/val_dataloaders.py` for exact image/metadata layouts.
Datasets are not included in Git.

## Teacher training

Entry: `src.training.teacher.train`. Blocks are zero-based, with half-open tuning
ranges: **0–19 frozen, 20–35 LoRA, 36–39 full fine-tuning**.
Teacher protocol JSONs are references, not a `--config` interface; the trainer
accepts CLI arguments and a separate user-prepared DeepSpeed JSON.

```bash
python -m src.training.teacher.train --help
deepspeed --include localhost:0,1,2,3,4,5,6,7 --module src.training.teacher.train \
  --experiment_id T0-CERTIFIED-R224-S0 \
  --data_dir data/U1652 --img_size 224 --epochs 10 --seed 0 --batch_size 4 \
  --training_stage paired_cross_view \
  --lora_start_block 20 --lora_end_block 36 \
  --full_finetune_start_block 36 --full_finetune_end_block 40 \
  --infonce_weight 1 --triplet_weight 0 --same_domain_triplet_weight 0 \
  --lr 0.0001 --scheduler cosine --warmup_ratio 0.05 \
  --deepspeed_config assets/teacher_deepspeed.json \
  --output_dir src/checkpoint/teacher/T0-CERTIFIED-R224-S0
```

Prepare `assets/teacher_deepspeed.json` for the intended protocol before launch:
8 ranks × 4 local pairs = 32 global pairs, accumulation 1, BF16. This file is not
shipped; verify optimizer/offload/runtime settings against the chosen run.
The named R224 certified experiment uses the canonical training selection wrapper:
shared final encoder/metric core and globally contiguous extraction groups of 8.
Other Teacher experiment IDs retain their existing selection path; do not assume
this wrapper is enabled for arbitrary renamed experiments.

## Middle Teacher

Entry: `src.middle_teacher.train`; recipes: `configs/middle_teacher/`.
Retained implementations include R0 partial/full trainability, NRKD, Margin,
AdaptiveBridge V1/V2, RDD, SAM, LC-RD, RE-Gated LC-RD, RMD and SRMD.

```bash
python -m src.middle_teacher.train --help
deepspeed --include localhost:0,1 --no_local_rank --module src.middle_teacher.train \
  --config configs/middle_teacher/srmd_s1.json \
  --teacher-run assets/teacher_run \
  --teacher-checkpoint assets/teacher_run/best_model.pth \
  --val-data-dir data/U1652
```

Configure initialization, data and a new output path in a copy of the JSON before
launching. Run from the repository root. `scripts/train_middle_teacher.sh` uses
the active environment's DeepSpeed and accepts config followed by trainer arguments;
the explicit command above controls GPU inclusion.

**Current limitation:** non-SAM recipes are retained for component/configuration
reproducibility, but the clean trainer explicitly rejects its non-SAM update loop.
They are not advertised as launch-ready end-to-end training recipes. The SAM path
exists, but this documentation audit does not certify a new full training run.
`--build-only` constructs the model without loading foundation weights; it is not
a complete runtime preflight.

## Student

Entry: `src.student.train`. Edit a copy of `configs/student/baseline.json` or
`configs/student/dual_stst.json` to supply assets and a fresh output directory.

```bash
python -m src.student.train --config configs/student/baseline.json --validate-only
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m src.student.train --config configs/student/baseline.json
# Dual-STST uses the same entry with configs/student/dual_stst.json.
```

The checked config is 2 GPUs, 16 pairs/GPU, 30 epochs, DeepSpeed ZeRO-1 BF16.
Dual-STST accepts `middle_checkpoint`, `middle_config` and `stst_asset` in JSON;
no historical Middle run is hard-coded. `src.student.subspace` provides the
TRAIN-only centroid/SVD/seeded-QR construction; no bank is built implicitly.
Interface/unit coverage is not a claim of completed Student benchmark results.

## Unified formal evaluation

The public entry is **`python -m src.evaluation.evaluate`**, for
`--model-type teacher|middle|student` and `--dataset u1652|sues200|gta|all`.
Middle evaluation also needs its matching `--config`.

```bash
python -m src.evaluation.evaluate \
  --model-type teacher --checkpoint assets/teacher_run/best_model.pth \
  --dataset all --data-root data --device cuda --batch-size 8 \
  --output-dir outputs/teacher_evaluation
```

Formal preprocessing is deterministic R224, no augmentation, eval mode and no
gradients. Descriptors are FP32 L2-normalized; similarity is `q @ g.T` with
official-compatible NumPy descending ordering. Fix extraction batch size as part
of the protocol: BF16 descriptors may depend on batch composition. Current R224
Teacher consistency was checked at batch size 8.

| Dataset | Protocol | Metrics |
|---|---|---|
| University-1652 | D2S and S2D, retain distractors | R@1, R@5, AP |
| SUES-200 | 150/200/250/300 m, D2S and S2D | R@1, AP |
| GTA-UAV | cross-area, D2S only | R@1, AP, DIS@1 (m), SDM@3 (%) |

**Training and formal benchmark evaluation are separate stages.** U1652
`D2S_R1 + S2D_R1` selects checkpoints, updating only on strict improvement (ties
keep the earlier checkpoint). SUES/GTA never enter checkpoint selection.
The CLI writes `test_1652.json`, `test_sues200.json` and
`test_gta_cross_area_d2s.json`; it does not itself rename files or package a run.

## Experiment assets and reproducibility

The local formal run handoff convention is:

```text
best_model.pth
best_metrics.json
train.log
test_1652_best.json
test_sues200_all_best.json
test_gta_cross_area_d2s_best.json
RESULT_MANIFEST.txt
<RUN_NAME>_RESULTS.tar.gz
```

Trainer-native names vary (Student also writes `run_config.json` and
`epoch_metrics.jsonl`). Collect and rename completed outputs explicitly for this
convention, recording provenance; do not overwrite historical results. Results
archives exclude weights. Checkpoint assets remain local and excluded from Git.
`analysis/` is historical evidence, not new benchmark certification. Publication
copies use repository-relative or `<LOCAL_ASSET_ROOT>` provenance paths; original
source hashes are retained and are not hashes of sanitized publication copies.

Record fixed seeds, explicit image size, resolved configs, global pair batch,
world size, precision, checkpoint SHA256, selection criterion and evaluator version.
Without per-epoch weights, a current certified evaluation does not retrospectively
certify every historical checkpoint's selection ranking.

```bash
python -m compileall -q src
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests
```

## License and citation

No project-level LICENSE file is currently provided; no blanket license is asserted.
DINOv3 and RepViT retain upstream notices and terms. Model and dataset access is
governed by their respective owners. Paper/citation: under preparation.
