# Lora-Pyra-Geo

UAV–satellite cross-view geo-localization with a three-stage research pipeline:
**DINOv3 ViT-7B Teacher → DINOv3 ViT-B Middle Teacher → RepViT-M1.5 Student**.

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
  middle_teacher/      Middle E3 model, HRD/ABV2 and KD-guided SAM
  student/             RepViT S3 and TRAIN-only subspace tools
  evaluation/          Unified evaluator, strict loaders and metric certification
  models/              Backbone adapters and tuning policies
  data/                Middle paired-data pipeline
  dataset/             Teacher/Student data and benchmark loaders
  utils/               Shared metrics, distributed extraction and run utilities
configs/
  teacher/             Teacher protocol reference JSONs
  middle_teacher/      Formal E3 configurations
  student/             Formal S3 configurations
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
S3 also requires a validated E3 Middle checkpoint, config, and SHA256-matched
TRAIN-only Top128 and calibration assets. Missing assets fail closed. No weights,
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
Formal training uses `configs/teacher/t0_certified_224.json` or
`configs/teacher/t0_certified_256.json` through `--config`.

## Middle Teacher

Final entry: `src.middle_teacher.fchain_train`; use
`scripts/train_middle_teacher.sh` with the matching R224/R256 E3 config.

## Student

Final entry: `src.student.launch`; use `scripts/train_student.sh` with the
matching R224/R256 S3 config. Build Top128 and calibration with
`scripts/build_student_assets.sh`. Detailed reproduction documentation is
scheduled for Phase 5C.

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
