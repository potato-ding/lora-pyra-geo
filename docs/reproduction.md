# Reproducing the progressive chain

[Back to README](../README.md)

This guide describes the implemented methods and executable entrypoints.
Run commands from the repository root after `conda activate pyra_geo`.
The examples use R256; R224 uses the same method with resolution-matched
configs, providers and assets.

## Data layout

Download and prepare the datasets using their authors' instructions. The loaders
expect the following layout (images are grouped by identity where shown):

```text
data/
├── U1652/
│   ├── train/
│   │   ├── drone/<identity>/<image>
│   │   └── satellite/<identity>/<image>
│   └── test/
│       ├── query_drone/<identity>/<image>
│       ├── gallery_satellite/<identity>/<image>
│       ├── query_satellite/<identity>/<image>
│       └── gallery_drone/<identity>/<image>
├── SUES-200/SUES-200-512x512/
│   └── Testing/
│       └── {150,200,250,300}/
│           ├── query_drone/<identity>/<image>
│           ├── gallery_satellite/<identity>/<image>
│           ├── query_satellite/<identity>/<image>
│           └── gallery_drone/<identity>/<image>
└── GTA-UAV-LR/GTA-UAV-LR-baidu/
    ├── drone/images/
    ├── satellite/
    ├── cross-area-drone2sate-test.json
    └── same-area-drone2sate-test.json
```

The braces above abbreviate four real directories, not a literal folder name.
The current SUES loader consumes the prepared `Testing` layout, not the raw
`drone_view_512`/`satellite-view` directories. Preserve the benchmark split.
Other downloaded views and GTA training JSONs may remain on disk but are not
used by this progressive training pipeline.

Formal evaluation accepts `--data-root` or individual `--u1652-dir`,
`--sues200-dir`, and `--gta-dir` overrides. Training uses `data_dir` in each
config. The Top128 builder requires exactly 701 matching training identities.

## Foundation assets

Place the upstream DINOv3 source package at `src/models/dinov3_main/` and
obtain these exact pretrained weight variants:

| Backbone | File |
| --- | --- |
| Teacher | `src/models/dinov3-pth/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` |
| Middle | `src/models/dinov3-pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| Student | `src/models/repvit/repvit_m1_5_distill_450e.pth` |

These are foundation weights, not the trained retrieval checkpoints.
Follow the upstream access terms linked in the README. Record the compatible
DINOv3 source revision alongside the experiment; this repository does not
download licensed model assets automatically.

## Config and run identity

The JSON chooses the method and resolution. The shell script chooses visible
physical GPUs. Before launching:

1. Set a **fresh** `output_dir`.
2. Set the intended provider checkpoint and corresponding config paths.
3. For distillation Students, set `supervision_asset` and `top_calibration`.
4. Keep all resolutions matched along the chain.

The validators recognize method-specific filenames directly inside the stage's
config directory. Do not rename a config arbitrarily or copy it to `/tmp` and
expect the formal launcher to accept it. Existing checked-in paths describe local
runs; replace them for your environment **before** building dependent assets.
Do not edit an upstream config after building assets whose identity includes
its exact bytes. Preserve each run's original config content.

Read-only config checks:

```bash
python - <<'PY'
from src.training.teacher.formal_config import parse_args
parse_args(["--config", "configs/teacher/t0_256.json"])
print("Teacher config PASS")
PY
bash scripts/train_middle_teacher.sh configs/middle_teacher/m3_sam_hrd_sem_256.json --check
bash scripts/train_student.sh configs/student/s3_sam_adsd_256.json --check
```

The Teacher snippet checks config contents only; its training launcher additionally
checks four GPU IDs and refuses a nonempty output directory.
Middle/Student `--check` validates the config template; it is not a GPU smoke test
or a proof that every model asset can be loaded. Training performs additional
asset checks. There is no automatic launch-all-configs command.

## Teacher

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_teacher.sh \
  configs/teacher/t0_256.json
```

Four GPUs, eight image pairs per GPU, 10 epochs, PairInfoNCE.
The zero-based adaptation ranges are frozen `0:20`, LoRA `20:36`,
and full fine-tuning `36:40`. The LoRA and full-fine-tuning ranges do not overlap.
U1652 selection begins after epoch 6 and repeats through epoch 10.

## Middle methods

Each config stem below has both `_224.json` and `_256.json` variants in
`configs/middle_teacher/`.

| ID | Config stem | Objective | KD-guided SAM |
| --- | --- | --- | --- |
| M0 | `m0_infonce` | PairInfoNCE | No |
| M1 | `m1_hrd` | PairInfoNCE + 0.1 HRD | No |
| M2 | `m2_hrd_sem` | PairInfoNCE + 0.1 HRD + 0.05 semantic | No |
| M3 | `m3_sam_hrd_sem` | PairInfoNCE + 0.1 HRD + 0.05 semantic | Yes |
| M4 | `m4_sam_hrd` | PairInfoNCE + 0.1 HRD | Yes |

M0 does not load a distillation Teacher. M1–M4 use a frozen, resolution-matched
T0 checkpoint. The semantic config uses the implemented adaptive bridge,
Teacher layer indices 28/36, and Middle target layer index 10.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_middle_teacher.sh \
  configs/middle_teacher/m3_sam_hrd_sem_256.json
```

Two GPUs, 16 pairs per GPU, cross-GPU gathering enabled, 10 epochs.
M3/M4 use standard L2 SAM with KD-only search, `rho=0.10`,
`adaptive=false`, all-trainable perturbation and a full-objective second pass.
Selection is on one GPU after epochs 6–10.

The R256 negative-selection control is
`m1_hrd_middle_hnd_256.json`: the Middle chooses the hard negatives while
the Teacher still supplies the target. It is a separate experiment, not the
default M1 configuration.

## Build Student supervision assets

Run this step **after** the Middle's best checkpoint is available.
The following example builds a new M3-256 asset directory; change the paths
to your actual run and use an output that does not exist.

```bash
MIDDLE_CKPT="$PWD/src/checkpoint/middle_teacher/R256/M3-256/best_model.pth"
MIDDLE_CONFIG="$PWD/configs/middle_teacher/m3_sam_hrd_sem_256.json"
ASSET_DIR="$PWD/src/checkpoint/student/R256/middle_assets/M3-256-new"

CUDA_VISIBLE_DEVICES=0 bash scripts/build_student_assets.sh top \
  --middle-checkpoint "$MIDDLE_CKPT" \
  --middle-config "$MIDDLE_CONFIG" \
  --image-size 256 \
  --train-root "$PWD/data/U1652/train" \
  --output "$ASSET_DIR/top128" \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 bash scripts/build_student_assets.sh calibration \
  --middle-checkpoint "$MIDDLE_CKPT" \
  --middle-run-config "$MIDDLE_CONFIG" \
  --student-pretrained "$PWD/src/models/repvit/repvit_m1_5_distill_450e.pth" \
  --train-root "$PWD/data/U1652/train" \
  --image-size 256 --seed 0 \
  --asset-output "$ASSET_DIR/top_calibration.pt" \
  --device cuda:0
```

The `--middle-run-config` argument above points to the matching recognized
Middle config in `configs/middle_teacher/`.

Expected asset files:

```text
M3-256-new/
├── top128/
│   ├── manifest.json
│   ├── teacher_mean.pt
│   └── top128_basis.pt
├── top_calibration.pt
└── top_calibration.pt.json
```

Top128 fits a mean and PCA basis from 1,402 training identity centroids
(701 drone + 701 satellite), using the builder's recorded FP64 SVD protocol.
Calibration stores initial Student descriptors from 384 training images per
view for initialization scaling. Neither step uses test images.

Bind these paths in the Student config:

| Field | Value for the example |
| --- | --- |
| `middle_checkpoint` | `$MIDDLE_CKPT` expanded to its real path |
| `middle_config` | `$MIDDLE_CONFIG` expanded to its real path |
| `supervision_asset` | `$ASSET_DIR/top128/manifest.json` expanded |
| `top_calibration` | `$ASSET_DIR/top_calibration.pt` expanded |
| `output_dir` | A new Student run directory |

JSON paths do not expand shell variables. Write the actual path strings.
A new Middle checkpoint or resolution requires matching assets; never reuse
an M3 basis as if it came from M4 or from another checkpoint with the same name.

## Student methods

All formal Students use RepViT-M1.5, one GPU, 32 pairs, 30 epochs and no
cross-GPU gather. U1652 selection runs after epochs 11–30.

| ID | Distillation | Student SAM | Configs |
| --- | --- | --- | --- |
| S0 | None; PairInfoNCE baseline | No | `s0_infonce_{224,256}.json` |
| S1 | TSD: Top128 only | No | `s1_tsd_m3_{224,256}.json`, `s1_tsd_m4_{224,256}.json` |
| S2 | ADSD: Top128 + Random32 + GBW | No | `s2_adsd_m3_{224,256}.json`, `s2_adsd_m4_{224,256}.json` |
| S3 | ADSD: Top128 + Random32 + GBW | Yes | `s3_sam_adsd_{224,256}.json` (M3), `s3_sam_adsd_m4_{224,256}.json` |
| S4 | ADSD with a SAM-trained Middle; retained non-SAM Student alias | No | `s4_adsd_sam_middle_{224,256}.json` |

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_student.sh \
  configs/student/s3_sam_adsd_256.json
```

The base Student learning rate is `1e-4`, weight decay `1e-4`,
cosine scheduling with `0.1` warmup epochs and minimum LR ratio `0.01`.
KD weight is `0.2 × min(1, epoch/5)` for one-based epochs.
GBW uses `w_top = 0.5 + sigmoid(d)` and
`w_random = 1.5 - sigmoid(d)`, initially 1/1 and always summing to 2.
Its optimizer is separate from the Student optimizer.

### Independent Random32 repetitions

`s3_sam_adsd_independent1_m3_256.json` through
`s3_sam_adsd_independent4_m3_256.json` retain training seed 0 and sample a
new Random32 seed at runtime. The replicate number is an experiment label,
not the numeric random seed. Each run records the sampled seed and basis hash in
`subspace_identity.json` and its training checkpoint.

Together with the original `S3-256-M3`, these define five random-subspace runs.
They are not five different global-training-seed runs. Before interpreting their
spread, compare the saved config, initialization and data-order settings as well
as the random-basis identities. The current independent variants explicitly
configure deterministic data handling; the original run must not be assumed
identical in every other random condition solely because its training seed is 0.

`s3_sam_adsd_shared_m3_256.json` is a separate control that loads recorded
subspace tensors. It does not generate an independent random basis.

### Direct Teacher control

`s3_sam_adsd_t0_rmlp_linear_256.json` uses T0-256 directly as the frozen
provider. Its assets are built by `scripts/build_student_assets.sh teacher-top`;
inspect that action's `--help` for the separate Teacher asset contract.

This control retains Top-RMLP/Random-Linear and Student DG-SAM. T0 descriptors
are 4,096-dimensional; Middle descriptors are 768-dimensional. Reusing the
same seed and Gaussian→QR protocol across those dimensions does **not** produce
the same random matrix or the same subspace.

## Saved artifacts and precision

Every stage writes `best_model.pth` and `train.log` under its output directory.
Middle and Student additionally save `config.json`. Student runs record asset
identities; shared/independent subspace controls add initialization and subspace
identity JSONs. The best checkpoint is not necessarily the final epoch, and
a resumable optimizer checkpoint is not implied by `best_model.pth`.

Formal inference uses the recorded BF16 model execution with FP32 descriptor
normalization and similarity computation. Reload restores per-tensor dtypes
and verifies the precision signature. FP32 tensors in a saved file do not mean
the model was evaluated entirely in FP32. Auxiliary Student residual predictors
and balancing parameters have their own FP32 requirements during training.

The v3 source manifest defines the current stage source identities. New S3
runs additionally record `src/student/distill_sam.py`. Older checkpoint
metadata is retained as written, not backfilled with current hashes.
Archive the matching source revision, run config and provider/asset hashes.

## Practical checks

| Symptom | Check before retrying |
| --- | --- |
| Unknown config or method | Use a recognized filename in the stage config directory |
| Nonempty output directory | Select a fresh directory; do not erase another run |
| Provider/asset SHA mismatch | Verify the exact checkpoint and config used to build the asset |
| Image-size mismatch | Match T/M/S resolution and regenerate the corresponding assets |
| CPUAdam build failure | Check compiler, CUDA toolkit and environment activation |
| No best checkpoint in early epochs | Selection begins at epoch 6 for T/M and epoch 11 for S |
| U1652 reload consistency failure | Preserve the checkpoint; inspect dtype, preprocessing and batch-16 agreement |

These checks do not bypass a failed contract or modify an existing experiment.
