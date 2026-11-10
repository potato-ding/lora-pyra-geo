# R256 Student component ablations (B2--B7)

All six use frozen M3-256, RepViT-M1.5, online targets, DG-SAM rho=0.1,
KD-only first pass and full Task+KD second pass. Base training: 30 epochs,
seed 0, one GPU, pair batch 32, no gather, AdamW lr=1e-4 / wd=1e-4,
cosine schedule, lr warmup 0.1 epoch, KD weight 0.2 with 5-epoch ramp.
U1652 selection is after epochs 11--30, single GPU, batch 16.

| Config stem | Target/predictor | Allocation |
|---|---|---|
| b2_full768_sam_m3_256 | normalized full Middle descriptor; Linear 512->768 | single coefficient 1 |
| b3_top128_sam_m3_256 | centered Top128; RMLP 512->920->128 | single coefficient 1 |
| b4_random32_sam_m3_256 | centered Random32; Linear 512->32 | single coefficient 1 |
| b5_dual_linear_sam_m3_256 | Top128 Linear + Random32 Linear | learnable bounded GBW |
| b6_dual_rmlp_sam_m3_256 | Top128 RMLP + Random32 RMLP (hidden 920) | learnable bounded GBW |
| b7_fixed_equal_sam_m3_256 | Top128 RMLP + Random32 Linear | fixed 1+1; no gate optimizer |

Each branch loss is the equal average of drone/satellite cosine distances.
Single branches are not multiplied by 2 after deletion of the other branch.
B2 uses the full normalized Middle descriptor directly, without PCA centering.
B7 preserves the total branch-weight mass 2 of the final GBW method.
RMLPs use a linear skip plus alpha=0.001 times a once-calibrated residual.
Both residual widths are 920, including the Random32 residual in B6.

All configs strictly bind the same original S3-256-M3 reference checkpoint,
SHA and exact teacher_mean/Top128/Random32 tensors. No trained predictor weights
are reused. Random32 generation count must be zero. B2/B3 keep the reference
buffers only for provenance: their KD never uses the Random target; B2 does
not use either subspace target. Inactive predictors are absent, not optimized.
B3/B4/B2/B7 do not construct a learnable allocation gate.

CPU preflight (no training, no output experiment directory created):

```bash
conda activate pyra_geo
python scripts/check_student_ablations.py
```

When a GPU is available and training is explicitly requested:

```bash
# Replace <gpu> and select one of the six configs above.
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/train_student.sh configs/student/b2_full768_sam_m3_256.json
```

Run through tmux for persistent sessions. No automatic queue is installed.
Existing nonempty output directories are rejected. Each run writes config.json,
asset_identity.json, subspace_identity.json, initialization_identity.json,
train.log and selected best_model.pth. Unified evaluation continues to use
model-type=student; deployment has only the original 512D Student encoder.

A shared subspace does not prove bit-identical historical augmentation order.
The controls share the current explicit seed/worker protocol with one another.
CPU synthetic SAM tests do not substitute for actual BF16/DeepSpeed GPU smoke.
The first three real batches print variant-specific SAM audits at launch.


## Added controls B1 and B8

- `configs/student/b1_m0_provider_sam_m0_256.json`: M0-INFONCE-R256 supplies
  its own online descriptors and its own canonical TRAIN mean/Top128. Student
  ADSD/GBW/predictors and DG-SAM are the final settings. Random32 is the exact
  768x32 matrix stored in S3-256-M3, not a regenerated seed equivalent.
- `configs/student/b8_no_sam_m3_256.json`: final M3 ADSD/GBW with the exact
  final S3 mean/Top128/Random32; SAM is disabled. There is one forward/backward
  and one main optimizer/scheduler step per logical batch, plus the original
  independent gate update. There is no search pass or perturbation.

B1 currently requires new M0 assets. No M3 Top fallback is allowed. Only the
explicit `--allow-m0-ablation` builder path admits M0; existing final methods
keep their provider restrictions. After the first four runs AND their tests
finish, assign a free GPU and activate pyra_geo, then prepare B1 assets:

```bash
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/prepare_b1_m0_assets.sh
```

This command only builds/validates assets; it does not train. It refuses to
replace existing partial/corrupt output. Then select B1/B6/B7/B8 with the normal
train_student.sh launcher in tmux. Do not restart B2--B5. This preparation adds
no polling or automatic launch queue.

CPU-only readiness report:

```bash
python scripts/check_additional_student_ablations.py
```

Before M0 extraction the expected B1 status is ASSET_BUILD_PENDING; do not
interpret this as GPU-ready or fill its Top hashes with M3 values. B8 can be
validated completely using existing assets. Actual GPU BF16/DeepSpeed startup
validation remains necessary for the new controls when runs are authorized.
