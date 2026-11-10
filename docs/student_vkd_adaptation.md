# VkD feature-distillation adaptation for cross-view retrieval

This is a retrieval adaptation of VkD (CVPR 2024), not a reproduction of its
ImageNet classification training. Official reference files:

- https://github.com/roymiles/VkD/blob/master/train.py (orthogonal projector)
- https://github.com/roymiles/VkD/blob/master/engine.py (feature normalization and SmoothL1)

## Locked protocol and actual objective

Frozen M3-256, RepViT-M1.5 initialized from the same public pretrained weights,
University-1652 TRAIN, 256 input, 32 pairs / one GPU, 30 epochs. Existing
optimizer/scheduler, augmentation, selection epochs 11-30 and single-GPU batch16
U1652 D2S/S2D R1-sum selection remain in use. No external-dataset selection.

Let b_s be the existing shared BN512 output before retrieval L2 normalization,
and z_t the frozen M3 retrieval descriptor. For a minibatch of 64 images:

    W = orthogonal_parameterization(A), W.shape = [768, 512], W.T W = I
    prediction = b_s @ W.T                 # no bias or output L2 normalization
    target = layer_norm(z_t.detach(), [768], eps=1e-5)
    L_VkD = smooth_l1(prediction, target, beta=1, reduction="mean")
    lambda(e) = lambda_max * min(e / 5, 1), epochs numbered from 1
    L_full = L_bidirectional_InfoNCE + lambda(e) * L_VkD

The 64 images contain both views, so elementwise mean weights them equally.
The teacher target is its existing descriptor followed by LayerNorm; unlike
the official classifier, this adapter does not read a CNN spatial feature map.
The Student predictor takes pre-L2 features to avoid the unit-norm versus
LayerNorm target scale incompatibility. Both choices must be stated in papers.
There are no PCA/Top128/Random32 assets, GBW, residual MLP or logits KD.

DG-SAM retains rho=.10, KD-only first pass, full-objective second pass, same
augmented batch, RNG/buffer replay, and exact restore before one optimizer and
scheduler update. SAM acts on the optimizer's underlying parameter coordinates,
including the projector's original Householder coordinates; it does not add an
unconstrained perturbation to the materialized W. Both passes use the same VkD
objective and same frozen teacher target.

## Precision and constrained optimizer parameters

Student stays BF16 with FP32 retrieval descriptors/loss. The entire projector,
including its parametrization base buffer, stays FP32. PyTorch orthogonal uses
Householder with dynamic trivialization (explicitly configured here); the
768x512 matrix has orthonormal columns, not 768 orthonormal rows.

The Householder original parameter has a fixed +/-1 diagonal. The installed
PyTorch forward casts that diagonal to int. AdamW decay would shrink +/-1 and
make this cast zero, destroying orthogonality. Therefore only the constrained
projector coordinate tensor uses an explicit zero-weight-decay group. Student
parameter groups retain lr=1e-4 and weight_decay=1e-4. The projector uses the
same LR and schedule. No extra regularization or post-step projection is added.
Predictor initialization follows the configured training RNG; training seed and
all head parameter/buffer hashes are recorded in initialization_identity.json.
This differs from the historical fixed C2 predictor initialization policy and
must not be described as identical predictor initial weights across methods.

## Entry points

Three predeclared seed0 candidates use weights .5 / 1 / 2:

    configs/student/vkd_sam_m3_w05_256_seed0.json
    configs/student/vkd_sam_m3_w10_256_seed0.json
    configs/student/vkd_sam_m3_w20_256_seed0.json

output_dir is null until an experiment is assigned. No formal run was started
while implementing this adapter. Freeze a weight based only on U1652 selection
before adding seed1/2. vkd_config supports seeds 0/1/2 with strict matching
filenames; only seed0 configs are materialized at this stage.

CPU preflight (no training or output experiment folders):

    conda activate pyra_geo
    python scripts/check_student_vkd.py

Explicit three-logical-batch GPU smoke on a confirmed idle GPU:

    CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=2 python -m torch.distributed.run \
      --standalone --nproc_per_node=1 scripts/check_student_vkd.py --gpu-smoke

The smoke uses real M3 and Student weights, real TRAIN images, BF16 DeepSpeed,
and the same optimizer/scheduler precision grouping. It also tests checkpoint
packing and unified deployment reload with synthetic metric fixtures in a
TemporaryDirectory, deleted on exit. Those fixtures are not experiment results.
No real retrieval evaluation is performed by this preflight.

After assigning a new, empty output_dir, use the existing train_student.sh.
The unified student evaluator works unchanged. Deployment checkpoints contain
only the Student encoder; the strict training_auxiliary holds the FP32 original
projection parameter and its orthogonal base. Wrong inventory, precision,
nonfinite data, invalid Householder diagonal or nonorthogonal base are rejected.

## Implementation verification (2026-10-28)

- Seven dedicated CPU tests cover recipe math, detached teacher, preserved
  deployment forward, hook cleanup, precision, three SAM batches, exact restore,
  optimizer constraints, strict roundtrip and corruption rejection.
- Full suite: 380 passed, 1 skipped (CUDA numerical test hidden in CPU regression).
- Real GPU smoke: three logical batches pass with losses 3.2272358, 3.6319835,
  3.1149628; post-update orthogonality error approximately 1.19e-6.
- Unified deployment reload max diff = 0 on the same real input.
- No formal VkD training, commit or push was performed.
