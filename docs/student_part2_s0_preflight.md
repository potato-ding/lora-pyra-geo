# Part-II S0 formal integration preflight

Reference: P1-T128-R32-S0, sealed source 58921e3683b37f478f7eefbca6ec367005edb74a.
Frozen component source: baf1f3250b10d1b2ee9317c9b78e6eff50b879d6.
Only P2-TOP-RMLP-S0 and P2-TOP-RKAN-S0 are accepted, seed0 only.
Reference input config matches the actual reference run_config field-for-field.
New configs differ only in run identity/output, dedicated source-seal location,
Top interface selector and the immutable calibration path/SHA.
Teacher/pretrained/Top128/Random32_A SHAs match the reference.

Both reference and P2 use scripts/train_student_certified.sh -> student.launch
-> single-process torchrun -> student.train -> existing DeepSpeed BF16 ZeRO1.
DeepSpeed config, accumulation, AdamW/scheduler, InfoNCE, KD, Random, augmentation,
canonical selector and evaluator are unchanged. P1 dispatch is byte-compatible
after removing explicit P2 construction/metadata/logging hooks.

Precision integration: the initial probe showed mixed BF16/FP32 parameters in
one AdamW group caused DeepSpeed flatten to promote Student to FP32. P2 groups
are therefore partitioned by dtype before the unchanged DeepSpeed initializer.
This reuses the same AdamW object and preserves each parameter's LR, decay,
betas, eps and scheduler. Alpha is scalar no-decay (weight_decay=0).
A two-step direct AdamW regression verifies identical parameter updates and LR
with/without dtype subdivision. Real DeepSpeed smoke verifies the actual
Student/base/Random BF16 and residual/gate FP32 storage both before and after step.
No old P1 optimizer group or source is changed.

Both one-step real TRAIN smokes ran the same loader, Teacher, Student, Top KD,
Random KD, InfoNCE, backward and optimizer step. All required groups received
finite nonzero gradients and changed parameters. Teacher frozen/grad0;
FP32 loss/projection; no NaN/Inf; peak allocated memory approximately 5.68GiB.
Base and Random initial values match reference exactly. Bare deployment contains
only Student; KAN out-of-grid smoke passes.

P1_TOP_PARAMS=65664
P2_MLP_TOP_PARAMS=655513
P2_KAN_TOP_PARAMS=655489
P2_MLP_RESIDUAL_PARAMS=589848
P2_KAN_RESIDUAL_PARAMS=589824
PARAM_MISMATCH_PERCENT=0.004069010416666667
TOP_TARGET_DIM=128
RANDOM_TARGET_DIM=32
RANDOM_BRANCH_UNCHANGED=True

Frozen component src/student/part2.py remains byte-identical to implementation
commit. G5/k3/range[-.17,.17], alpha.001, H920 and existing once-only RMS
initialization are not changed. No new calibration or sweep.
Full pytest including available-GPU component checks: 235 passed, 2 skipped.
git diff --check passed. No SUES/GTA evaluation or multi-seed experiment.
