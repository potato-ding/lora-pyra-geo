# S3 dimensional controls at R256

All five controls use frozen M3-256, seed0, 256 input, RepViT-M1.5 from the
same pretrained weights, 30 epochs, one GPU / batch32, existing optimizer and
scheduler, DG-SAM rho=.1 with KD-only first pass and full second pass. U1652
selection starts after epoch11 (single GPU / batch16, D2S+S2D R1 sum).

| Variant | GPU | Top dimension | Random dimension | Allocation |
|---|---:|---:|---:|---|
| d1_top256_random32 | 4 | 256 | 32 | existing bounded GBW |
| d2_top256_random64 | 5 | 256 | 64 | existing bounded GBW |
| d3_top256 | 6 | 256 | none | single branch coefficient1 |
| d4_top512 | 7 | 512 | none | single branch coefficient1 |
| d5_top512_random32 | 0 | 512 | 32 | existing bounded GBW |

## Assets and initialization

The new asset is isolated in
src/checkpoint/student/R256/middle_assets/M3-256/dimension_extension/.
Existing mean/Top128/Random32 assets and all historical checkpoints are unchanged.

M3 TRAIN descriptors are pooled into the same 701 identity centroids per view,
1402 rows total. The original mean and first128 directions remain byte-identical.
Additional384 directions are fitted by FP64 SVD of centered TRAIN centroids in
the orthogonal complement of that fixed Top128; the resulting tail is projected
and orthonormalized. Top256/512 are nested prefixes of this512-column matrix.
This is a constrained PCA extension with an exact historical prefix, not an
independent refit of all512 directions. No test examples are used.

Random32 is exactly the original final S3 tensor. Random64 keeps those32 columns
and adds32 orthonormalized Gaussian directions in their orthogonal complement,
using the same RNG after the original768x32 draw. No orthogonalization against
Top is performed. Original32 protocol/tensor is unchanged; Random64 has this
explicit nested-completion protocol, not a fresh independent768x64 matrix draw.

Top heads use Linear512->d plus FP32 residual MLP512->920->d, GELU, alpha=.001
and the same once-only calibration RMS matching. Random heads use Linear512->r.
Top base first128 outputs and Random base first32 outputs are copied from the
canonical initial heads, never trained heads. Added rows use fixed initialization.
Changing dimensions changes parameter counts and the calibration scale; predictor
initial outputs beyond the preserved base rows are not claimed identical.

## Loss and comparison limits

Every target is normalize((f_M - mean) @ basis), prediction is L2-normalized,
and branch loss averages cosine distance equally over D2S/S2D image views.

    lambda(e) = .2 * min(e / 5, 1)
    dual KD = w_top * L_top + w_random * L_random
    w_top + w_random = 2, each bounded in (.5,1.5)
    single KD = 1 * L_top
    L_full = existing PairInfoNCE + lambda(e) * KD

Single controls intentionally follow the existing B3 Top-only coefficient1;
they do not rescale to preserve the dual-branch sum2. Thus single versus dual
comparisons change both branch information and effective loss allocation.
Keep that limitation explicit when interpreting Random-branch benefits.

## Strict contracts and readiness

Five configs are configs/student/d*_sam_m3_256.json. All use the existing
train_student.sh -> formal_train entry, with a distinct dimension-specific
supervision/auxiliary validator. Tensor shapes, FP32 knowledge/residuals,
manifest SHA, Middle identity, TRAIN path identity and original prefix hashes
are checked. Full deployment remains the unchanged512D Student encoder.

Build (once on an idle GPU; refuses an existing asset directory):

    CUDA_VISIBLE_DEVICES=<gpu> python scripts/build_student_dimension_assets.py \
      --output src/checkpoint/student/R256/middle_assets/M3-256/dimension_extension

CPU preflight:

    python scripts/check_student_dimensions.py --variant d1_top256_random32

Explicit GPU smoke:

    CUDA_VISIBLE_DEVICES=<gpu> python -m torch.distributed.run --standalone \
      --nproc_per_node=1 scripts/check_student_dimensions.py \
      --variant d1_top256_random32 --gpu-smoke

The smoke runs three real BF16 DeepSpeed batches and a unified deployment reload
using a temporary synthetic-metric fixture that is deleted. These are not real
evaluation results and are never placed in formal experiment folders.

Training launchers are frozen copies with an explicit repository ROOT, avoiding
in-place shell edits changing the read offset of a still-running bash launcher.
No existing running launch script is modified.
