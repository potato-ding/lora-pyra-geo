# Part-II fixed-grid residual Top projectors

Scope: component implementation, read-only input diagnostic and synthetic/tiny
descriptor backward only. No formal run/configuration, benchmark evaluation,
optimizer step, trainer dispatch change, or source-seal update is performed.

## Frozen reference

P1-T128-R32-S0, A-Dual-STST, Top128 + Random32_A, branch weights 1:1.
The existing PartISupervision, BandProjector, Student, optimizer, scheduler,
InfoNCE, Teacher/bases, canonical selector and evaluator sources are unchanged.
The base head is an existing BandProjector.linear object, not a new/reinitialized
Linear. Install after original Top and Random construction; the residual RNG is
isolated so it does not advance the reference RNG stream.

Teacher: SAM-MABV2-RHO010-S0. Student initialization: certified RepViT-M1.5
pretrained weights (never a trained Part-I checkpoint). All future training
protocol fields remain STU-1G-B32-R224-v1, 224, batch32, single GPU, 30 epochs,
outer KD0.2, warmup5, existing optimizer/scheduler/precision/selector.

## Input diagnostic and fixed grid

StudentModel.forward returns FP32 L2-normalized BN output. The Top head receives
this descriptor, not raw BN features.

The audit samples 12 TRAIN batches, seed0, sampler epoch1, 32 unique identity
pairs per batch, 768 images total, 393216 descriptor elements. It uses existing
TRAIN augmentation, pretrained BF16 Student and training-mode batch statistics
under no_grad. All BN buffers are restored after each forward; full state and
the pretrained/Teacher/bank SHA256 identities are checked unchanged. No
optimizer is created and no backward is called by this diagnostic.

Observed element statistics:
min=-0.25932836532592773
max=0.18653316795825958
mean=0.0000026057123663333092
population_std=0.044194173950574775
p0.1=-0.13288030952215193
p1=-0.10149098187685013
p99=0.10140489414334293
p99.9=0.13281416557729245

R = ceil(1.20 * q99.9(abs(input)) * 100) / 100 = 0.17.
Observed coverage of [-0.17,0.17] is 0.9999262491861979. This is a sampled
initialization distribution, not a claim about all future trained descriptors.
Spline inputs outside this range clamp to the endpoint. The SiLU base component
continues to use the original input. No grid updates are implemented.

## Exact KAN formula and provenance

This is a one-layer spline KAN, not FastKAN/FourierKAN/GaussianKAN.

For output o and input i:
phi_oi(x_i) = a_oi * SiLU(x_i) + sum(j=0..7) c_oij * B_j(clamp(x_i,-R,R)).
KAN(z)_o = sum(i=0..511) phi_oi(z_i).

The trainable spline scale is absorbed into c, and the effective separate spline
scale is fixed to 1; a and c are both trainable. No output bias is added.
G=5 uniform intervals, cubic degree k=3 (the KAN convention for order),
G+k=8 basis coefficients per edge. Knots extend by k steps at each endpoint.
Basis evaluation uses the Cox-de Boor recurrence. Positive fixed denominators
avoid division by zero. Boundary behavior is tested against SciPy's independent
B-spline implementation (test only; runtime has no third-party KAN dependency).

The single G=5,k=3 choice follows the small KANLayer defaults, not a sweep.
Concept sources, not copied implementation:
- Original paper: https://arxiv.org/abs/2404.19756
- Author KANLayer: https://github.com/KindXiaoming/pykan/blob/master/kan/KANLayer.py
- Author grid documentation: https://kindxiaoming.github.io/pykan/API_demo/API_5_grid.html

## Shared residual wrapper and initialization

raw_top = F.linear(z_fp32, base_weight.float(), base_bias.float())
          + alpha * residual(z_fp32)
top = FP32 L2(raw_top)

Both controls use the existing cosine KD and equal Drone/Satellite averaging.
Random32 remains the original Linear -> L2. No GBW or per-view coefficient is
introduced. Alpha is an independent scalar parameter, initialized to 0.001,
not zero. The optimizer's existing no-decay rule places scalar alpha and biases
in no-decay, other residual matrices in the existing decay group.

Base Linear initialization is untouched, including inherited D0 first32 rows and
the Part-I fixed extra Top rows.
KAN a: nn.Linear-style Kaiming uniform(a=sqrt(5)).
KAN c: Uniform(-0.01/sqrt(512), +0.01/sqrt(512)).
MLP: Linear(512,920,bias=True) -> GELU -> Linear(920,128,bias=True),
using nn.Linear default initialization.
Both residual constructors use isolated CPU seed0.

To prevent one candidate gaining a larger initial contribution, the same
once-only initialization procedure matches residual RMS to base RMS on the
same fixed diagnostic inputs. It multiplies KAN a/c, or MLP final weight/bias,
by base_RMS/residual_RMS. This is initialization only, not an online weight or
additional forward multiplier. Repeat calibration and calibration after
backward are rejected. Neither base nor Random is rescaled.

Measured at the canonical BF16 base storage:
MLP initialization scale=1.7106369733810425.
KAN initialization scale=2.738116979598999.
Both gated residual/base norm ratios=0.0010000001639127731.
The shared nonzero gate passes nonzero gradients into both branches initially.

## Precision

KAN basis, edge projection, MLP residual, gate, addition, normalization and loss
are FP32, even inside an outer CPU/CUDA BF16 autocast context. Residual/gate/grid
remain FP32 through module BF16 casts without rounding the stored values.
The base Linear follows the historical BF16 storage, then uses its existing
FP32 projection computation. Student backbone is untouched.
No global default dtype changes occur.

## Exact trainable parameter counts

Base Linear: 512*128+128 = 65,664.
KAN residual: 512*128*(1+8) = 589,824.
Scalar alpha: 1.
Total residual-KAN Top: 65,664+589,824+1 = 655,489.
MLP residual: 513*H+128*H+128 = 641H+128.
Nearest integer matching gives H=920, P_MLP=589,848.
Mismatch relative to KAN residual = 0.004069010416666667 percent.
Total residual-MLP Top = 655,513.
Unchanged Random32: 16,416.
All projectors: Linear82,080; residual-KAN671,905; residual-MLP671,929.
Fixed grid is a buffer and contributes zero trainable parameters.

## Verification and use boundary

tests/test_student_part2.py covers shapes, counts, original base/RNG preservation,
zero-gate regression probe, unchanged Random outputs/basis, gradient flow,
partition of unity, endpoint and out-of-range behavior, independent SciPy basis
agreement, FP32 storage/autocast behavior, CUDA backward, optimizer inclusion,
actual deployment stripping, and unchanged source seals.
Existing regression tests are also run.

The smoke report contains per-group one-backward gradient norms and initial
output ratios for Linear/MLP/KAN, without performance evaluation or optimizer
updates. Bare Student state contains no residual, gate, Top or Random parameters.

The explicit install_residual_top helper is available for future integration.
The formal trainer/config validator is deliberately not wired to new Part-II
run names in this component-only task. A future authorized preparation must
add and seal the two S0 configs/dispatch and verify the actual DeepSpeed training
runtime before formal launch. Passing component tests does not certify that
not-yet-created formal launch path.

Audit assets:
src/checkpoint/student/CERTIFIED_R224/_AUDITS/P2_RESIDUAL_IMPLEMENTATION/
(input_distribution.json, input_batch_manifest.json, diagnostic_inputs.pt,
smoke_report.json and final status/report).
