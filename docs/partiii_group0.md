# Part-III Group-0 preflight

`tools/audit/partiii_group0.py` is a TRAIN-only diagnostic entry point. It has no
optimizer, scheduler, formal training entry, evaluation/selection call or model
save. Four independent spawned processes isolate physical GPU0/1/2/3. The
parent merges point/relational gradients on CPU through pipes; no full gradient,
descriptor, feature or weight cache is written. A busy GPU is skipped and its
worker is marked failed/busy. No processes are killed or moved to another GPU.

All new helpers live in `src/student/spatial_group0.py`. Existing trainers and
configuration files are unchanged. The five spatial flags default to false.

## Gradient workers

GPU0 and GPU1 use the existing pure single-GPU global32 data/forward convention:
one Student train-mode forward of Drone32 concatenated with Satellite32. They
load the certified RepViT pretrained weights, construct fresh PartISupervision,
and run the unchanged current Top-RMLP initializer/calibration. Initialization
seed0 matches the current S0 parent setup. Student, Top base and Random storage
are BF16; Top residual/gate and teacher target bases stay FP32. Teacher is frozen
BF16 and supplies both global CLS and final-normalized spatial patch targets.

Each of 16 independent batches restores the same initial Student and parent
head state, including BN buffers. Manifest and augmentation seed20260917 are
separate from model initialization. Existing U1652PairDataset and transforms
are called directly after explicitly resetting Python/NumPy/torch and both
Albumentations transform RNGs. Both workers verify identical input checksums,
Student/head initial hashes and gradient parameter layouts. Teacher and Student
consume identical BF16 image tensors. All 171 Student BN modules see N64 once.

The point projector is a fresh, shared Conv2d(256,768,1,bias=True), isolated
seed20260917, with explicitly recorded FP32 storage and computation. It never
changes the parent's RNG or parameters. The relational loss has no projector:
FP32 per-token L2, self-similarity Gram, exclude diagonal, independently center
each image's off-diagonal values, then one minus flattened cosine. Both losses
average separate same-image drone and satellite supervision equally.

The gradient audit computes actual autograd gradients of retrieval, 0.2 times
Top+Random, their sum, spatial, spatial-drone and spatial-satellite losses.
The diagnostic uses full-strength KD w=1 only; it does not alter the future
warmup w(e)=min(e/5,1). Norms exclude Teacher, all heads and the Student neck.
ALL_BACKBONE includes all backbone parameters. STAGE3_AND_DOWNSTREAM means
features.12 through features.42, including zero unused spatial gradients in
downstream parameters. BF16 parameter gradients are cast FP32 for transfer;
norms, cosines and statistics use CPU float64, sample std ddof=1.

Lambda selection uses ALL_BACKBONE only:
`min(median(kd_norm), 0.25*median(ret_norm))/median(spatial_norm)`.
Values inside [0.01,1] receive three significant digits; outside values are
preserved with scale_warning=true and lambda_value=null. No sweep or clamping.

## Representation and shift workers

GPU2 uses the read-only P2 best Student and the prior verified 256-identity,
512-image TRAIN manifest. It obtains the final stage boundary from actual
backbone output indices and forward-hook inventory. Teacher raw final-norm
14x14 tokens are mean-pooled in non-overlapping 2x2 blocks and only then L2
normalized. No resize is used. Geometry excludes the diagonal. Structural
checks include exact7x7, row order, finite features and exact non-collapse:
positive spatial variance and effective rank above1 for every image. No
correlation threshold chooses a training candidate. Diversity definitions are
the same as the previous interface audit: raw/normalized sample token variance,
pre-L2 norm statistics and entropy effective rank from normalized token singular
values recovered via clipped FP32 Gram eigenvalues.

GPU3 uses four integer shifts with (dx,dy) in pixels, positive right/down.
New pixels use constant zero in normalized coordinates (ImageNet mean RGB), no
wrap. Coordinate-ID images prove valid-token mappings exclude padding. The
Teacher stability statistics cover all512 images and all four cardinal shifts;
pooled statistics weight valid tokens equally. Weights are detached continuous
confidence divided by within-image overlap mean. Range clamping only guards
floating-point cosine excursions, not a hard selection mask. Pair weights are
the outer product. A separate two-image, eval-mode P2 Student smoke runs one
weighted overlap-consistency backward, checks Stage3 gradients, clears gradients
and verifies unchanged state. No optimizer step or model save occurs.

## Outputs and limits

Results are JSON/CSV plus factual README under PARTIII-GROUP0-PREFLIGHT-V1.
Logs remain outside the result archive. No U1652 TEST, SUES or GTA input is used.
No Group-I experiment is automatically enabled regardless of preflight status.
