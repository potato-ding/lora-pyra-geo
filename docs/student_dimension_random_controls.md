# Top256 Random-basis stability controls

Four runs, all training seed0, same frozen M3-256, same exact mean and Top256,
pretrained Student, predictor initialization, optimizer, augmentation/data seed,
GBW and DG-SAM as the original D1/D2. Only the run-level Random target basis
changes within each dimensional setting. Training is30 epochs / single GPU /
batch32 at256. Existing U1652 best selection remains unchanged.

| GPU | Method | Random replicate | Training seed |
|---|---|---:|---:|
|4|D1 Top256+Random32|1|0|
|5|D1 Top256+Random32|2|0|
|6|D2 Top256+Random64|1|0|
|7|D2 Top256+Random64|2|0|

Names end in RANDOMBASIS1/2, never SEED1/2. Original D1/D2 results are retained.
Each method therefore has three Random-basis runs including the original.

The independent basis seeds are2318409933239863148 and194854658740768743.
Each is sampled once and explicitly stored in random_basis_control. The common
training seed remains0 in every config. Random32 uses the unchanged CPU FP64
Gaussian->QR->max-absolute-pivot sign->FP32 protocol. Random64 uses the same
nested Gaussian completion protocol as the original D2: its first32 columns
are the same-index Random32 tensor, with32 additional orthogonal Gaussian
columns. Neither basis is residualized against the Top subspace.

The generated assets are in
src/checkpoint/student/R256/middle_assets/M3-256/dimension_random_controls/.
They have separate file and tensor SHA, generation_count1, source dimension
asset SHA, dimension, resolution, source seed and protocol. Runtime loads the
fixed asset (generation_count0), so basis construction does not consume the
training RNG. Predictor initial parameters and RNG states are checked to match
across the two new bases per method. The original subspace_reference remains
historical mean/Top/canonical-head provenance; random_basis_control is the
actual Random target source and is recorded separately in subspace_identity.

Strict checkpoint validation verifies the actual Random basis tensor against
its own asset and checks its actual basis seed, rather than comparing it to
the historical original Random32 seed. Original configurations and old strict
checkpoint paths remain supported. New Random files never replace the old
nested dimension asset. Deployment is unchanged512D Student-only retrieval.

Tests cover RNG independence, deterministic generation, orthogonality,32/64
nested prefixes, closed configs with training seed0, identical predictor
initialization despite different Random targets, checkpoint roundtrip and
corrupted Random tensor/seed rejection. No empirical result is implied by
these integrity checks. No commit/push.
