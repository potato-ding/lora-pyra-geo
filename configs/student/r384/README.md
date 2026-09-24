# R384 Student preparation

Two independent single-GPU runs, 30 epochs, pair batch 32, one concatenated
64-image Student forward, no cross-GPU gather:

- GPU6: `s0-infonce-r384.json` / `S0-INFONCE-R384`.
- GPU7: `s3-adual-learnable-r384.json` / `S3-ADUAL-LEARNABLE-R384`.

The final method retains S3 Top128 residual MLP (512->920->128, alpha 0.001),
Random32 Linear, and bounded Learnable GBW (initial 1/1, total 2).
KD outer weight remains 0.2 * min(epoch / 5, 1).

Per the corrected run-level protocol, Random32 is generated using the existing
isolated CPU float64 Gaussian + reduced QR + max-absolute-pivot sign convention,
then stored as FP32 and fixed throughout the run. The seed in the final config
was drawn once from OS entropy before training without consulting metrics.
It is not inherited from S3, S12, or S13. A new independent formal run requires
its own seed and output identity; reloading this run restores the saved tensor.

The final config uses generated_fixed and TOP128_CANONICAL assets. The obsolete
original32 asset bindings and historical A/B seed fields are removed. Its
lambda_top/lambda_random placeholders are 1/1 as required by the generated-fixed
schema; S3's 1.247/0.753 placeholders were inactive in Learnable GBW, which uses
AllocationGate values. Neither the gate nor objective mathematics changed.

PCA/mean were refitted from M2-HRD-SEM-R384 using the canonical deterministic
TRAIN path ordering: 701 identities, 37854 Drone images and 701 Satellite
images, aggregated into 1402 normalized identity/view representatives. No
augmented training feature cache is used. R384 RMS calibration uses the same
canonical first 384 TRAIN images per view and initial Student as R224.
Assets and their bound hashes are under
`src/checkpoint/student/R384/TOP128_CANONICAL/` (not tracked model artifacts).

Both entry points propagate img_size into the shared full U1652 selector.
Selection remains single rank, eval batch 32, D2S_R1 + S2D_R1, strict greater,
and earlier checkpoint on ties. Formal directories contain best_model.pth and
train.log only before separate formal testing. Final best checkpoints include
exact R32, seed/provenance/SHA, PCA mean/basis, Top/Random heads and GBW state.
The existing strict auxiliary loader restores the stored R32 without generation.

Preparation smoke outputs are in /tmp/student_r384_{baseline,final}_smoke_v2.*.
The first smoke attempts exited before training because the Conda activation
was omitted; successful v2 attempts used the activated pyra_geo environment.

Do not start formal training from this uncommitted working tree. The user must
commit the source, configs and tests first; the existing formal launcher enforces
that gate. No formal training or formal testing was started during preparation.
