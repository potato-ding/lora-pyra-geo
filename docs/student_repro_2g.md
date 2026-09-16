# B0-2G-REPRO-S0

The only approved reproduction uses GPU0,1, seed0, 30 epochs, local16/global32 pairs, ordinary local BN, and the existing differentiable descriptor gather. No Teacher, KD asset, projector, BNCC, or additional objective is instantiated.

The base config is `configs/student/certified_r224/b0_baseline_s0.json`, checked against the completed canonical `B0-BASELINE-S0/run_config.json`. Only world size, local pair batch, gather, GPU count, protocol name and experiment/provenance paths change. Model, augmentation, sampler, optimizer, LR, scheduler, precision and objective sources are checked against that completed run's commit. DeepSpeed configuration matches historical commit `655c5a926757bb8be888b79e9c0e891b03d41497`.

`CrossViewPairSampler` builds unique-identity global32 batches before contiguous per-rank slicing. Its complete epoch1 global batch sequence matches 1G. `GatherLayer.backward` all-reduces each rank's contributions, then returns the owning slice; DeepSpeed performs its existing parameter-gradient averaging. A two-rank synthetic identity and rank-weighted backward test validates these semantics.

DeepSpeed does not wrap this model in DDP, so DDP `broadcast_buffers` is not applicable. No per-forward buffer broadcast or SyncBN is added. The existing BN-fix-era epoch-end rank0 registered-buffer synchronization is retained before selection. Both rank BN checksums are logged before that synchronization.

Only rank0 serializes the canonical candidate. A non-distributed rank0 subprocess strictly reloads its bare Student state and calls the unchanged `canonical_selection.select_epoch`; this preserves canonical precision, full U1652 batch32 evaluation, strict greater-than, metadata and atomic checkpoint publication. The subprocess overrides only the checkpoint protocol ID to `STU-2G-B32-R224-REPRO-v1`. The outer rank1 blocks on the result broadcast and barrier. The candidate is reserialized losslessly by the shared selector and removed by its normal cleanup. No alternative evaluation implementation is added.

The launcher follows the existing reserve-output/tee/torchrun pattern, using exactly two ranks and the same conda environment. Smoke performs three real paired training steps, then a marked canonical U1652 subset solely for selector/barrier validation. Formal training starts from pretrained in a fresh process, completes 30 epochs, strictly reloads the final selected bare checkpoint and writes `training_completion.json`. It never starts SUES/GTA tests.
