# STU-1G-B32-R224-v1

All certified Student methods use one GPU, 32 local/global pairs, no cross-GPU
feature gather, and no gradient accumulation. A step forwards 64 images and uses
32-by-32 bidirectional InfoNCE. DeepSpeed ZeRO stage 1, BF16 training, optimizer,
scheduler and loss implementations are unchanged. Native BatchNorm now sees the
full local batch. No SyncBatchNorm is introduced.

Each epoch saves a detached FP32 deployment candidate (integer buffers retain
their type). Selection and the formal Student U1652 wrapper both invoke
canonical_selection.evaluate_checkpoint, which runs the unchanged unified
evaluator in a fresh interpreter without torchrun rendezvous variables.
Strict reload, batch 32, deterministic test transforms, FP32 parameters with
CUDA BF16 autocast, and FP32 normalized descriptors are shared. The approved
selection dataset is University-1652 test; score is D2S R1 + S2D R1, every epoch,
strict greater-than. SUES/GTA never participate in selection.

The evaluated candidate bytes become last_model.pth and, on improvement,
best_model.pth. Candidate files are cleaned up; epoch and best JSON metrics come
directly from this evaluator. Training remains in its own process, so evaluation
does not advance its RNG or mutate its BN state. Evaluation temporarily requires
memory for both the resident training engine and the fresh evaluation model.

B0 and D0 configs are seed-matched. D0 retains the certified SAM-MABV2-RHO010-S0
Middle checkpoint, shared train-only Top32/Random32 bank, weight 0.2 and five-epoch
KD warmup. Do not change these paths to other teachers without a new protocol audit.

The launcher requires the protocol source seal and a clean worktree. It refuses
nonempty output directories. Existing two-GPU B0 S0/S1 assets are legacy evidence,
not STU-1G-B32-R224-v1 results; this migration does not remove or overwrite them.
The source seal is stored outside Git under _PREFLIGHT and binds the final commit,
source hashes, all six config hashes and smoke evidence, avoiding a self-referential
commit hash in tracked config files.

Bounded smoke entry: src.student.single_gpu_smoke, launched with torchrun
--standalone --nproc_per_node=1, GPU0 for B0 and GPU1 for D0. It requires a fresh
_PREFLIGHT output and executes exactly three real training steps. B0 additionally
compares two fresh canonical reloads on an explicitly marked deterministic audit
subset. Subset payloads cannot be published as formal results. Smoke metrics are
not benchmark results or checkpoint selection for an official experiment.

No 30-epoch training is part of the migration.
