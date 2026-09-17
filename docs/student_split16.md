# B0-1G-SPLIT16-S0

This is an explicitly enabled, fixed causal control. Existing B0, P2 and other
trainers do not import it. Launch only with `scripts/train_student_split16.sh`
and `configs/student/certified_r224/b0_1g_split16_s0.json`, on physical GPU2.

The base global32 single-GPU protocol identifier remains STU-1G-B32-R224-v1;
`control_protocol_id=STU-1G-SPLIT16-B32-R224-v1` and `split_forward_enabled=true`
identify the different forward and persistent-buffer semantics. The canonical
selector and its checkpoint schema are unchanged.

## Reference evidence

The reference source is commit 3f942a8eb50e8fbd3b5718f016a5306d9adb009c.
`src/student/repro_2g.py:epoch_buffer_sync` broadcasts registered buffers from
rank0 only at epoch end. `run` executes this before `distributed_select`, which
saves rank0's canonical Student state. Thus rank1's local running statistics do
not contribute to saved buffers. There is no forward-time buffer collective,
SyncBN, buffer averaging, or copy from rank1 to rank0. The audit preserves exact
source excerpts with line numbers, the first/last epoch log records and both
final rank buffer checksums in `_PREFLIGHT/B0_1G_SPLIT16_S0`.

## Forward and buffers

Each global batch has 32 unique paired identities. Two ordinary train-mode
Student forwards each concatenate Drone16 and Satellite16, so all BN modules
see N32. Both graphs use the same parameters. Before the second forward, BN
running buffers are temporarily rebound to detached clones. After it, the
original objects (already updated by forward0) are restored. No in-place copy
touches a tensor saved by either autograd graph. This preserves rank0's local
buffer stream, including one num_batches_tracked increment per global step.
Normalization in both forwards uses the respective batch statistics.

The two drone descriptor halves and two satellite halves are concatenated in
rank order. The unchanged PairInfoNCE computes one FP32 global32 symmetric loss.
One backward and one AdamW call follow. No gather or process group is created.

## Precision and optimizer

The reference BF16 model precision is retained, including FP32 normalized
descriptors, similarity and loss. The reference builds torch.optim.AdamW and
DeepSpeed replaces its parameter groups with FP32 master partitions. This
control removes that runtime and uses the same AdamW constructor, no-decay
grouping, betas, epsilon, scheduler and LR. Its independent adapter maintains
unpartitioned FP32 flattened masters/moments, initialized after BF16 conversion,
converts the one global batch's gradients to FP32, steps once and copies back
to the BF16 model. No gradient/loss scaling, clipping, accumulation across
batches, or LR scaling is introduced. Bitwise optimizer equivalence to
distributed partition/reduction is not asserted.

## Sampler and augmentation

Two unchanged CrossViewPairSampler instances reproduce rank0/rank1 shards of
the same seed+epoch global32 batches. The full sequences for epochs 1, 2 and 30
are checked against the existing global sampler, with the first 20 epoch1
partitions recorded. The unchanged U1652PairDataset/transforms and worker seed
function run in independent 8-worker loader pools. Both loader generators use
seed0, matching the reference's absence of a rank offset, but historical worker
base seeds depended on prior process RNG consumption. Therefore
AUGMENTATION_RNG_EXACT_MATCH=False; augmentation policy is unchanged.

## Gates and lifecycle

Unit tests check gradient equivalence to independent local replicas, buffer
persistence across two steps, positive ordering, FP32 master AdamW behavior,
exception restoration, fixed config and unchanged historical sources. The real
TRAIN smoke additionally checks all 171 BN contexts, both local contributions,
finite gradients, one optimizer step, fresh canonical selection and bare reload.
Smoke state is isolated and never used for initialization of the formal run.

The launcher requires the clean sealed source, a successful smoke and idle GPU2.
It refuses a nonempty formal output directory. Formal training always starts
from the verified pretrained Student and runs 30 complete epochs, with no early
stopping. Every epoch uses the existing canonical U1652 selector. Completion
strict-loads bare best weights and records the epoch30 completion status. No
SUES/GTA evaluation or subsequent experiment is launched.
