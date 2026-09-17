# Fixed and learnable GBW with Top-RMLP

Four approved S0 controls use unchanged P2 Student/Teacher construction, STST
assets, Top residual MLP calibration, optimizer grouping, precision, data pipeline,
and canonical U1652 checkpoint selector. Only branch allocation differs.
All launch directly as one Python process with DeepSpeed stage 1, world size 1,
one concatenated 64-image Student forward, 30 epochs, pair batch 32 and seed 0.

Fixed allocation is 1.247/0.753, using the historical GBW reducer. Bounded gates
use 0.5+sigmoid(d), 1.5-sigmoid(d) with d=1.0826756964052977 or d=0.
Unbounded allocation uses 2*sigmoid(d), 2*(1-sigmoid(d)) with
d=0.504430717880143. The allocation budget is 2. FP32 rounding applies.

Student loss is InfoNCE plus 0.2*min(epoch/5,1) times the two weighted branch
losses. Learnable weights are detached for Student backward. A separate FP32
scalar, outside the DeepSpeed model and Student optimizer, receives only
min(epoch/5,1)*(log(w_top*G_top+1e-8)-log(w_rand*G_rand+1e-8))^2.
Its AdamW uses lr=1e-4, betas=(0.9,0.999), wd=0 and the exact Student LR schedule.

G is the mean of the norms of the two unweighted per-view descriptor gradients.
The existing per-view cosine losses and their aggregation are reused. To avoid
DeepSpeed backward hooks during diagnostic autograd.grad calls, a detached
descriptor leaf and detached head tensors run the same heads through
torch.func.functional_call. This repeats only head computations, not Student or
Teacher forward. Loss and descriptor-gradient equality are checked against the
live graph in CPU tests and the GPU preflight. All G values are detached and
create_graph=False; no second-order or additional Student gradient is introduced.

The fixed path has no gate parameter. Deployment checkpoints remain bare Student
through the unmodified canonical selector. No gate, projector, Teacher or optimizer
state enters those checkpoints. No Spatial KD, BNCC, SAM or Split16 path is used.

The shared GPU7 preflight checks historical P2 1:1 and GBW Linear equivalence,
all three gate initializations, gradient isolation, four actual DeepSpeed steps,
Teacher freezing, N64 BN context and strict bare-state reload. Formal processes
start independently after source sealing, on GPUs 4/5/6/7. Initial model/head/RNG
and first augmented batch hashes allow matched-seed verification.

The original frozen calibration file was missing. The unchanged historical
part2_input_audit script reproduced its exact SHA256
0953c48128e5c9e0c0ea396bdbcd1ca0a5d95199d8e353bb65615079755cf584.
Only that byte-identical file was restored. No PCA or random basis was regenerated.
