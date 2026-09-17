# Final Part III relational validation

Four frozen configs: S3 S1/S2 at 0.645, S4 S0 at 0.308, and multi-stage S0
at 0.3225/0.154. All inherit their matched P2 Top-RMLP seed config. They start
from certified RepViT pretrained weights; no checkpoint resume is supported.
Global loss, heads, augmentation seed policy, optimizer, per-step scheduler,
precision and canonical selection remain unchanged.

Launch with one visible GPU:
`python -m src.student.launch_spatial_final --config <config>`.
The launcher starts Python directly. The existing DeepSpeed BF16 stage1 runtime
retains a world-size-one process group for its optimizer and the canonical
sampler; there is no distributed launcher, other rank, or retrieval gather.
Output must be fresh, the assigned GPU idle, source/config hashes sealed, and
the worktree clean. Each run uses its own loopback rendezvous port.

The parent performs one concatenated Drone32+Satellite32 Student forward.
All 171 native BN modules assert N64. Stage3 and Stage4 hooks assert their audited
shapes. Relations use per-token FP32 L2, off-diagonal Gram values, per-image
mean centering and cosine alignment. Each Teacher target supervises the exact
same image/view. S4 Teacher tokens are mean-pooled 2x2 before token L2. MS has
two independent Teacher anchors; no inter-stage matching or inference fusion.
Stage4-only computes no Stage3 loss. All relation helpers have zero parameters.

Smoke mode is isolated under `_PREFLIGHT/PARTIII_FINAL_VALIDATION`, runs one
training step and the existing explicitly marked subset selector twice,
requiring strict bare reload and identical descriptor hashes/metrics. It checks
each enabled spatial gradient separately. Formal runs log epoch loss averages,
LR, logit scale, each view/stage loss, warmup and effective coefficients.
All backbone gradients are checked every 200 steps; Teacher has no gradients.
Formal best/last remain bare Student through the unchanged canonical selector.

Only new files/configs are added. Group-I and all prior methods are unchanged.
After preflight and commit, code is frozen before the four formal launches.
No SUES/GTA tests or new experiments are dispatched after training.
