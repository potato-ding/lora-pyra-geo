# Part III Group-I

Additive entry: `src.student.launch_spatial_group1 --config <config>` with one
visible authorized GPU. It retains the Part-II Student, A-Dual assets, residual
Top MLP, Random32, optimizer grouping, BF16 DeepSpeed stage1, data pipeline and
canonical fresh-reload selector. Existing trainers and evaluators are unchanged.

The three configs differ only in identity and spatial objective. Point weight
1.0 is the user-frozen conservative cap; relation weight is 0.645. Both use the
existing min(epoch/5,1) warmup. Native BN sees one concatenated N64 Student
forward. Both models receive the identical augmented image tensor in identical
order. Spatial targets always pair the same image and position, never views.
The FP32 256-to-768 point projector is train-only. Relation consumes raw Stage3
features, without a projector. Shift and stability weighting are disabled.

`--smoke-output <fresh preflight path>` performs one training step and marked
subset selector/reload checks, without creating a formal run. Formal launch
requires a clean sealed commit, matching source/config hashes and a fresh output.
Each epoch logs all spatial/global components; finite head/backbone gradients
are sampled every 200 steps. Expensive independent spatial gradient verification
occurs only during preflight. Best/last contain only bare Student tensors.

GPU3: `python -m tools.audit.partiii_stage4_grad`. Uses the exact Group-0 16 TRAIN
batches/augmentations and initial Student/parent heads, restored before each
batch. Stage4 membership follows the actual backbone output boundary and module
parameter identities. Raw final-norm Teacher tokens are pooled before L2.
No optimizer, selection or saved tensor files. Six small reports are archived.
