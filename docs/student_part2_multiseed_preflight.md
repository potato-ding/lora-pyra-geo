# Shared Top-RMLP matched seeds 1 and 2

Reference: P2-TOP-RMLP-S0, source 7bbd78cc8fb5794f782e07e68c8e163c217db632.
New configs are copied from the actual S0 launch config. Only seed, experiment/output identity and the new source-seal output path differ. The S0 configuration and original seal remain intact. The normalized training diff is empty. Runtime part/research_axis metadata retain the same existing writer overrides.

The validator now permits shared residual_mlp only with exact S0/S1/S2 name-to-seed mapping and rejects altered training fields, S3, mismatches or alternate calibration. KAN/factorial validation semantics remain unchanged. No model structure, training-loop, optimizer, scheduler, data/augmentation, seed-runtime, launcher or selector source changed.

A numerical regression loads the original S0 integration source from Git. For seeds 0,1,2, identical initial states and fixed synthetic inputs give bitwise-identical Top base, residual, gated residual, normalized Top and Random Linear outputs. Top is shared across both views; Random remains Linear. Frozen component and canonical source files are byte-identical to the S0 commit.

22 selected regression tests passed; one historical KAN config case was deselected. No KAN or factorial experiment was executed.
S1/S2 each passed one real pair-batch32 DeepSpeed smoke with seed and sampler seed 1/2 respectively. A single Student forward receives concatenated N=64. Teacher stays frozen. All losses/gradients are finite, active gradients nonzero, NaN/Inf zero, optimizer policy and bare deployment preserved.
Both Student initializations loaded 1131/1131 feature keys, missing=0 and unexpected=0.
Total trainable parameters per run: 14289338. Top Linear: 65664; Top residual: 589848; Random Linear: 16416. Independent scalar Top gate adds one parameter.
Formal launch is a fresh process from the same pretrained Student, not smoke state or any trained Student checkpoint.

Teacher SHA256: 1f5dd3a94e38d5e79bfff05b407959195eb59b9b9359f2380727f6a68fed3d78
Student pretrained SHA256: d645a2de5481c9aac1639d0e97b04cd4bdb0df9d7347920b132dd0ed45de8b39
Extended STST SHA256: 3fdcd8bc62f7204a36469ba05c0cd65d4792fcf7dafeda8d4ffb6780f769b50c

Detailed raw/normalized config differences, asset verification and smoke reports are under src/checkpoint/student/CERTIFIED_R224/_PREFLIGHT/P2_TOP_RMLP_MULTI_SEED.
