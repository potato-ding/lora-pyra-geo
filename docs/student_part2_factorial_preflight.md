# Fixed Part-II view and branch residual controls

All four S0 configurations inherit the existing P2-TOP-RMLP-S0 launch configuration, except experiment identity, output directory, source seal path and fixed interface dispatch. The frozen Teacher, pretrained Student, Top128/Random32_A bank, batch, optimizer, DeepSpeed, augmentation, objective and selector are unchanged.

Top controls directly reuse ResidualTopProjector and its 512-920-128 GELU residual, scalar gate, FP32 projection/normalization and once-only frozen TRAIN calibration. View-only controls retain one shared Linear and one residual; only the active half of the concatenated [Drone, Satellite] batch enters the residual. Inactive rows bypass it exactly.

Random uses fixed 512-1082-32 layers and inherits the same GELU FP32 forward, residual form, independent scalar gate and once-only amplitude initializer. The same frozen 768-row TRAIN calibration is used, with no fitting or search. Dual has independent Top and Random gates.

Parameter counts excluding gates: Top residual 589848; Random residual 589722; difference 126 (0.021361435488464826% of Top). Shared Top Linear 65664; Random Linear 16416. Extra parameters including scalar gates: Top Drone 589849; Top Satellite 589849; Random 589723; Dual 1179572.

The original AdamW object, options and no-decay policy are preserved. Existing P2 dtype partitioning keeps Student/base/Random BF16 and residual/gate FP32. Original basis loading and runtime conversion are unchanged, and all protected bank/checkpoint files retain their reference SHA256.

Validation:
- 25 regression tests passed, one historical KAN configuration case deselected; no KAN model executed.
- Exact view bypass, active gradients, independent gates, fixed config rejection, optimizer inclusion, bare deployment and historical shared Top equivalence covered.
- Four real paired-batch smoke runs passed using batch32 and one full forward/backward/DeepSpeed optimizer step per run.
- Teacher remains frozen, losses and active gradients finite, NaN/Inf zero; view masks match specified activation.
- Peak allocated VRAM approximately 5.69 GiB or less per smoke.
- train.py, part2.py, part1.py, dual_stst.py, optimizer, scheduler, model, data, canonical evaluator/selector and launcher remain byte-identical to 7bbd78cc8fb5794f782e07e68c8e163c217db632.

Detailed smoke reports and parameter summaries are in src/checkpoint/student/CERTIFIED_R224/_PREFLIGHT/P2_FACTORIAL. Formal launches must pass the same clean source seal gate as existing P2 runs, then complete epoch1 canonical selection confirmation. No SUES/GTA testing is dispatched.
