# Evaluation and efficiency

[Back to README](../README.md)

## One checkpoint, three datasets, four result files

```bash
conda activate pyra_geo
bash scripts/eval.sh --gpu 0 --model-type student \
  --checkpoint src/checkpoint/student/R256/S3-256-M3/best_model.pth \
  --image-size 256 --dataset all --data-root data
```

The physical GPU selected by `--gpu` becomes logical `cuda:0`.
Formal evaluation requires exactly one visible GPU and batch size 16.
The evaluator loads the specified Teacher/Middle/Student strictly, restores its
recorded runtime precision, and checks resolution and selection provenance.
It does not infer the model type from a directory name.

| Dataset | Direction | Saved metrics | Result file |
| --- | --- | --- | --- |
| University-1652 | D2S, S2D | R@1, R@5, AP | `test_1652.json` |
| SUES-200 | D2S, S2D at 150/200/250/300 m | R@1/5/10, top-1% recall, AP | `test_sues200.json` |
| GTA cross-area | D2S | R@1/5/10, top-1% recall, AP, SDM@1/3/5, DIS@1/3/5 | `test_gta_cross_area_d2s.json` |
| GTA same-area | D2S | Same metrics as cross-area | `test_gta_same_area_d2s.json` |

`D2S` means drone-to-satellite and `S2D` satellite-to-drone.
The JSON field `R@top1` denotes top-1% recall, not R@1.
Recall and AP are percentages; SDM is unitless in [0,1] and higher is better;
DIS is measured in meters and lower is better. Some existing U1652 result
files also contain R@10; the current formal exporter guarantees R@1/R@5/AP.

For our models the four files are written **beside the checkpoint**. A separate
formal output directory is rejected. Each JSON identifies the checkpoint SHA,
input size, precision signature, dataset protocol and measured values.
This is why a result should be associated with a checkpoint hash, not just an
experiment folder name.

## University-1652 and SUES-200

The sole formal protocol is `query_identity_retrieval`:

- Decode RGB with OpenCV, resize to the requested square input with
  `INTER_LINEAR_EXACT`, and apply ImageNet normalization.
- Extract one descriptor from the original image. No horizontal-flip fusion,
  test-time augmentation, multiscale averaging or reranking is used.
- Normalize descriptors and compute cosine similarity in FP32.
- Mark gallery identities absent from the query set as junk (-1); remove them
  after ranking the full gallery.
- Compute AP by the shared trapezoidal integration implementation.

Training selection and final U1652 evaluation use the same loaders, metrics,
resolution, batch size 16 and recorded precision. Formal evaluation requires
the selected checkpoint's D2S/S2D R@1, R@5 and AP to reproduce the saved
selection metrics. The result records `BEST_MODEL_RELOAD_CONSISTENCY=PASS`.

SUES-200 uses its prepared `Testing/{150,200,250,300}` split. These numbers
are flight heights in meters, not input image sizes. Each height is evaluated
in both directions. In the prepared split, each effective satellite gallery
contains 80 images and each effective drone gallery contains 4,000 images.
SUES results are zero-shot transfer from University-1652 training.

This is the repository's explicitly unified evaluation protocol. It is not a
claim that every upstream paper used the same interpolation, augmentation,
gallery filtering or AP implementation. Compare methods measured with matching
settings, and label paper-reported results separately.

## GTA-UAV

The default `--gta-split both` evaluates cross-area followed by same-area.
The loader reads the corresponding `*-drone2sate-test.json` and uses:

- D2S queries with at least one valid positive satellite match.
- `pair_pos_sate_img_list` for positives, without expanding to semi-positives.
- The full satellite gallery, not a query-dependent positive-only gallery.
- Deterministic resized inputs and normalized descriptors, without
  post-retrieval matching.

The result records raw JSON count, retained positive-query count, gallery count,
split, coordinate units and tile length. Counts depend on the downloaded
benchmark release; use the result's protocol metadata rather than silently
substituting counts from another release.

**The coordinate tile length is 256 independently of model input size.**
The satellite filename/zoom/offset geometry uses this fixed tile convention;
changing an encoder input from 224 to 256 or 384 must not change the
coordinates or SDM distance scale. The code's SDM and DIS operate on those
metric coordinates.

Our models are trained on University-1652. Both GTA evaluations are therefore
transfer tests on GTA benchmark splits. Calling one split `same-area` does
not mean our model was trained on GTA's same-area training split.

## Complete a partial evaluation

Existing formal result files are never overwritten. `--dataset all` is for a
run without the requested final results, not a generic resume switch.

```bash
# If only SUES-200 is missing:
bash scripts/eval.sh --gpu 0 --model-type student \
  --checkpoint src/checkpoint/student/R256/S3-256-M3/best_model.pth \
  --image-size 256 --dataset sues200

# If only GTA same-area is missing:
bash scripts/eval.sh --gpu 0 --model-type student \
  --checkpoint src/checkpoint/student/R256/S3-256-M3/best_model.pth \
  --image-size 256 --dataset gta --gta-split same-area
```

An existing matching U1652 result is verified and reused for the reload
consistency prerequisite. If it is absent, U1652 runs before the requested
transfer dataset. A conflicting existing result is an error, not a reason to
skip identity checks.

## Comparison models

Comparison papers use explicit adapters in
[`comparison_models.py`](../src/evaluation/comparison_models.py), not the
formal Teacher/Middle/Student loader. The adapter specifies the author's
backbone, weights, descriptor extraction and input size; the shared dataset
loaders and metrics then evaluate that descriptor.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.comparison --model camp --dataset u1652
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.comparison --model camp --dataset sues200
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.comparison --model camp --dataset gta --gta-split both
```

Available locally audited adapters are `sample4geo`, `dac`, `camp`,
`mean`, `mccg`, `lpn`, `muse`, `ccr`, `spikevimformer`, and
`infogeo_rd_trial`. `dmnil` requires separately supplied source and weights.
The adapter determines its asset/result location under `src/checkpoint/others/`;
GTA cross-area retains `gta.json` and same-area uses `gta_same_area.json`.
There is no `--dataset all` option for this comparison command.

Loading successfully does not establish equivalence to the author's evaluation.
MCCG, SpikeViMFormer and InfoGeo retain source/checkpoint compatibility
qualifications; LPN records assumed defaults when its original config is absent;
MuSeNet records extra training data. Retain the adapter audit when reporting
comparisons, and do not merge compatibility-trial outputs into confirmed results.

To add another model, implement and register an explicit adapter, verify its
state loading and descriptor, and declare its own input-size/pretraining
requirements. Do not load unrelated weights through `--model-type student`.

## Efficiency

The efficiency suite measures separately from retrieval:

```bash
# Input/output preflight only: no models or GPU work.
python scripts/run_efficiency_suite.py --own-only \
  --output-dir /path/to/new_efficiency_run --check

# Actual measurement on one idle GPU.
CUDA_VISIBLE_DEVICES=0 python scripts/run_efficiency_suite.py --own-only \
  --output-dir /path/to/new_efficiency_run
```

Its own-model entries are M3-224/256 and S3-224/256-M3. Without `--own-only`,
the default list also includes the ten supplied comparison adapters.
DMNIL is opt-in via `--models dmnil` after supplying its assets.
Missing inputs fail preflight. Preserve the measurement protocol and hardware
with parameter/FLOPs/throughput values; do not mix diagnostic timings into
formal efficiency tables.

`scripts/diagnose_student_mean.py` is an optional performance diagnostic:

```bash
python scripts/diagnose_student_mean.py student /path/to/new_diagnostic \
  --image-size 256 --check
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_student_mean.py student \
  /path/to/new_diagnostic --image-size 256 \
  --checkpoint /path/to/run/best_model.pth
```

Metric reference checks, cached-descriptor certification and detailed precision
contracts live in `src/evaluation/`. They do not replace the standard fresh
U1652 reload check for formal Student/Middle results.
