# Results and checkpoint provenance

[Back to README](../README.md)

## Scope

This snapshot was assembled on **2026-10-17** from the final evaluation JSONs of six completed T0/M3/S3-M3 runs. It does not rerun evaluation or aggregate unfinished experiments. All four result files for each row carry the same checkpoint SHA and matching input size; their U1652 reload-consistency field is PASS. These checks establish consistency of saved records, not a fresh feature-extraction audit.

These are selected reference runs, not a complete ablation table or a state-of-the-art claim. The five R256 random-subspace repetitions and direct T0-to-S3 control are not yet summarized as completed multi-dataset experiments. Missing measurements are not replaced with training-selection values.

Recall and AP are percentages. Each SUES height is reported separately. SDM is unitless and higher is better; DIS is in meters and lower is better. Values are rounded only for presentation.

## University-1652

| Model | Input | D2S R@1 | D2S R@5 | D2S AP | S2D R@1 | S2D R@5 | S2D AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 | 224 | 96.52 | 99.43 | 97.20 | 96.86 | 97.72 | 96.03 |
| T0 | 256 | 96.81 | 99.50 | 97.43 | 97.29 | 97.86 | 96.52 |
| M3 | 224 | 95.75 | 99.04 | 96.52 | 97.00 | 98.43 | 95.66 |
| M3 | 256 | 96.15 | 99.15 | 96.84 | 97.00 | 98.15 | 95.89 |
| S3-M3 | 224 | 91.96 | 97.65 | 93.26 | 94.72 | 96.58 | 91.53 |
| S3-M3 | 256 | 93.20 | 97.87 | 94.28 | 95.15 | 96.58 | 92.34 |

## SUES-200 transfer

| Model | Input | Height (m) | D2S R@1 | D2S AP | S2D R@1 | S2D AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 | 224 | 150 | 96.58 | 97.41 | 100.00 | 98.50 |
| T0 | 224 | 200 | 98.72 | 98.99 | 100.00 | 99.37 |
| T0 | 224 | 250 | 99.20 | 99.35 | 100.00 | 99.28 |
| T0 | 224 | 300 | 99.38 | 99.49 | 100.00 | 99.79 |
| T0 | 256 | 150 | 97.97 | 98.48 | 100.00 | 99.00 |
| T0 | 256 | 200 | 99.52 | 99.61 | 100.00 | 99.53 |
| T0 | 256 | 250 | 99.52 | 99.62 | 100.00 | 99.37 |
| T0 | 256 | 300 | 99.70 | 99.75 | 100.00 | 99.92 |
| M3 | 224 | 150 | 89.48 | 91.34 | 93.75 | 90.57 |
| M3 | 224 | 200 | 93.65 | 94.76 | 95.00 | 93.85 |
| M3 | 224 | 250 | 96.70 | 97.25 | 95.00 | 95.55 |
| M3 | 224 | 300 | 97.10 | 97.61 | 96.25 | 95.95 |
| M3 | 256 | 150 | 91.47 | 93.01 | 91.25 | 89.27 |
| M3 | 256 | 200 | 94.95 | 95.81 | 95.00 | 93.39 |
| M3 | 256 | 250 | 97.45 | 97.84 | 95.00 | 95.15 |
| M3 | 256 | 300 | 98.10 | 98.38 | 97.50 | 96.07 |
| S3-M3 | 224 | 150 | 86.17 | 88.59 | 86.25 | 82.82 |
| S3-M3 | 224 | 200 | 92.47 | 93.93 | 91.25 | 89.41 |
| S3-M3 | 224 | 250 | 95.55 | 96.31 | 93.75 | 92.90 |
| S3-M3 | 224 | 300 | 97.38 | 97.75 | 95.00 | 94.45 |
| S3-M3 | 256 | 150 | 86.12 | 88.73 | 92.50 | 83.88 |
| S3-M3 | 256 | 200 | 94.10 | 95.22 | 97.50 | 91.56 |
| S3-M3 | 256 | 250 | 96.53 | 97.19 | 98.75 | 94.16 |
| S3-M3 | 256 | 300 | 97.58 | 98.05 | 96.25 | 95.00 |

## GTA-UAV transfer

Both splits use positive-query filtering and the complete satellite gallery. They are benchmark split names; none of these models is trained on GTA. See the [evaluation protocol](evaluation.md#gta-uav) before comparing with GTA-trained systems.

| Model | Input | Split | R@1 | R@5 | AP | SDM@3 | DIS@1 (m) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| T0 | 224 | cross-area | 41.53 | 67.07 | 52.30 | 0.6905 | 571.78 |
| T0 | 224 | same-area | 39.53 | 62.50 | 49.29 | 0.6506 | 813.79 |
| T0 | 256 | cross-area | 40.23 | 65.59 | 51.03 | 0.6894 | 564.73 |
| T0 | 256 | same-area | 36.63 | 61.05 | 47.08 | 0.6457 | 813.52 |
| M3 | 224 | cross-area | 48.70 | 71.29 | 58.28 | 0.6524 | 630.48 |
| M3 | 224 | same-area | 46.06 | 67.24 | 54.86 | 0.6139 | 922.27 |
| M3 | 256 | cross-area | 49.65 | 71.23 | 58.71 | 0.6559 | 610.79 |
| M3 | 256 | same-area | 46.04 | 66.77 | 54.70 | 0.6180 | 881.44 |
| S3-M3 | 224 | cross-area | 40.81 | 62.20 | 49.93 | 0.5636 | 879.85 |
| S3-M3 | 224 | same-area | 38.34 | 57.42 | 46.52 | 0.5045 | 1287.43 |
| S3-M3 | 256 | cross-area | 43.48 | 63.09 | 51.99 | 0.5681 | 892.75 |
| S3-M3 | 256 | same-area | 40.37 | 58.15 | 48.16 | 0.5139 | 1282.56 |

In these records, cross-area retains **9,056** positive queries from 18,024 JSON entries; same-area retains **3,443** from 6,743. Each gallery contains **14,640** images. These counts describe the evaluated dataset copy; a different release must be identified rather than silently substituted.

## Run locations and checkpoint identity

Paths below are relative to src/checkpoint/. They identify local experiment artifacts, not public download links. Old absolute paths embedded in a JSON can reflect a directory rename; use the full recorded checkpoint SHA to resolve identity. No result or checkpoint metadata was edited to prepare this page.

| Model | Input | Run directory | Checkpoint SHA-256 |
| --- | ---: | --- | --- |
| T0 | 224 | `teacher/T0-224/` | `e1f26afbc3d93c21ec18b93a9f569810f51d245ab45a7774c93628eee2195197` |
| T0 | 256 | `teacher/T0-256/` | `36b98250ecba349b2d757b7079eeb31dfa4f22cd0d182d43a540c91bfa3e0730` |
| M3 | 224 | `middle_teacher/R224/M3-224/` | `79a04f82620ba1834a7d3bd05ff37a2ab1fc63dc94adf286d0268b2d1db612ba` |
| M3 | 256 | `middle_teacher/R256/M3-256/` | `8cfa4bc9ba0edd161024cebd43923906a517fd2546530d8c1c5b3ca2ee9e24e7` |
| S3-M3 | 224 | `student/R224/S3-224-M3/` | `ea217f59c9056c21c9f7f3b91a78d72e58b2ce13f6e4e630f349a0e8ccdfd8fe` |
| S3-M3 | 256 | `student/R256/S3-256-M3/` | `c81054a83d64110db2bc0f3a3635baaabd03abe9d7459a9d055db1c0003ac47e` |

## Result-file provenance

Each digest below is the SHA-256 of the source JSON as read for this documentation snapshot. Together with the checkpoint identities above, it traces rounded table values to exact records. Files are local artifacts, not bundled with the code release.

<details>
<summary>Source JSON SHA-256 digests</summary>

| Run | File | SHA-256 |
| --- | --- | --- |
| T0-224 | `test_1652.json` | `0fd950aeedb731eb03cde5b1fec8f30a27fe3bfa1e44ca2319b3d1dbbc1ace9f` |
| T0-224 | `test_sues200.json` | `33400455ed03abe2ac0426c9c41656d578f484785d62f1d1a44b1fcbdebdd9dc` |
| T0-224 | `test_gta_cross_area_d2s.json` | `93b657d742566c985482ba993d8429e4c6e17248e0d79f34faf7420bb369b25e` |
| T0-224 | `test_gta_same_area_d2s.json` | `a7037403ac13326043847579397f7296065bc24087c0a6846fd880c8ebdc490f` |
| T0-256 | `test_1652.json` | `a326c0f1bf45f73b3468eb9c9bc50cadc4a3ea7147b5499ff4b551e577e19930` |
| T0-256 | `test_sues200.json` | `9949db97cf75333f2a9b32bf8a4bbfdd25d3dcf992491620daad7f4efaffca1c` |
| T0-256 | `test_gta_cross_area_d2s.json` | `c93d07abe81629775ffd425cddd06c2dc2c7ab5595e35b78914ccbc25e2c0481` |
| T0-256 | `test_gta_same_area_d2s.json` | `4c45c025d6e1ae5d29f8461d2ee04bd31f5d443fd1c0b859f753f18f96f49127` |
| M3-224 | `test_1652.json` | `0a9e714cb86dc14bc63e629b0dd2facfee92cb9efed797f616de317b15b0d361` |
| M3-224 | `test_sues200.json` | `b24cb8cd640158cc975241934bf0bb99cce24022cd0287d4748925dc3bb8c0e5` |
| M3-224 | `test_gta_cross_area_d2s.json` | `09d191ced3637cec13854c84b28d89e2a882081ccb63ce61f49c7e36248d739d` |
| M3-224 | `test_gta_same_area_d2s.json` | `07eed0541902c3ae01fb062ad955b45b8db058062df5c2b8b6a8589e1ec1405b` |
| M3-256 | `test_1652.json` | `febc2b0007c5312c1e018d4ad7510ff470139366ac03187f3d78ae4dd3745335` |
| M3-256 | `test_sues200.json` | `484276876727e68049a9202e6c4cd5db36b2c48d7514e16c02a1930c4f66ff44` |
| M3-256 | `test_gta_cross_area_d2s.json` | `efcf319d5b42a1104b3e02a4948ad3552d7eb30ee9bb77aee467213aac987827` |
| M3-256 | `test_gta_same_area_d2s.json` | `8cad8bd2255e3ca91d7450fc6110363386b27fd2672fa5c402c770b20c36d97c` |
| S3-M3-224 | `test_1652.json` | `f29c7573e4e08c550c702bc2ec701d8e09d0713ee596e98025bbb818216ecb4b` |
| S3-M3-224 | `test_sues200.json` | `fb7f49998310f90ac2098800648647ff97d1a0a72e71d1c1098cf05a05ee952a` |
| S3-M3-224 | `test_gta_cross_area_d2s.json` | `1da3bca95ad9ab8e3c5b3620fe87e406cdcb7fe3d1c6fb4bd7c4b6e88f46f285` |
| S3-M3-224 | `test_gta_same_area_d2s.json` | `0087273573b78d6a861e046234bc14e8917de665e8b18a55d298293a9ee848d7` |
| S3-M3-256 | `test_1652.json` | `30093b9856f48a1e7045139c5ff48171f2b079a00d85968d29d03228609e1a85` |
| S3-M3-256 | `test_sues200.json` | `f6e82a7b415fb58b171c2cccb57ff249015641dbea4f7550b125d47f3322c630` |
| S3-M3-256 | `test_gta_cross_area_d2s.json` | `660af5dfd693cdb5b51d447870f1eb3570f405dbce82ac4dcf7130b7f0166642` |
| S3-M3-256 | `test_gta_same_area_d2s.json` | `cbb3c1b77874e456e0a307fa605021953211d72b061e90dc7f35b8c24c205cd6` |

</details>

## Remaining result release

After all five random-subspace runs and the direct Teacher control complete their requested evaluations, report every run, the aggregation definition, and the exact random/config identities. The original S3 and independent variants must be checked for initialization/data-order differences before describing the comparison as a single-factor control. No multi-run mean or standard deviation is claimed here.

Paper metadata, public checkpoint links and the project-level license remain release information to be supplied by the authors.
