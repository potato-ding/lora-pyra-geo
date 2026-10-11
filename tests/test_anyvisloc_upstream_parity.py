"""Protect parity with UAV-AVL/Benchmark retrieval-only at the pinned commit.

Expected function hashes come from independently downloaded upstream sources,
not from the local implementation. No network is used when running tests.
"""
import ast
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from src.evaluation import anyvisloc
from src.evaluation import anyvisloc_official as official

UPSTREAM_COMMIT = "6c8f7980f35defd0676d82e71b19fcd214da4033"
# avl_utils.py; Retrieval_Models/{feature_extract,multi_model_loader}.py;
# run_avl_retrieval_only.py, all at UPSTREAM_COMMIT.
UPSTREAM_FUNCTION_SHA256 = {'dumpRotateImage': 'a5b6944c87164f8c7885650dab00b40e3e7bf48b95dabb283c8876bb9e95feec', 'compute_block_mid_wo_black': '297ef2d4fa6b360c6a8cd48d6ad75e990f2633ea028de1d8f31735d77f4c3d43', 'ensure_dir': 'fb59eed4554cbb1bf3dc27c4e774e5aeb5d66244f76d834cd2b2cdb84d604bff', 'to_numpy': '2d43c6fe2a91059f2fc4a2ef2774dfafaafd6ea49720bfc81830334cc83edcf7', 'to_python_str': 'ac091ea8c7672e4159cac2f8955e95a227a5c82100f79d7f4460501e8de5a77e', 'normalize_uint8': '1d1e4c2d0badc86bf9710bc73573a18e955031c66eb0240527e90b085390d38c', 'ensure_bgr_uint8': '929e3064745ddd19c0064295751564f5822c2e41858e3be9a61fab24332cef28', 'as_vec2': '6d382f441d49bde03af3c3ae0d3cdb22ff2ea33c730aa80d0de310f9dd4b410b', 'tensor_rgb_to_bgr_uint8': 'aeb059a31bfedf0037e7232b758e9e1660e0d5694e4bf7e4fcb0c27a8a0edca0', 'crop_center_from_bgr': '2f3245669af3bc29f106deb9207bc3a4ee9ae8e02615f30771aa9238831ec513', 'anyvisloc_sample_to_truepos': '752d855a7bf52bbf532cc25f45571930e78ab6b2bbb913b7a2cc2ae99f875071', '_even_ceil': 'f21782a83595f99d910243b69a20e92c651d923c4be723ab67c1ee93b5963b12', 'resolution_size_anyvisloc': 'a0dde3b6cd12c578b51d55ed55a6eb4568aacc55cc4c5c5a2d9a531e5b5a4130', '_apply_affine_xy': '3e6359d9c1c59f770791a8a60033810de9ce931a67f3edbd37c04f2bd18699a9', 'view_center_anyvisloc': 'b06ee0f338e66b5cff31cb074e371223c183a6cb271800ebff62f1318a2c809f', '_sanitize_filename_token': '2bc52330a9edde09e31b34e488647b0b08e5f4e23208d8927dc1226a065566e6', '_circled_number': 'ffd60adef8b985e3de02ab65bc4aa5bfd7c98ceb59d2f0448fd50f4f8f0d41f4', '_draw_red_cross_if_inside': 'e5831f292f59a27583e067f3a537560d778e04e2b49c2f0a9d93f40998ceebc8', '_draw_cross_at_image_center': 'c3d4c64e101d57f48e465ad6ae8700ea8966a72e6a451f116e19584ea1dd904c', '_make_ref_feature_cache_meta': 'acf4ea5934214ec9830d3ab5d1213034715f072afa4e3138b6c79175b72822ed', '_default_ref_feature_cache_root': 'e9e7807d11ac53012fd064ff0d6cfb86330fc3de5e757068d7e5cad28f0a6bc4', '_make_ref_feature_cache_path': '6e42ccd4c2255759f9deebcee287be1f62a47ca489b5de694e95a668da3b2e7a', '_load_cached_ref_features': 'b9d775b7c0935d0728fe08e7e1f9ae21a36f937c3a9459340b5643fbeeb5adbe', '_save_cached_ref_features': '87160d9b96174e3971358b4309e5a61213d0b60f68434d56f90cc3d0e9a967aa', '_extract_query_feature_with_dummy_gallery': '05a61cbe50c0a30829a3a8f01dab839afd4d34b008241eb26d1215433fce1da2', 'retrieval_all_anyvisloc': '6bb65a6c5f75c7f8be826c0a33d1aceeaeea314ae8a104bbe64335d906b2f96b', 'extract_features': '0e2f55c1a156cf621d58bbea3ef9875d5ed5b334eb47d60130f4d7f26bc9f4bf', 'feature_fusion_all': 'dad881f4b30219f1073f4bb2cc0126b95874183354de96599f17dc18fa3cb991', 'get_transforms_new': '8df0b9796b5ab1e80190f93ddafcfb6677cd58e0ff4ee96fe774aaeed43ab655', 'safe_float': 'c9b94c987cb3d072f4e0ca146cf37dc69a63f10f9a7015dcbf85f678fa42ff55', 'retrieval_metrics_from_ratios': 'cbc2eb88188fd3c023ce8deec0161a59748f798655a12dd1b71ef3cdd284edc0', 'retrieval_metrics_for_ks': '925d0b9b09e85dcb22fbcea3b76349346679373cf5107c38b4f65a5dda0aafe2', 'save_json': '1b5ebcde8ac04d2f948b58eb436d6147f9c3835d67d1cb768bae4b9c56595916', 'dsm_to_2d': 'e1c346f4ef25da12f349781951e0e22e5044a58748b32bf673c1c3857905ea3d', '_resolve_ref_json_path': '27224f0100500075beb56eac69ee6de4a07739698c7f349c9142b9e5ccae27f8', '_load_reference_json': '3ae69a73728c5df8d0088bd0ceee32ffe765c261fc32a96a1d5e9c9b2c7d153f', '_json_path_to_abs': '0ef525b30a3667484f9f7718ed5050b0bd8d425a2c9075d1c7413c99e7c21be7', '_load_ref_array': 'df40ff0867aa462ac7f19148d037e38bcbcb647265e29b48c5244561f3c0f81a', 'normalize_reference_mode': '50183ba09dd80d275568cb1d2826d66747dd951cbdd97c5903eb9f938374843e', '_reference_view_from_json': 'ee407a2e1fb3c33ffbdc2c3fb0ba7e34a67c2474bc82be7fc905370d8c9b2e95', 'build_anyvisloc_reference_view': '4d1c7609efe716ee03b8fd5dd41672a6cfe0403af38822b4b62fcdd5faeaa3e6', 'as_text': '1c21a96338015bd09419c7d1759ff0207a8dab1fcd110029755f43cccba8ec55', 'process_one_sample': '32762618421a0f9c84b81623fc208622921655a01d5977c9d42d59176d5dc78e', 'summarize_retrieval': '01375f06e881c3abb2aecca26c4c82783b032dcb39d39f26db24d82f5fc13d1c'}


def test_embedded_functions_match_independent_pinned_upstream():
    assert official.UPSTREAM_COMMIT == UPSTREAM_COMMIT
    source = Path(official.__file__).read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest() for n in nodes}
    assert actual == UPSTREAM_FUNCTION_SHA256


def test_odd_query_crop_and_default_pillow_resize_are_preserved():
    # Odd requested side 101 is actually cropped to 100 by upstream //2 bounds.
    rgb = (np.arange(101 * 153 * 3, dtype=np.int64) % 256).astype(np.uint8).reshape(101, 153, 3)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).contiguous()
    bgr = official.tensor_rgb_to_bgr_uint8(tensor)
    crop = official.crop_center_from_bgr(bgr, 101, 101)
    assert np.array_equal(np.asarray(crop), rgb[0:100, 26:126])
    assert crop.size == (100, 100)
    assert np.array_equal(np.asarray(crop.resize((224, 224))),
                          np.asarray(crop.resize((224, 224), Image.Resampling.BICUBIC)))


def test_black_filter_uses_first_channel_and_preserves_duplicate_edges():
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    assert official.compute_block_mid_wo_black(bgr, [20, 20], [10, 10]).size == 0
    bgr[:, :, 0] = 1
    mids = official.compute_block_mid_wo_black(bgr, [20, 20], [10, 10])
    assert mids.shape == (100, 2)
    assert len(np.unique(mids, axis=0)) < len(mids)


def test_recall_uses_nearest_candidate_rank_without_absolute_distance_threshold():
    close = anyvisloc.retrieval_metrics([0.1, 0.05, 3.0])
    far = anyvisloc.retrieval_metrics([10.0, 11.0, 12.0])
    assert close['Recall@1'] == 0.0
    assert close['retrieval_gt_rank'] == 2
    assert far['Recall@1'] == 1.0
    assert far['retrieval_gt_rank'] == 1
    # Equal minimum ratios accept the first such tile in retrieval order.
    assert anyvisloc.retrieval_metrics([1.0, 0.1, 0.1])['retrieval_gt_rank'] == 2


@pytest.mark.parametrize('kind', ['camp', 'dac', 'mean'])
def test_comparison_adapter_uses_raw_penultimate_descriptor_then_official_fusion(kind):
    class Raw(nn.Module):
        def forward(self, x):
            return (torch.zeros(len(x), 2), torch.tensor([[3., 4.]]).expand(len(x), -1), torch.ones(len(x), 7))
    encoder = nn.Module()
    encoder.model = Raw()
    encoder.kind = kind
    wrapped = anyvisloc._OfficialRetrievalModel(encoder, torch.device('cpu'))
    raw_outputs = wrapped(torch.ones(2, 3, 8, 8))
    selected = official.feature_fusion_all({'retrieval_method': wrapped.fusion}, raw_outputs)
    assert torch.equal(selected, torch.tensor([[3., 4.], [3., 4.]]))
    assert torch.equal(torch.nn.functional.normalize(selected, dim=-1), torch.tensor([[.6, .8], [.6, .8]]))


def test_overall_aggregation_is_weighted_by_queries_not_equal_scene_weights():
    rows = []
    for value in [1., 1., 1., 0.]:
        rows.append({**{f'{name}@{k}': value for name in ['Recall', 'PDM'] for k in [1, 3, 5]},
                     'mean_unused': 0., 'retrieval_time_per_feature_ms': 1.})
    result = anyvisloc._summarize(rows)
    assert result['queries'] == 4
    assert result['Recall@1'] == .75


UPSTREAM_DATA_AST_SHA256 = {'AnyVisLocNPZDataset': '0bfe21f6c21f4637f3a307544e4997e29fb05f152d6d776528c0c0a97f633a2f',
 '_as_string': '7f861d390821c81477ef878c6ec7fd806c62d021a4067f967de41ea1ff1484da',
 '_load_npz': 'b25d85e4747c42f243235b6e1853a9ac614a329cab99ec0ded6b69ad99ef2288',
 '_normalize_scenes': '4e1de2a24d2a0632a66c806966bdab4d37293a138b6abef4bbe9c7c7d26c4617',
 '_read_image_from_relpath': '133b8dd2ab7bd0825c1064adf0bc2c35582f0a82569946b461fe5d9e36456f8c'}


def test_official_dataset_class_and_helpers_are_unmodified():
    from src.evaluation import anyvisloc_data
    nodes = [n for n in ast.parse(Path(anyvisloc_data.__file__).read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest() for n in nodes}
    assert actual == UPSTREAM_DATA_AST_SHA256
