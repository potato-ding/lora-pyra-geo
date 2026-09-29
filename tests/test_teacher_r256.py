import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.training.teacher.formal_config import parse_args
from src.training.teacher.artifacts import is_formal_teacher, checkpoint_metadata, save_best_checkpoint
from src.utils.save_path import get_save_pth

def test_r256_config_matches_canonical_four_gpu_t0():
    configs = {
        size: json.loads(Path(f"configs/teacher/t0_{size}.json").read_text())
        for size in (224, 256)
    }
    for size, config in configs.items():
        args = parse_args(["--config", f"configs/teacher/t0_{size}.json"])
        assert args.experiment_id == f"T0-INFONCE-R{size}"
        assert args.img_size == size
        assert args.batch_size * 4 * args.grad_accum_steps == 32
        assert args.epochs == 10 and args.val_batch_size == 16
        assert args.init_checkpoint is None
        assert is_formal_teacher(args)
        assert (args.lora_rank, args.lora_alpha, args.lora_dropout) == (8, 16, .1)
        assert args.lora_target_names == "qkv,proj"
        assert (args.lora_start_block, args.lora_end_block,
                args.full_finetune_start_block, args.full_finetune_end_block) == (20, 36, 36, 40)
    left = {key: value for key, value in configs[224].items()
            if key not in ("img_size", "experiment_id", "output_dir")}
    right = {key: value for key, value in configs[256].items()
             if key not in ("img_size", "experiment_id", "output_dir")}
    assert left == right

@pytest.mark.parametrize('size',[224,256])
def test_formal_resolution_no_sidecars(tmp_path,size):
    args=SimpleNamespace(experiment_id=f'T0-INFONCE-R{size}',img_size=size)
    assert is_formal_teacher(args)
    assert list(tmp_path.iterdir())==[]

def test_r256_real_patch_embed_geometry():
    import sys
    sys.path.insert(0,str(Path('src/models/dinov3_main').resolve()))
    from dinov3.layers.patch_embed import PatchEmbed
    with torch.no_grad():
        layer=PatchEmbed(img_size=224,patch_size=16,in_chans=3,embed_dim=16)
        output=layer(torch.zeros(1,3,256,256))
    assert output.numel()==256*16
