import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.training.teacher.args import parse_args
from src.training.teacher.artifacts import is_formal_teacher, checkpoint_metadata, save_best_checkpoint
from src.utils.save_path import get_save_pth

def test_r256_config_matches_canonical_four_gpu_t0():
    cfg=json.loads(Path('configs/teacher/t0_certified_256.json').read_text())
    args=parse_args([v for k,x in cfg.items() for v in ('--'+k,str(x))])
    for size in (224,384):
        old=json.loads(Path(f'configs/teacher/t0_certified_{size}.json').read_text())
        # Historical templates use 8 ranks x 4 pairs. Formal 4-rank T0 uses 8 pairs.
        old['batch_size']=8
        old['weak_paired_cross_view_weight']=old.pop('weak_sample4geo_weight')
        old['img_size']=256
        assert all(cfg[k]==v for k,v in old.items())
    assert args.experiment_id=='T0-INFONCE-R256' and args.img_size==256
    assert args.batch_size*4*args.grad_accum_steps==32 and args.epochs==10
    assert args.val_batch_size==8 and args.init_checkpoint is None
    assert get_save_pth(args)=='/home/dingyi/lora-pyra-geo/src/checkpoint/teacher/R256/T0-INFONCE-R256'
    assert is_formal_teacher(args)
    assert (args.lora_rank,args.lora_alpha,args.lora_dropout)==(8,16,.1)
    assert args.lora_target_names=='qkv,proj'
    assert (args.lora_start_block,args.lora_end_block,args.full_finetune_start_block,args.full_finetune_end_block)==(20,36,36,40)

@pytest.mark.parametrize('size',[224,256,384])
def test_formal_resolution_no_sidecars(tmp_path,size):
    from src.training.teacher.hparams import save_training_record
    args=SimpleNamespace(experiment_id=f'T0-INFONCE-R{size}',img_size=size)
    assert is_formal_teacher(args)
    save_training_record(str(tmp_path),args,[],None,0)
    assert list(tmp_path.iterdir())==[]

def test_r256_real_patch_embed_geometry():
    import sys
    sys.path.insert(0,str(Path('src/models/dinov3_main').resolve()))
    from dinov3.layers.patch_embed import PatchEmbed
    with torch.no_grad():
        layer=PatchEmbed(img_size=224,patch_size=16,in_chans=3,embed_dim=16)
        output=layer(torch.zeros(1,3,256,256))
    assert output.numel()==256*16
