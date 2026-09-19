"""Bind a new config to explicit generated assets; never edit historical metadata."""
import argparse,json
from pathlib import Path
from .artifacts import file_sha256
from .core_config import validate_config,assert_assets

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',required=True);p.add_argument('--output-config',required=True)
    for name in ('middle-checkpoint','middle-run-config','asset-output','original-asset','calibration','student-pretrained','run-output'):p.add_argument('--'+name)
    args=p.parse_args(argv);cfg=json.loads(Path(args.template).read_text())
    mapping={'middle_checkpoint':'middle_checkpoint','middle_run_config':'middle_config','asset_output':'stst_asset','original_asset':'original_stst_asset','calibration':'p2_calibration_path','student_pretrained':'student_pretrained','run_output':'output_dir'}
    for arg,key in mapping.items():
        if getattr(args,arg):cfg[key]=str(Path(getattr(args,arg)).resolve())
    hashes={'middle_checkpoint':'middle_checkpoint_sha256','middle_config':'middle_config_sha256','stst_asset':'extended_stst_asset_sha256','original_stst_asset':'original_stst_asset_sha256','p2_calibration_path':'p2_calibration_sha256','student_pretrained':'student_pretrained_sha256'}
    for key,target in hashes.items():
        if cfg.get(key):cfg[target]=file_sha256(cfg[key])
    validate_config(cfg,check_assets=True)
    output=Path(args.output_config);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:json.dump(cfg,handle,indent=2)
if __name__=='__main__':main()
