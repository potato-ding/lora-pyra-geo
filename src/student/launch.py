"""Validate and launch only the formal S3 R224/R256 training chain."""
import argparse
import json
from pathlib import Path
from .core_config import validate_config

def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(argv)
    cfg=load_config(args.config)
    if args.validate_only:
        print(json.dumps(cfg,indent=2));return
    from .formal_launch import launch
    return launch(cfg,args.config)

if __name__=='__main__':main()
