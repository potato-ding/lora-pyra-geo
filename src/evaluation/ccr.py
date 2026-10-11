"""CCR checkpoint-bundled source adapter for the shared four-dataset protocol."""
import importlib.util
from pathlib import Path
from src.middle_teacher.checkpoint import sha256

def load_ccr_comparison(base,device):
    path=base/'source/compatibility_adapter.py'
    spec=importlib.util.spec_from_file_location('ccr_checkpoint_adapter',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder,audit,size=module.load_ccr(device)
    audit.update(model='ccr',training_dataset='University-1652',
        upstream_commit='39d9860cbbc3db4eac3bc3579aa3a40472d679ac',
        adapter_code_sha256=sha256(path),
        descriptor_branches=['normal part 1','normal part 2','global'],
        precision='FP32 parameters and forward; FP32 normalization and similarity',
        horizontal_flip=False,multiscale=False,
        protocol_note='Shared single-forward comparison; author test script uses horizontal-flip accumulation')
    return encoder,audit,size
