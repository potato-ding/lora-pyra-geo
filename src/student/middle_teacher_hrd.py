"""M3-to-Student HRD control using the same bidirectional HRD/SAM adapter."""
from .direct_teacher_hrd import HRD_PROTOCOL, sam_hrd_adsd_backward


def validate_config(cfg):
    if (cfg.get("teacher_provider") is not None
            or cfg.get("middle_provider") != "M3"
            or cfg.get("experiment_id") != "S3-SAM-ADSD-R256"
            or cfg.get("img_size") != 256
            or "subspace_reference" not in cfg
            or cfg.get("hrd") != HRD_PROTOCOL):
        raise ValueError("M3 HRD + ADSD control protocol mismatch")
