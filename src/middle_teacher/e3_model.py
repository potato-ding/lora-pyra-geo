"""Construct the formal E3 Middle while preserving P0 initialization RNG."""
import copy
import torch
from src.middle_teacher.model import build_middle_teacher
from src.middle_teacher.losses.adaptive_bridge_v2 import AdaptiveBridgeV2Bank


def build_stage3_model(config):
    base=copy.deepcopy(config)
    base['distillation']={'base_loss':'pair_infonce'}
    model=build_middle_teacher(base)
    component=config['distillation']['adaptive_bridge_v2']
    with torch.random.fork_rng(devices=[]):
        bank=AdaptiveBridgeV2Bank(component)
    model.layer_semantic_projectors=bank
    model.bridge_config=component
    return model
