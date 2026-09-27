"""RNG replay for two-pass E3 SAM."""
import random
import numpy as np
import torch

def capture_rng_state():
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(), "numpy": np.random.get_state()}

def restore_rng_state(state):
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
