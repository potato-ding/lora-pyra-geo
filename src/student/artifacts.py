"""Student deployment-state and file identity utilities."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_valid_run(run):
    if (Path(run) / "INVALIDATED.json").exists():
        raise ValueError("INVALIDATED Student run cannot issue formal results: " + str(run))


def deployment_state_dict(model):
    raw = getattr(model, "module", model)
    raw = getattr(raw, "student", raw)
    state = {name: tensor.detach().cpu() for name, tensor in raw.state_dict().items()}
    if any(any(token in name.lower() for token in (
        "stst", "projector", "middle", "allocation_gate",
    )) for name in state):
        raise ValueError("Training-only distillation state leaked into deployment")
    return state
