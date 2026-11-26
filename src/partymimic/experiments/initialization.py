"""Opt-in component seeds for paired online experiments; legacy RNG is unchanged."""
from contextlib import contextmanager

import torch

OFFSETS = {"wm": 1000003, "pi": 2000003, "critic": 3000017, "runtime": 4000037}


def stream_seeds(seed):
    return {name: seed + offset for name, offset in OFFSETS.items()}


@contextmanager
def model_rng(seed, component, enabled=False):
    if not enabled:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(stream_seeds(seed)[component])
        yield


def check_initial_reference(modules, path):
    from .runtime import state_hash
    reference = torch.load(path, map_location="cpu", weights_only=False)
    for name in ("pi", "critic"):
        class State:
            def state_dict(self):
                return reference[name]
        if state_hash({name: modules[name]}) != state_hash({name: State()}):
            raise ValueError(f"Paired initial {name} differs; stopped before training")
