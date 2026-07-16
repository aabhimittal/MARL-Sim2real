"""Domain randomization profile selection -- the seam between training-time robustness and
the "real-world proxy" used to measure the Sim2Real gap.

Two profiles live in configs/env.yaml under `domain_randomization`:
  - "train": a narrow randomization range the agents train against.
  - "real_proxy": a wider range standing in for unmodeled real-world variation (since this
    project has no physical rig to benchmark against -- see docs/SIM2REAL.md). Only used by
    evaluation/benchmark.py, never during training.
"""

VALID_MODES = ("train", "real_proxy")


def get_randomization_profile(env_config: dict, mode: str) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown domain randomization mode {mode!r}, expected one of {VALID_MODES}")
    return env_config["domain_randomization"][mode]
