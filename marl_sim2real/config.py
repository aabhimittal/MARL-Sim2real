"""Central configuration loaded from configs/*.yaml with sane defaults."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class PackingConfig:
    bin_size: tuple = (10, 10, 10)      # grid cells (W, D, H)
    cell_size: float = 0.05             # metres per cell
    max_items: int = 12
    min_item_dim: int = 2               # cells
    max_item_dim: int = 5               # cells


@dataclasses.dataclass
class PhysicsConfig:
    gravity: float = -9.81
    sim_steps: int = 120                # steps to settle a placed item
    timestep: float = 1.0 / 240.0
    max_displacement: float = 0.02      # metres before "unstable"
    max_tilt_deg: float = 10.0


@dataclasses.dataclass
class TrainConfig:
    episodes: int = 200
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    seed: int = 7


@dataclasses.dataclass
class GNNConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    lr: float = 1e-3
    node_feat_dim: int = 4
    edge_feat_dim: int = 3


@dataclasses.dataclass
class DriftConfig:
    k_sigma: float = 3.0
    window_size: int = 30               # rolling window of real observations
    min_samples: int = 10               # before drift can trigger
    recalib_steps: int = 50             # gradient steps during self-correction
    bias_lr: float = 0.1                # lr for bias-only recalibration
    ema_alpha: float = 0.2              # per-edge bias EMA


@dataclasses.dataclass
class RouterConfig:
    low_threshold: float = 0.33
    high_threshold: float = 0.66
    models: tuple = (
        "claude-haiku-4-5-20251001",    # simple tasks
        "claude-sonnet-5",              # moderate tasks
        "claude-opus-4-8",              # complex tasks
    )
    # rough $/MTok (input, output) used for savings estimates
    pricing: tuple = ((1.0, 5.0), (3.0, 15.0), (15.0, 75.0))


@dataclasses.dataclass
class Config:
    packing: PackingConfig = dataclasses.field(default_factory=PackingConfig)
    physics: PhysicsConfig = dataclasses.field(default_factory=PhysicsConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    gnn: GNNConfig = dataclasses.field(default_factory=GNNConfig)
    drift: DriftConfig = dataclasses.field(default_factory=DriftConfig)
    router: RouterConfig = dataclasses.field(default_factory=RouterConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        data = yaml.safe_load(Path(path).read_text()) or {}
        for section, values in data.items():
            target = getattr(cfg, section, None)
            if target is None or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if hasattr(target, key):
                    if isinstance(value, list):
                        value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
                    setattr(target, key, value)
        return cfg
