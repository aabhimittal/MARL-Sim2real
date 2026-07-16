"""YAML config loading. Every module in this repo takes plain dicts loaded via load_config
rather than bespoke config classes, so configs stay diffable YAML and don't require code
changes to tune."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
