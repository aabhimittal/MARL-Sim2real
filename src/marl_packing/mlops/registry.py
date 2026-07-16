"""Local JSON model registry.

Every trained (packer, physics) checkpoint pair is a "version"; the registry additionally
tracks which version is the current `champion`. This is a deliberately thin stand-in for a
real model registry (e.g. the MLflow Model Registry, or a cloud artifact store) -- the
interface below (add/get/list/set_champion) is the seam that would need to be
re-implemented against a real backend. Everything else in `mlops/` only talks to this
interface, not to the JSON file directly.

Not safe for concurrent writers (no file locking) -- fine for the local, single-process
training/eval workflow this repo targets.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class ModelVersion:
    version: str  # e.g. "v3"
    created_at: str  # ISO-8601 timestamp, supplied by the caller
    packer_path: str
    physics_path: str
    train_config_hash: str
    metrics: dict | None = None  # populated later by mlops/challenger.py's evaluation

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelVersion:
        return cls(**d)


class ModelRegistry:
    def __init__(self, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "registry.json"
        if not self._index_path.exists():
            self._write_index({"champion": None, "versions": {}})

    def _read_index(self) -> dict:
        with open(self._index_path) as f:
            return json.load(f)

    def _write_index(self, index: dict) -> None:
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2)

    def next_version_id(self) -> str:
        index = self._read_index()
        existing = [int(v[1:]) for v in index["versions"] if v.startswith("v") and v[1:].isdigit()]
        return f"v{max(existing, default=0) + 1}"

    def add_version(self, version: ModelVersion) -> None:
        index = self._read_index()
        index["versions"][version.version] = version.to_dict()
        self._write_index(index)

    def update_metrics(self, version_id: str, metrics: dict) -> None:
        index = self._read_index()
        if version_id not in index["versions"]:
            raise KeyError(f"Unknown model version {version_id!r}")
        index["versions"][version_id]["metrics"] = metrics
        self._write_index(index)

    def get_version(self, version_id: str) -> ModelVersion:
        index = self._read_index()
        if version_id not in index["versions"]:
            raise KeyError(f"Unknown model version {version_id!r}")
        return ModelVersion.from_dict(index["versions"][version_id])

    def list_versions(self) -> list[ModelVersion]:
        index = self._read_index()
        return [ModelVersion.from_dict(v) for v in index["versions"].values()]

    def get_champion(self) -> ModelVersion | None:
        index = self._read_index()
        champion_id = index.get("champion")
        return self.get_version(champion_id) if champion_id else None

    def set_champion(self, version_id: str) -> None:
        index = self._read_index()
        if version_id not in index["versions"]:
            raise KeyError(f"Unknown model version {version_id!r}")
        index["champion"] = version_id
        self._write_index(index)

    def version_dir(self, version_id: str) -> Path:
        path = self.registry_dir / version_id
        path.mkdir(parents=True, exist_ok=True)
        return path
