"""Export a promoted model version as a self-contained, integrity-checked deployment
bundle.

`mlops/deploy.py` promoting a challenger only flips a pointer in the local JSON registry --
useful for this repo's own gate, but not something a real serving system can consume
directly. `export_bundle()` is that handoff artifact: the version's SB3 checkpoints plus a
manifest carrying its evaluation metrics and a sha256 integrity hash per file, so the
bundle can be verified before being loaded into whatever pushes it to production (an
inference server, an OTA update, an edge device).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from marl_packing.mlops.registry import ModelVersion

FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_bundle(version: ModelVersion, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    packer_dst = out / "packer.zip"
    physics_dst = out / "physics.zip"
    packer_dst.write_bytes(Path(version.packer_path).read_bytes())
    physics_dst.write_bytes(Path(version.physics_path).read_bytes())

    manifest = {
        "format_version": FORMAT_VERSION,
        "version": version.version,
        "created_at": version.created_at,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "train_config_hash": version.train_config_hash,
        "metrics": version.metrics,
        "files": {
            "packer": {"filename": packer_dst.name, "sha256": _sha256(packer_dst)},
            "physics": {"filename": physics_dst.name, "sha256": _sha256(physics_dst)},
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def verify_bundle(manifest_path: str | Path) -> bool:
    """Recomputes each file's sha256 against the manifest -- catches corruption or
    tampering before a bundle is loaded into a serving system."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format: {manifest.get('format_version')}")
    for entry in manifest["files"].values():
        file_path = manifest_path.parent / entry["filename"]
        if not file_path.exists() or _sha256(file_path) != entry["sha256"]:
            return False
    return True
