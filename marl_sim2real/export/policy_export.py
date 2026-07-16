"""Export trained agents as a deployable edge bundle.

The bundle is a single ``.npz`` (weights) + ``.json`` (manifest) pair — no
pickle, no framework dependency — so the EdgePack repo (or any device runtime)
can load it with nothing but numpy.  The manifest carries the metrics needed
by the Edge MLOps registry: reality-gap numbers, bin geometry, and a content
hash for integrity checks during OTA rollout.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..agents.physics_agent import PhysicsAgent
from ..agents.proposer import ProposerAgent
from ..sim2real.bridge import BridgeResult

FORMAT_VERSION = 1


def export_bundle(
    out_dir: str | Path,
    proposer: ProposerAgent,
    physics: PhysicsAgent,
    bin_size: tuple[int, int, int],
    bridge_result: BridgeResult | None = None,
    name: str = "packing_policy",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    weights_path = out / f"{name}.npz"
    arrays: dict[str, np.ndarray] = {}
    for prefix, params in (("proposer", proposer.get_params()), ("physics", physics.get_params())):
        for k, v in params.items():
            arrays[f"{prefix}.{k}"] = v
    np.savez_compressed(weights_path, **arrays)

    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "created_unix": int(time.time()),
        "bin_size": list(bin_size),
        "weights_file": weights_path.name,
        "weights_sha256": digest,
        "metrics": {},
    }
    if bridge_result is not None:
        manifest["metrics"] = {
            "sim_utilization": bridge_result.post_gap.sim_utilization,
            "real_utilization": bridge_result.post_gap.real_utilization,
            "reality_gap": bridge_result.post_gap.gap,
            "calibration_agreement": bridge_result.calibration.agreement,
            "accepted": bridge_result.accepted,
        }
    manifest_path = out / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def load_bundle(manifest_path: str | Path) -> tuple[ProposerAgent, PhysicsAgent, dict]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format: {manifest.get('format_version')}")

    weights_path = manifest_path.parent / manifest["weights_file"]
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if digest != manifest["weights_sha256"]:
        raise ValueError("weights hash mismatch — bundle corrupted or tampered")

    data = np.load(weights_path)
    proposer_params = {k.split(".", 1)[1]: data[k] for k in data.files if k.startswith("proposer.")}
    physics_params = {k.split(".", 1)[1]: data[k] for k in data.files if k.startswith("physics.")}

    from ..agents.networks import MLP

    p_sizes = [int(s) for s in proposer_params["sizes"]]
    proposer = ProposerAgent(obs_size=p_sizes[0], num_actions=p_sizes[-1])
    proposer.net = MLP.from_params(proposer_params)

    f_sizes = [int(s) for s in physics_params["sizes"]]
    physics = PhysicsAgent(obs_size=f_sizes[0] - 5)
    physics.net = MLP.from_params(physics_params)
    return proposer, physics, manifest
