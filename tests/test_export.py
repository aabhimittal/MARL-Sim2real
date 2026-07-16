"""Tests for mlops/export.py's deployment bundle export/verify round-trip."""

from __future__ import annotations

from marl_packing.mlops.export import export_bundle, verify_bundle
from marl_packing.mlops.registry import ModelRegistry, ModelVersion


def _make_version(tmp_path, registry: ModelRegistry) -> ModelVersion:
    version_dir = registry.version_dir("v1")
    (version_dir / "packer.zip").write_bytes(b"fake-packer-weights")
    (version_dir / "physics.zip").write_bytes(b"fake-physics-weights")
    version = ModelVersion(
        version="v1",
        created_at="2026-01-01T00:00:00+00:00",
        packer_path=str(version_dir / "packer.zip"),
        physics_path=str(version_dir / "physics.zip"),
        train_config_hash="abc123",
        metrics={"sim_utilization_pct": 50.0},
    )
    registry.add_version(version)
    return version


def test_export_bundle_writes_manifest_and_files(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    version = _make_version(tmp_path, registry)

    manifest_path = export_bundle(version, tmp_path / "bundle")

    assert manifest_path.exists()
    assert (tmp_path / "bundle" / "packer.zip").read_bytes() == b"fake-packer-weights"
    assert (tmp_path / "bundle" / "physics.zip").read_bytes() == b"fake-physics-weights"


def test_verify_bundle_passes_for_untampered_bundle(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    version = _make_version(tmp_path, registry)
    manifest_path = export_bundle(version, tmp_path / "bundle")

    assert verify_bundle(manifest_path) is True


def test_verify_bundle_fails_for_tampered_file(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    version = _make_version(tmp_path, registry)
    manifest_path = export_bundle(version, tmp_path / "bundle")

    (tmp_path / "bundle" / "packer.zip").write_bytes(b"tampered-content")

    assert verify_bundle(manifest_path) is False


def test_verify_bundle_fails_for_missing_file(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    version = _make_version(tmp_path, registry)
    manifest_path = export_bundle(version, tmp_path / "bundle")

    (tmp_path / "bundle" / "physics.zip").unlink()

    assert verify_bundle(manifest_path) is False


def test_manifest_carries_version_metadata(tmp_path):
    import json

    registry = ModelRegistry(tmp_path / "registry")
    version = _make_version(tmp_path, registry)
    manifest_path = export_bundle(version, tmp_path / "bundle")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == "v1"
    assert manifest["train_config_hash"] == "abc123"
    assert manifest["metrics"] == {"sim_utilization_pct": 50.0}
    assert set(manifest["files"].keys()) == {"packer", "physics"}
