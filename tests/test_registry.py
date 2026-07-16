"""Pure unit tests for marl_packing.mlops.registry using a tmp_path-backed registry.

Never touches the real models/registry/ directory.
"""

import pytest

from marl_packing.mlops.registry import ModelRegistry, ModelVersion


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


def make_version(version_id: str, metrics: dict | None = None) -> ModelVersion:
    return ModelVersion(
        version=version_id,
        created_at="2026-07-16T00:00:00+00:00",
        packer_path=f"models/{version_id}/packer.zip",
        physics_path=f"models/{version_id}/physics.zip",
        train_config_hash="abc123",
        metrics=metrics,
    )


class TestFreshRegistry:
    def test_starts_with_no_champion(self, registry):
        assert registry.get_champion() is None

    def test_starts_with_no_versions(self, registry):
        assert registry.list_versions() == []

    def test_first_version_id_is_v1(self, registry):
        assert registry.next_version_id() == "v1"


class TestAddAndGet:
    def test_add_version_get_version_round_trip(self, registry):
        original = make_version("v1")
        registry.add_version(original)
        fetched = registry.get_version("v1")
        assert isinstance(fetched, ModelVersion)
        assert fetched == original

    def test_list_versions_after_adds(self, registry):
        v1, v2 = make_version("v1"), make_version("v2")
        registry.add_version(v1)
        registry.add_version(v2)
        listed = registry.list_versions()
        assert len(listed) == 2
        assert v1 in listed and v2 in listed

    def test_next_version_id_increments(self, registry):
        assert registry.next_version_id() == "v1"
        registry.add_version(make_version("v1"))
        assert registry.next_version_id() == "v2"
        registry.add_version(make_version("v2"))
        assert registry.next_version_id() == "v3"
        registry.add_version(make_version("v3"))
        assert registry.next_version_id() == "v4"


class TestUpdateMetrics:
    def test_update_metrics_attaches_to_right_version(self, registry):
        registry.add_version(make_version("v1"))
        registry.add_version(make_version("v2"))
        metrics = {"box_utilization_pct": 71.2, "stability_success_rate_pct": 95.0}
        registry.update_metrics("v2", metrics)
        assert registry.get_version("v2").metrics == metrics
        assert registry.get_version("v1").metrics is None


class TestUnknownVersionErrors:
    def test_get_version_unknown_raises_keyerror(self, registry):
        with pytest.raises(KeyError):
            registry.get_version("v99")

    def test_update_metrics_unknown_raises_keyerror(self, registry):
        with pytest.raises(KeyError):
            registry.update_metrics("v99", {"box_utilization_pct": 1.0})

    def test_set_champion_unknown_raises_keyerror(self, registry):
        with pytest.raises(KeyError):
            registry.set_champion("v99")


class TestChampion:
    def test_set_and_get_champion_round_trip(self, registry):
        v1 = make_version("v1")
        registry.add_version(v1)
        registry.set_champion("v1")
        champion = registry.get_champion()
        assert champion is not None
        assert champion == v1

    def test_champion_can_be_reassigned(self, registry):
        registry.add_version(make_version("v1"))
        registry.add_version(make_version("v2"))
        registry.set_champion("v1")
        registry.set_champion("v2")
        assert registry.get_champion().version == "v2"
