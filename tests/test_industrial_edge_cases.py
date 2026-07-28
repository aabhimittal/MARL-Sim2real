"""Industrial edge cases for the packing cell and warehouse routing layers:
crushed cargo, oversized items, saturated bins, blocked aisles, dead sensors,
NaN telemetry, and drift-correction thrash."""

import numpy as np
import pytest

from marl_sim2real.agents.coordinator import MARLCoordinator
from marl_sim2real.agents.physics_agent import PhysicsAgent
from marl_sim2real.agents.proposer_agent import ProposerAgent
from marl_sim2real.config import DriftConfig, GNNConfig, PackingConfig
from marl_sim2real.envs import CargoSpec, ConstrainedPackingEnv, Item, LoadTracker, Placement
from marl_sim2real.gnn import DynamicGNN, NoRouteError, PathOptimizer, WarehouseGraph
from marl_sim2real.sim2real import DriftDetector, DriftGovernor
from marl_sim2real.sim2real.ideal_data_generator import SimBaseline


# ----------------------------------------------------------- load / crush
def _placement(dims, x, y, z, orientation=0):
    return Placement(item=Item(dims=dims), orientation=orientation, x=x, y=y, z=z)


class TestLoadBearing:
    def test_heavy_item_crushes_fragile_supporter(self):
        tracker = LoadTracker()
        base = _placement((4, 4, 2), 0, 0, 0)
        tracker.commit([], base, CargoSpec(mass=1.0, max_load=2.0))  # fragile carton
        heavy = _placement((4, 4, 2), 0, 0, 2)
        report = tracker.check([base], heavy, CargoSpec(mass=5.0))
        assert not report.ok
        idx, current, added, limit = report.worst()
        assert idx == 0 and added == pytest.approx(5.0) and limit == 2.0

    def test_light_item_on_fragile_supporter_is_fine(self):
        tracker = LoadTracker()
        base = _placement((4, 4, 2), 0, 0, 0)
        tracker.commit([], base, CargoSpec(mass=1.0, max_load=2.0))
        light = _placement((4, 4, 2), 0, 0, 2)
        assert tracker.check([base], light, CargoSpec(mass=1.5)).ok

    def test_load_distributes_across_two_supporters(self):
        tracker = LoadTracker()
        left = _placement((2, 4, 2), 0, 0, 0)
        right = _placement((2, 4, 2), 2, 0, 0)
        tracker.commit([], left, CargoSpec(mass=1.0, max_load=3.0))
        tracker.commit([left], right, CargoSpec(mass=1.0, max_load=3.0))
        # 4-wide bridge rests half on each: 2.5 kg per supporter -> both hold
        bridge = _placement((4, 4, 2), 0, 0, 2)
        assert tracker.check([left, right], bridge, CargoSpec(mass=5.0)).ok
        # 8 kg bridge -> 4 kg per supporter -> both crush
        report = tracker.check([left, right], bridge, CargoSpec(mass=8.0))
        assert not report.ok and len(report.violations) == 2

    def test_cumulative_load_accumulates_across_commits(self):
        tracker = LoadTracker()
        base = _placement((4, 4, 2), 0, 0, 0)
        tracker.commit([], base, CargoSpec(mass=1.0, max_load=3.0))
        mid = _placement((4, 4, 2), 0, 0, 2)
        tracker.commit([base], mid, CargoSpec(mass=2.0))
        # base already carries 2.0; another 2.0 on top of mid does NOT load base
        # directly (direct-support model) but a second 2.0 directly on base would:
        side = _placement((4, 4, 2), 0, 0, 2)
        report = tracker.check([base, mid], side, CargoSpec(mass=2.0))
        assert not report.ok  # 2.0 existing + 2.0 new > 3.0 limit

    def test_floor_placements_never_crush(self):
        tracker = LoadTracker()
        floor_item = _placement((3, 3, 3), 0, 0, 0)
        assert tracker.check([], floor_item, CargoSpec(mass=100.0)).ok


class TestConstrainedEnvEpisode:
    def test_constrained_episode_runs_and_respects_limits(self):
        env = ConstrainedPackingEnv(
            PackingConfig(bin_size=(8, 8, 8), max_items=10),
            seed=3, fragile_fraction=0.6, fragile_max_load=1.0,
        )
        proposer = ProposerAgent(env)
        physics = PhysicsAgent(use_pybullet=False)
        coord = MARLCoordinator(env, proposer, physics)
        stats = coord.run_episode(train=True)
        assert stats.placed + stats.rejected <= 10
        # invariant: no committed supporter may exceed its rating
        for load, spec in zip(env.tracker.load_on, env.tracker.specs):
            assert load <= spec.max_load + 1e-9

    def test_fragile_sampling_fraction(self):
        env = ConstrainedPackingEnv(PackingConfig(max_items=200), seed=0,
                                    fragile_fraction=1.0)
        assert all(s.fragile for s in env.item_specs)


# --------------------------------------------------------- geometry limits
class TestGeometryEdgeCases:
    def test_oversized_item_yields_empty_mask_and_episode_completes(self):
        env = ConstrainedPackingEnv(
            PackingConfig(bin_size=(4, 4, 4), max_items=5, min_item_dim=6, max_item_dim=7),
            seed=0,
        )
        assert not ProposerAgent.feasible_mask(env).any()
        proposer = ProposerAgent(env)
        physics = PhysicsAgent(use_pybullet=False)
        stats = MARLCoordinator(env, proposer, physics).run_episode(train=True)
        assert stats.placed == 0
        assert env.done()

    def test_saturated_bin_stops_accepting_and_density_bounded(self):
        env = ConstrainedPackingEnv(
            PackingConfig(bin_size=(4, 4, 4), max_items=40, min_item_dim=2, max_item_dim=4),
            seed=1, fragile_fraction=0.0,
        )
        proposer = ProposerAgent(env)
        physics = PhysicsAgent(use_pybullet=False)
        stats = MARLCoordinator(env, proposer, physics).run_episode(train=False)
        assert 0.0 <= stats.density <= 1.0
        assert (env.heightmap <= env.H).all()  # never packed through the ceiling

    def test_exact_fit_item_is_placeable(self):
        env = ConstrainedPackingEnv(PackingConfig(bin_size=(4, 4, 4), max_items=1), seed=0)
        env.items = [Item(dims=(4, 4, 4))]
        env.item_specs = [CargoSpec()]
        env.item_idx = 0
        placement = env.try_place(0)  # identity orientation at (0, 0)
        assert placement is not None and placement.z == 0
        env.commit(placement)
        assert env.packing_density() == pytest.approx(1.0)


# ---------------------------------------------------------- blocked aisles
def _routing_setup():
    graph = WarehouseGraph(num_nodes=8, seed=0)
    gnn = DynamicGNN(graph, GNNConfig(hidden_dim=32, num_layers=2))
    return graph, gnn, PathOptimizer(graph, gnn)


class TestBlockedAisles:
    def test_blocking_an_edge_reroutes(self):
        graph, gnn, opt = _routing_setup()
        nf, ef = graph.sample_features()
        path, cost = opt.shortest_path(0, graph.num_nodes - 1, nf, ef)
        # close the first aisle on the chosen route (both directions)
        blocked = opt.block_between(path[0], path[1])
        assert blocked
        path2, cost2 = opt.shortest_path(0, graph.num_nodes - 1, nf, ef)
        assert (path2[0], path2[1]) != (path[0], path[1])
        assert cost2 >= cost - 1e-9  # detour can't be cheaper than the optimum

    def test_unblocking_restores_the_short_route(self):
        graph, gnn, opt = _routing_setup()
        nf, ef = graph.sample_features()
        path, cost = opt.shortest_path(0, graph.num_nodes - 1, nf, ef)
        for e in opt.block_between(path[0], path[1]):
            opt.unblock_edge(e)
        path3, cost3 = opt.shortest_path(0, graph.num_nodes - 1, nf, ef)
        assert path3 == path and cost3 == pytest.approx(cost)

    def test_fully_blocked_goal_raises_no_route(self):
        graph, gnn, opt = _routing_setup()
        nf, ef = graph.sample_features()
        goal = graph.num_nodes - 1
        src, dst = graph.edge_index
        for e in range(graph.num_edges):  # sever every edge touching the goal
            if goal in (int(src[e]), int(dst[e])):
                opt.block_edge(e)
        with pytest.raises(NoRouteError, match="no route"):
            opt.shortest_path(0, goal, nf, ef)

    def test_block_edge_validates_id(self):
        _, _, opt = _routing_setup()
        with pytest.raises(ValueError):
            opt.block_edge(10_000)


# ------------------------------------------------------- telemetry / drift
def _baseline(num_edges=6, mean=2.0, std=0.2):
    return SimBaseline(mean=np.full(num_edges, mean), std=np.full(num_edges, std), samples=100)


def _governor(confirm=2, cooldown=10, **cfg):
    detector = DriftDetector(_baseline(), DriftConfig(window_size=8, min_samples=4, **cfg))
    return DriftGovernor(detector, confirm_ticks=confirm, cooldown_ticks=cooldown)


class TestDriftUnderRealTelemetry:
    def test_nan_telemetry_does_not_poison_detection(self):
        gov = _governor()
        rng = np.random.default_rng(0)
        event_seen = False
        for t in range(30):
            reading = rng.normal(2.0, 0.05, 6)
            reading[t % 6] = np.nan             # rotating sensor dropout
            if t >= 10:
                reading[2] = 6.0                # genuine drift on edge 2
            report = gov.update(reading)
            assert np.isfinite(report.sensor_dropout_rate)
            if report.event is not None:
                event_seen = True
                assert 2 in report.event.edge_ids
        assert event_seen  # NaNs never masked the real drift

    def test_dead_sensor_does_not_disable_fleetwide_detection(self):
        gov = _governor()
        rng = np.random.default_rng(1)
        events = []
        for t in range(30):
            reading = rng.normal(2.0, 0.05, 6)
            reading[5] = np.nan                 # edge 5's sensor is dead all shift
            if t >= 8:
                reading[0] = 8.0
            r = gov.update(reading)
            if r.event is not None:
                events.append(r.event)
        assert events and all(0 in e.edge_ids for e in events)

    def test_debounce_suppresses_single_tick_spike(self):
        gov = _governor(confirm=3)
        rng = np.random.default_rng(2)
        for _ in range(10):
            gov.update(rng.normal(2.0, 0.05, 6))
        spike = rng.normal(2.0, 0.05, 6)
        spike[1] = 50.0                          # one-off forklift stall
        r = gov.update(spike)
        assert r.event is None                   # held back by debounce
        for _ in range(6):                       # readings return to normal
            r = gov.update(rng.normal(2.0, 0.05, 6))
        assert not gov.confirmed_events          # spike washed out of the window

    def test_cooldown_prevents_correction_thrash(self):
        gov = _governor(confirm=2, cooldown=15)
        rng = np.random.default_rng(3)
        confirmed_ticks = []
        for t in range(40):
            reading = rng.normal(2.0, 0.05, 6)
            reading[3] = 7.0                     # persistently drifted edge
            r = gov.update(reading)
            if r.event is not None:
                confirmed_ticks.append(r.tick)
        assert confirmed_ticks                   # drift IS reported...
        gaps = np.diff(confirmed_ticks)
        assert (gaps >= 15).all()                # ...but never more often than cooldown

    def test_all_sensors_down_is_a_quiet_tick(self):
        gov = _governor()
        report = gov.update(np.full(6, np.nan))
        assert report.event is None
        assert report.sensor_dropout_rate == 1.0
