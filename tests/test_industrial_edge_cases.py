"""Industrial edge cases for the packing cell and warehouse routing layers:
crushed cargo, oversized items, saturated bins, LIFO unload-order burial,
pallet rollover and diversion, blocked aisles, dead sensors, NaN telemetry,
and drift-correction thrash."""

import numpy as np
import pytest

from marl_sim2real.agents.coordinator import MARLCoordinator
from marl_sim2real.agents.physics_agent import PhysicsAgent
from marl_sim2real.agents.proposer_agent import ProposerAgent
from marl_sim2real.config import DriftConfig, GNNConfig, PackingConfig
from marl_sim2real.envs import (
    CargoSpec,
    ConstrainedPackingEnv,
    DeliveryPackingEnv,
    Item,
    LoadTracker,
    MultiBinPackingEnv,
    Placement,
    check_unload_order,
)
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


# ------------------------------------------------- unload order (LIFO)
class TestUnloadOrder:
    def test_later_stop_burying_earlier_stop_is_vetoed(self):
        early = _placement((3, 3, 2), 0, 0, 0)
        late = _placement((3, 3, 2), 0, 0, 2)
        report = check_unload_order([early], [1], late, 2)
        assert not report.ok
        assert report.violations == [(0, 1, 2)]

    def test_earlier_stop_on_later_stop_is_fine(self):
        below = _placement((3, 3, 2), 0, 0, 0)
        above = _placement((3, 3, 2), 0, 0, 2)
        assert check_unload_order([below], [3], above, 1).ok

    def test_same_stop_stacking_is_fine(self):
        below = _placement((3, 3, 2), 0, 0, 0)
        above = _placement((3, 3, 2), 0, 0, 2)
        assert check_unload_order([below], [2], above, 2).ok

    def test_overhang_burial_without_direct_support_is_vetoed(self):
        # Short stop-1 carton at z<2, tall stop-3 tower beside it. A stop-2
        # bridge rests on the tower but overhangs the carton with an air gap:
        # still a burial -- the carton can't be lifted out vertically.
        short_early = _placement((2, 2, 2), 0, 0, 0)
        tall_late = _placement((2, 2, 4), 2, 0, 0)
        bridge = _placement((4, 2, 2), 0, 0, 4)
        report = check_unload_order([short_early, tall_late], [1, 3], bridge, 2)
        assert not report.ok
        assert report.violations == [(0, 1, 2)]  # only the carton is buried

    def test_floor_placement_never_violates(self):
        occupied = _placement((2, 2, 2), 0, 0, 0)
        beside = _placement((2, 2, 2), 2, 2, 0)
        assert check_unload_order([occupied], [1], beside, 3).ok

    def test_worst_violation_is_largest_stop_gap(self):
        a = _placement((2, 2, 1), 0, 0, 0)
        b = _placement((2, 2, 1), 2, 0, 0)
        wide = _placement((4, 2, 1), 0, 0, 1)
        report = check_unload_order([a, b], [1, 4], wide, 5)
        assert not report.ok and len(report.violations) == 2
        assert report.worst() == (0, 1, 5)  # gap 4 beats gap 1

    def test_delivery_env_appends_stop_to_observation(self):
        base = ConstrainedPackingEnv(PackingConfig(bin_size=(5, 5, 5), max_items=4), seed=0)
        env = DeliveryPackingEnv(PackingConfig(bin_size=(5, 5, 5), max_items=4), seed=0, num_stops=4)
        assert env.observe().shape[0] == base.observe().shape[0] + 1
        frac = env.observe()[-1]
        assert 0.0 < frac <= 1.0
        assert all(1 <= s <= 4 for s in env.item_stops)

    def test_delivery_env_vetoes_and_commit_records_stops(self):
        env = DeliveryPackingEnv(
            PackingConfig(bin_size=(6, 6, 8), max_items=2), seed=0, num_stops=3,
            fragile_fraction=0.0,
        )
        env.items = [Item(dims=(3, 3, 2)), Item(dims=(3, 3, 2))]
        env.item_stops = [1, 3]
        env.item_specs = [CargoSpec(), CargoSpec()]
        env.item_idx = 0
        first = env.try_place(0)
        env.commit(first)
        assert env.placement_stops == [1]
        stacked = env.try_place(0)  # same cell -> rests on the stop-1 box
        assert stacked.z == 2
        report = env.check_unload_order(stacked)
        assert not report.ok  # stop-3 over stop-1


# ------------------------------------------------- multi-bin rollover
class TestMultiBinRollover:
    def _cell(self, **kw):
        defaults = dict(
            config=PackingConfig(bin_size=(5, 5, 5), max_items=30, min_item_dim=2, max_item_dim=4),
            seed=0, max_bins=4, num_stops=1, fragile_fraction=0.0,
        )
        defaults.update(kw)
        return MultiBinPackingEnv(**defaults)

    def test_rollover_opens_new_bins_for_overflow(self):
        env = self._cell()
        while not env.done():
            mask = ProposerAgent.feasible_mask(env)
            if not mask.any():
                if not env.roll_bin_if_useful():
                    env.divert_item()
                continue
            env.commit(env.try_place(int(np.flatnonzero(mask)[0])))
        assert env.bins_used > 1
        assert env.bins_used <= env.max_bins
        assert all(0.0 < rec.density <= 1.0 for rec in env.bin_summary() if rec.items_placed)

    def test_rollover_refused_for_empty_bin(self):
        env = self._cell()
        assert not env.placements
        assert env.roll_bin_if_useful() is False  # fresh bin would be identical
        assert env.bins_used == 1

    def test_tracker_and_stop_state_reset_per_bin(self):
        env = self._cell(num_stops=3)
        mask = ProposerAgent.feasible_mask(env)
        env.commit(env.try_place(int(np.flatnonzero(mask)[0])))
        assert env.tracker.specs and env.placement_stops
        assert env.roll_bin_if_useful() is True
        assert not env.tracker.specs and not env.placement_stops
        assert (env.heightmap == 0).all()

    def test_overall_density_accounts_for_every_opened_bin(self):
        env = self._cell()
        volume = 0.0
        while not env.done():
            mask = ProposerAgent.feasible_mask(env)
            if not mask.any():
                if not env.roll_bin_if_useful():
                    env.divert_item()
                continue
            placement = env.try_place(int(np.flatnonzero(mask)[0]))
            env.commit(placement)
            volume += float(np.prod(placement.oriented_dims()))
        expected = volume / (env.W * env.D * env.H * env.bins_used)
        assert env.overall_density() == pytest.approx(expected)
        assert sum(r.items_placed for r in env.bin_summary()) + env.diverted == len(env.items)

    def test_coordinator_diverts_items_too_big_for_any_bin(self):
        env = MultiBinPackingEnv(
            PackingConfig(bin_size=(4, 4, 4), max_items=5, min_item_dim=6, max_item_dim=7),
            seed=0, max_bins=3,
        )
        proposer = ProposerAgent(env)
        physics = PhysicsAgent(use_pybullet=False)
        stats = MARLCoordinator(env, proposer, physics).run_episode(train=False)
        assert stats.diverted == 5 == env.diverted
        assert stats.placed == 0
        assert env.bins_used == 1  # rollover never helps an empty bin


# ------------------------------------- combined constraints, full episodes
class TestFullIndustrialEpisode:
    def test_episode_invariants_with_all_constraints_active(self):
        env = MultiBinPackingEnv(
            PackingConfig(bin_size=(5, 5, 5), max_items=20, min_item_dim=2, max_item_dim=4),
            seed=3, max_bins=3, num_stops=3, fragile_fraction=0.5, fragile_max_load=1.0,
        )
        proposer = ProposerAgent(env)
        physics = PhysicsAgent(use_pybullet=False)
        stats = MARLCoordinator(env, proposer, physics).run_episode(train=True)
        assert stats.placed + stats.rejected + stats.diverted == 20
        assert 0.0 <= stats.density <= 1.0
        assert (env.heightmap <= env.H).all()
        assert env.bins_used <= 3

    def test_committed_stack_satisfies_lifo_invariant(self):
        env = MultiBinPackingEnv(
            PackingConfig(bin_size=(6, 6, 6), max_items=25, min_item_dim=2, max_item_dim=3),
            seed=5, max_bins=2, num_stops=3, fragile_fraction=0.0,
        )
        proposer = ProposerAgent(env)
        physics = PhysicsAgent(use_pybullet=False)
        MARLCoordinator(env, proposer, physics).run_episode(train=False)
        # Every committed pair in the open bin must respect unload order.
        for j, upper in enumerate(env.placements):
            others = env.placements[:j]
            stops = env.placement_stops[:j]
            assert check_unload_order(others, stops, upper, env.placement_stops[j]).ok

    def test_committed_stack_satisfies_load_limits(self):
        env = MultiBinPackingEnv(
            PackingConfig(bin_size=(6, 6, 6), max_items=25, min_item_dim=2, max_item_dim=3),
            seed=7, max_bins=2, num_stops=1, fragile_fraction=0.6, fragile_max_load=1.5,
        )
        proposer = ProposerAgent(env)
        physics = PhysicsAgent(use_pybullet=False)
        MARLCoordinator(env, proposer, physics).run_episode(train=False)
        for load, spec in zip(env.tracker.load_on, env.tracker.specs):
            assert load <= spec.max_load + 1e-9

    def test_zero_item_episode_is_a_clean_noop(self):
        env = MultiBinPackingEnv(
            PackingConfig(bin_size=(5, 5, 5), max_items=0), seed=0, max_bins=2,
        )
        proposer = ProposerAgent(env)
        physics = PhysicsAgent(use_pybullet=False)
        stats = MARLCoordinator(env, proposer, physics).run_episode(train=False)
        assert stats.placed == stats.rejected == stats.diverted == 0
        assert env.done() and env.bins_used == 1
