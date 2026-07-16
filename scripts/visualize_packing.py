#!/usr/bin/env python
"""Runs one episode of PackingEnv and prints a human-readable text trace to stdout,
followed by an ASCII heightmap of the resulting packing.

Headless-friendly: no GUI, no video, console output only.

Usage:
    python scripts/visualize_packing.py --seed 1
    python scripts/visualize_packing.py --version v3 --registry-dir models/registry
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from marl_packing.envs.packing_env import PackingEnv
from marl_packing.mlops.registry import ModelRegistry
from marl_packing.utils.config import load_config

# Shading buckets from empty (low height) to full bin height.
_SHADES = " .:-=+*#%@"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PackingEnv episode and print a text trace + ASCII heightmap.")
    parser.add_argument("--env-config", default="configs/env.yaml", help="Path to env YAML config.")
    parser.add_argument("--registry-dir", default="models/registry", help="ModelRegistry directory.")
    parser.add_argument("--version", default=None, help='Model version id to load (e.g. "v3"). If omitted or not found, falls back to random actions.')
    parser.add_argument("--seed", type=int, default=0, help="Episode seed.")
    return parser.parse_args()


def try_load_models(registry_dir: str, version_id: str | None):
    """Attempts to load a (packer, physics) SB3 PPO model pair for the given version.

    Returns (packer_model, physics_model) or (None, None) if unavailable, in which case
    the caller should fall back to random actions.
    """
    if version_id is None:
        return None, None

    try:
        registry = ModelRegistry(registry_dir)
        version = registry.get_version(version_id)
    except (KeyError, OSError) as exc:
        print(f"[note] Could not load version {version_id!r} from registry {registry_dir!r} ({exc}); falling back to random actions.")
        return None, None

    try:
        from stable_baselines3 import PPO

        packer_model = PPO.load(version.packer_path)
        physics_model = PPO.load(version.physics_path)
    except Exception as exc:  # noqa: BLE001 - any load failure should fall back gracefully
        print(f"[note] Failed to load PPO checkpoints for version {version_id!r} ({exc}); falling back to random actions.")
        return None, None

    return packer_model, physics_model


def format_dims(dims: np.ndarray) -> str:
    return f"({dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f})"


def render_heightmap(heightmap: np.ndarray, bin_height: float) -> str:
    """Renders the heightmap as rows of shaded ASCII characters, normalized against the
    bin's max height into len(_SHADES) buckets."""
    if bin_height <= 0:
        bin_height = float(heightmap.max()) or 1.0

    lines = []
    # heightmap indexed [x, y]; print rows as y (top to bottom) x columns as x, so the
    # printed grid reads left-to-right / top-to-bottom like a top-down view of the bin.
    grid = heightmap.T
    for row in grid[::-1]:
        chars = []
        for h in row:
            frac = np.clip(h / bin_height, 0.0, 1.0)
            idx = int(round(frac * (len(_SHADES) - 1)))
            chars.append(_SHADES[idx])
        lines.append("".join(chars))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    env_config = load_config(args.env_config)
    packer_model, physics_model = try_load_models(args.registry_dir, args.version)
    using_random = packer_model is None or physics_model is None
    if args.version is not None and not using_random:
        print(f"Loaded trained model version {args.version!r} from {args.registry_dir!r}.")
    elif args.version is None:
        print("[note] No --version given; using random actions for both agents.")

    env = PackingEnv(env_config, randomization_mode="train")
    try:
        env.reset(seed=args.seed)
        print(f"Episode seed={args.seed}, bin_dims={list(env.bin_dims)}, items_total={env.num_items_per_episode}")
        print("-" * 78)

        episode_summary = None
        item_idx = 0
        pending_dims = None
        pending_pos = None

        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue

            obs = env.observe(agent)

            if agent == "packer":
                item_idx = env._item_idx
                if using_random:
                    action = env.action_space("packer").sample()
                else:
                    action, _ = packer_model.predict(obs, deterministic=True)
                env.step(np.asarray(action, dtype=np.float32))

                pp = env._pending_placement
                pending_dims = pp.dims
                pending_pos = pp.center
            else:  # physics
                if using_random:
                    action = env.action_space("physics").sample()
                else:
                    action, _ = physics_model.predict(obs, deterministic=True)
                accept = int(action)
                env.step(accept)

                outcome = env.infos["physics"].get("outcome", "?")
                utilization_pct = 100.0 * env._utilized_volume / env.bin_volume
                decision = "ACCEPT" if accept else "REJECT"
                print(
                    f"item {item_idx:2d}: dims={format_dims(pending_dims)} pos={format_dims(pending_pos)} "
                    f"physics={decision:6s} outcome={outcome:28s} running_utilization={utilization_pct:5.1f}%"
                )

                if env.terminations["physics"]:
                    episode_summary = env.infos["packer"].get("episode_summary") or env.infos["physics"].get(
                        "episode_summary"
                    )

        print("-" * 78)
        if episode_summary is None:
            episode_summary = {
                "utilization_pct": 100.0 * env._utilized_volume / env.bin_volume,
                "items_placed": env._items_placed,
                "items_total": env.num_items_per_episode,
            }

        print("Episode summary:")
        print(f"  utilization_pct: {episode_summary['utilization_pct']:.2f}%")
        print(f"  items_placed:    {episode_summary['items_placed']} / {episode_summary['items_total']}")
        print()
        print("Final heightmap (top-down view, shaded low->high height):")
        print(f"  shading: {_SHADES!r} (space=empty floor, '@'=at bin height {env.bin_dims[2]:.2f}m)")
        print(render_heightmap(env._heightmap, float(env.bin_dims[2])))
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
