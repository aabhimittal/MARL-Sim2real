"""Alternating independent-learner MARL training.

Trains the packer agent for one round with the physics agent's policy frozen, then swaps
and trains physics with the packer frozen, repeating for `schedule.n_rounds` rounds (see
configs/train_packer.yaml). This is a simple, standard MARL scheme (iterative best-response
/ alternating self-play) that lets each agent be trained with an off-the-shelf single-agent
algorithm (SB3 PPO) via the wrappers in agents/single_agent_wrappers.py, rather than
requiring a native multi-agent trainer.

Produces one challenger model version (packer.zip + physics.zip), registered in the local
model registry -- see mlops/registry.py and docs/MLOPS.md for what happens to it next
(evaluation against the current champion, then possible promotion).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from marl_packing.agents.single_agent_wrappers import (
    PackerSingleAgentEnv,
    PhysicsSingleAgentEnv,
    RandomOpponent,
)
from marl_packing.envs.packing_env import PackingEnv
from marl_packing.mlops.registry import ModelRegistry, ModelVersion
from marl_packing.training.callbacks import MlflowLoggingCallback
from marl_packing.utils.config import load_config
from marl_packing.utils.logging import get_logger

logger = get_logger(__name__)


def _config_hash(*configs: dict) -> str:
    blob = json.dumps(configs, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def train(
    env_config: dict,
    packer_config: dict,
    physics_config: dict,
    registry_dir: str,
    experiment_name: str,
    full: bool,
    mlflow_tracking_uri: str = "./mlruns",
) -> str:
    schedule = packer_config["schedule"]
    timesteps_per_round = schedule["timesteps_per_round"] if full else schedule["smoke_timesteps_per_round"]
    n_rounds = schedule["n_rounds"]

    base_env = PackingEnv(
        env_config, randomization_mode="train", reward_shaping=physics_config.get("reward_shaping")
    )

    packer_model: PPO | None = None
    physics_model: PPO | None = None

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "full_scale": full,
                "n_rounds": n_rounds,
                "timesteps_per_round": timesteps_per_round,
                **{f"packer_ppo_{k}": v for k, v in packer_config["ppo"].items()},
                **{f"physics_ppo_{k}": v for k, v in physics_config["ppo"].items()},
            }
        )

        for round_idx in range(n_rounds):
            train_packer_this_round = round_idx % 2 == 0
            if train_packer_this_round:
                logger.info("Round %d/%d: training packer (physics frozen)", round_idx + 1, n_rounds)
                physics_opponent = physics_model or RandomOpponent(base_env.action_space("physics"))
                wrapped = DummyVecEnv(
                    [lambda: PackerSingleAgentEnv(base_env, physics_policy=physics_opponent, seed=env_config.get("seed"))]
                )
                if packer_model is None:
                    packer_model = PPO(env=wrapped, **packer_config["ppo"])
                else:
                    packer_model.set_env(wrapped)
                packer_model.learn(
                    total_timesteps=timesteps_per_round,
                    callback=MlflowLoggingCallback("packer"),
                    reset_num_timesteps=False,
                )
            else:
                logger.info("Round %d/%d: training physics (packer frozen)", round_idx + 1, n_rounds)
                packer_opponent = packer_model or RandomOpponent(base_env.action_space("packer"))
                wrapped = DummyVecEnv(
                    [lambda: PhysicsSingleAgentEnv(base_env, packer_policy=packer_opponent, seed=env_config.get("seed"))]
                )
                if physics_model is None:
                    physics_model = PPO(env=wrapped, **physics_config["ppo"])
                else:
                    physics_model.set_env(wrapped)
                physics_model.learn(
                    total_timesteps=timesteps_per_round,
                    callback=MlflowLoggingCallback("physics"),
                    reset_num_timesteps=False,
                )

        base_env.close()

        registry = ModelRegistry(registry_dir)
        version_id = registry.next_version_id()
        version_dir = registry.version_dir(version_id)
        packer_path = str(version_dir / "packer.zip")
        physics_path = str(version_dir / "physics.zip")
        packer_model.save(packer_path)
        physics_model.save(physics_path)

        version = ModelVersion(
            version=version_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            packer_path=packer_path,
            physics_path=physics_path,
            train_config_hash=_config_hash(env_config, packer_config, physics_config),
        )
        registry.add_version(version)
        mlflow.log_param("registered_version", version_id)
        mlflow.set_tag("mlflow_run_id_for_version", run.info.run_id)
        logger.info(
            "Registered challenger version %s (packer=%s, physics=%s)", version_id, packer_path, physics_path
        )

    return version_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--packer-config", default="configs/train_packer.yaml")
    parser.add_argument("--physics-config", default="configs/train_physics.yaml")
    parser.add_argument("--registry-dir", default="models/registry")
    parser.add_argument("--experiment-name", default="marl-packing")
    parser.add_argument("--mlflow-tracking-uri", default="./mlruns")
    parser.add_argument("--full", action="store_true", help="Run full-scale timesteps instead of a smoke run.")
    args = parser.parse_args()

    version_id = train(
        env_config=load_config(args.env_config),
        packer_config=load_config(args.packer_config),
        physics_config=load_config(args.physics_config),
        registry_dir=args.registry_dir,
        experiment_name=args.experiment_name,
        full=args.full,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    print(version_id)


if __name__ == "__main__":
    main()
