"""SB3 training callback that mirrors rollout metrics into the active MLflow run, so every
training call (smoke or full-scale) shows up in `mlflow ui` without extra plumbing."""

from __future__ import annotations

import mlflow
from stable_baselines3.common.callbacks import BaseCallback

_TRACKED_KEYS = (
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss",
    "train/entropy_loss",
    "train/value_loss",
    "train/approx_kl",
)


class MlflowLoggingCallback(BaseCallback):
    """Logs SB3's internal `self.logger` scalars to MLflow, namespaced by `prefix` (e.g.
    "packer" or "physics") so both agents' curves coexist in one MLflow run."""

    def __init__(self, prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.prefix = prefix

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        for key in _TRACKED_KEYS:
            value = self.logger.name_to_value.get(key)
            if value is not None:
                metric_name = f"{self.prefix}/{key.split('/')[-1]}"
                mlflow.log_metric(metric_name, float(value), step=self.num_timesteps)
