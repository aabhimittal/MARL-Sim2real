#!/usr/bin/env bash
# Fast CPU smoke run of the full alternating-training loop. Used by CI to prove the
# training pipeline (env -> agents -> PPO -> MLflow -> model registry) is wired correctly
# end-to-end, without paying for full-scale training. Prints the registered challenger
# version id as the last line of stdout.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m marl_packing.training.train \
    --env-config configs/env.yaml \
    --packer-config configs/train_packer.yaml \
    --physics-config configs/train_physics.yaml \
    --registry-dir models/registry \
    --experiment-name ci-smoke
