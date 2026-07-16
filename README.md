# MARL-Sim2real

Multi-agent reinforcement learning for 3D bin packing: a **packer** agent proposes box positions and orientations, and a **physics** agent accepts or rejects each proposal, with PyBullet drop simulation as the ground truth for stack stability. The two agents are trained with alternating independent-learner PPO (Stable-Baselines3) on a PettingZoo `AECEnv`. Sim2Real transfer is measured against a domain-randomized "real-world proxy" benchmark, and an MLOps loop (MLflow tracking + a local JSON model registry) gates deployment through a Champion/Challenger promotion gate.

## Quickstart

Requires Python 3.10+.

```bash
# Set up a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .   # or: make setup

# Run the test suite
make test

# Fast CPU smoke run of the full training pipeline
# (env -> agents -> PPO -> MLflow -> model registry); prints the new version id
make smoke-train

# Run a challenger through the Champion/Challenger promotion gate
# (with no prior champion, the first challenger is auto-promoted)
make challenger-eval

# Browse MLflow runs
make mlflow-ui
```

Other targets: `make lint` (ruff), `make train` (full-scale timesteps instead of smoke), `make promote CHALLENGER=<version_id>`, `make clean`.

## Project layout

```
configs/                  YAML configs: env + domain randomization, PPO/schedule for both agents, promotion gate
scripts/                  smoke-train and challenger-eval shell entrypoints, packing visualizer
src/marl_packing/
  envs/                   PettingZoo AECEnv packing environment, PyBullet drop simulator, domain randomization profiles
  agents/                 single-agent Gymnasium wrappers so each agent trains with off-the-shelf SB3 PPO
  training/               alternating-round MARL training loop (train.py) + MLflow logging callbacks
  evaluation/             paired-seed benchmark, metrics, and the real-world reference CSV
  mlops/                  local JSON model registry, challenger evaluation, promotion/deploy CLI
  utils/                  config loading, logging, seeding
tests/                    pytest suite (env, metrics, registry, challenger gate logic)
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system overview and how the pieces fit together
- [docs/MARL_DESIGN.md](docs/MARL_DESIGN.md) — the two-agent design and alternating training scheme
- [docs/SIM2REAL.md](docs/SIM2REAL.md) — domain randomization and the real-world proxy benchmark
- [docs/MLOPS.md](docs/MLOPS.md) — MLflow tracking, model registry, and the Champion/Challenger gate

## Scope and limitations

- Training runs here are deliberately short CPU smoke runs (`smoke_timesteps_per_round: 256`) that verify the pipeline end-to-end; this project was built in a dev environment without a GPU or budget for long training, so no model quality claims are made. `make train` runs the full-scale schedule if you have the compute.
- There is no physical robot or camera rig. The "real-world" benchmark is a domain-randomization proxy — the same simulator with wider friction/mass/restitution ranges and pose noise (`real_proxy` in `configs/env.yaml`) — and `src/marl_packing/evaluation/real_world_reference.csv` is an illustrative placeholder, not genuine field data. See [docs/SIM2REAL.md](docs/SIM2REAL.md) for details.
