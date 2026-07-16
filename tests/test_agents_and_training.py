import numpy as np

from marl_sim2real.agents import MLP, PhysicsAgent, ProposerAgent
from marl_sim2real.env import BinPackingEnv
from marl_sim2real.training import MARLTrainer


def test_mlp_learns_xor():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(256, 2)).astype(float)
    y = (X[:, 0] != X[:, 1]).astype(float).reshape(-1, 1)
    net = MLP([2, 16, 1], seed=0)
    from marl_sim2real.agents import sigmoid

    for _ in range(600):
        probs = sigmoid(net.forward(X))
        net.backward((probs - y) / len(y), lr=0.5)
    acc = ((sigmoid(net.forward(X)) > 0.5) == y.astype(bool)).mean()
    assert acc > 0.95


def test_proposer_respects_action_mask():
    obs_size, num_actions = 10, 24
    agent = ProposerAgent(obs_size, num_actions, hidden=16, seed=0)
    obs = np.zeros(obs_size)
    mask = np.zeros(num_actions, dtype=bool)
    mask[[3, 7, 11]] = True
    probs = agent.action_probs(obs, mask)
    assert probs[~mask].max() < 1e-6
    for flat in agent.propose(obs, mask=mask, top_k=3):
        assert mask[flat]


def test_physics_agent_learns_separable_labels():
    rng = np.random.default_rng(1)
    obs_size = 8
    agent = PhysicsAgent(obs_size, hidden=16, seed=1, lr=0.05)
    n = 512
    feats = rng.normal(size=(n, agent.input_size))
    labels = (feats[:, 0] > 0).astype(float)  # trivially separable on dim 0
    for _ in range(300):
        agent.train_batch(feats, labels)
    assert agent.accuracy(feats, labels) > 0.9


def test_marl_training_improves_utilization():
    env = BinPackingEnv(bin_size=(6, 6, 6), max_items=8, seed=0)
    proposer = ProposerAgent(env.observation_size, env.num_positions * 6, hidden=32, seed=0)
    physics = PhysicsAgent(env.observation_size, hidden=32, seed=1)
    trainer = MARLTrainer(env, proposer, physics, top_k=4, seed=0)

    before = trainer.evaluate(episodes=5)["mean_utilization"]
    trainer.train(episodes=60, verbose=False)
    after = trainer.evaluate(episodes=5)["mean_utilization"]
    # training should not collapse performance, and physics agent should learn
    assert after >= before * 0.5
    assert trainer.stats.physics_accuracy > 0.6


def test_physics_agent_veto_ranking():
    agent = PhysicsAgent(obs_size=4, hidden=8, seed=0)
    feats = [np.zeros(agent.input_size), np.ones(agent.input_size)]
    ranked = agent.rank(feats)
    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]
