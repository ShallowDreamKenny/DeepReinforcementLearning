"""
# File       : test_pfrl_2.py
# Time       ：2022/12/8 13:50
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import numpy
import pfrl
import logging
import sys
import gym
import torch
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

env = gym.make('CartPole-v1')
obs = env.reset()

# Set the discount factor that discounts future rewards.
gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(numpy.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = 0

class MLP(torch.nn.Module):
    def __init__(self, obs_size, n_act):
        super().__init__()
        self.mlp = self.__mlp(obs_size, n_act)

    def __mlp(self, obs_size, n_act):
        return torch.nn.Sequential(
            torch.nn.Linear(obs_size, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, n_act),
        )

    def forward(self, x):
        return pfrl.action_value.DiscreteActionValue(self.mlp(x))
obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_func = MLP(obs_size, n_actions)

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
    gpu=gpu,
)

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=2000,           # Train the agent for 2000 steps
    eval_n_steps=None,       # We evaluate for episodes, not time
    eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
    train_max_episode_len=200,  # Maximum length of each episode
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='result',      # Save everything to 'result' directory
)
