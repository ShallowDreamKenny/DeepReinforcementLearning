"""
# File       : train.py
# Time       ：2022/12/8 12:43
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""

import gym
import numpy
import pfrl.experiments
import torch
import logging

from module import MLP
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class TrainMenager():

    def __init__(self, env, epo=1000, lr=0.001, gamma=0.9, e_greed=0.1,
                 batch_size=32,  # 每一批次的数量,
                 num_steps=4,  #进行学习的频次
                 memory_size = 2000,
                 replay_start_size = 200,
                 update_target_steps = 200, ):
        self.env = env
        self.epo = epo

        self.OBS = env.observation_space.shape[0]
        self.n_act = env.action_space.n
        self.q_func = MLP(self.OBS, self.n_act)
        rb = pfrl.replay_buffers.ReplayBuffer(capacity=memory_size, num_steps=num_steps)
        explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=e_greed, random_action_func=env.action_space.sample)

        self.agent = pfrl.agents.DQN(
            q_function=self.q_func,
            optimizer=torch.optim.Adam(self.q_func.parameters(), lr=lr),
            replay_buffer=rb,
            explorer=explorer,
            minibatch_size=batch_size,
            replay_start_size=replay_start_size,
            target_update_interval=update_target_steps,
            gamma=gamma,
            phi=lambda x: x.astype(numpy.float32, copy=False),
            gpu=0
        )



    def train(self):
        pfrl.experiments.train_agent_with_evaluation(
            self.agent,
            self.env,
            steps=20000,

            # 测试的时候的设置
            # 二选一
            eval_n_steps=None,  # 多少步
            eval_n_episodes=10, # 多少轮游戏

            train_max_episode_len=200,
            eval_interval=1000, # 每1000次测试一次
            outdir='result_2'
        )

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    tm = TrainMenager(env)
    tm.train()