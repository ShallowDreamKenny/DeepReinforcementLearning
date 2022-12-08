"""
# File       : DDQN.py
# Time       ：2022/12/8 12:35
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""

import time

import gym
import numpy as np
import torch

from module import MLP

class DDQNAgent():
    def __init__(self, q_func, n_act, e_greed=0.1, lr=0.1, gamma=0.9):
        self.q_func = q_func

        self.e_greed = e_greed
        self.n_act = n_act
        self.lr = lr
        self.gamma = gamma

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), self.lr)

    def predict(self, obs):
        Q_list = self.q_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    def act(self, obs):
        if np.random.uniform(0,1) < self.e_greed:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(obs)
        return action


    def learn(self, obs, action, reward, next_obs, done, truncated):
        cur_Q = self.q_func(obs)[action]

        next_pred_Vs = self.q_func(next_obs)
        best_V = next_pred_Vs.max()
        target = reward + (1 - (float(done) or float(truncated))) * self.gamma * best_V

        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(cur_Q, target)
        loss.backward()
        self.optimizer.step()