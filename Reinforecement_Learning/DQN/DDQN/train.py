"""
# File       : train.py
# Time       ：2022/12/8 12:43
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""

import gym
import torch

from DDQN import DDQNAgent
import time
from module import MLP


class TrainMenager():

    def __init__(self, env, epo=1000, lr=0.001, gamma=0.9, e_greed=0.1):
        self.env = env
        self.epo = epo

        self.OBS = env.observation_space.shape[0]
        self.n_act = env.action_space.n
        self.q_func = MLP(self.OBS, self.n_act)
        self.agent = DDQNAgent(
            q_func=self.q_func,
            n_act=self.n_act,
            lr=lr
        )

    def train_eposode(self, is_render):
        total_reward = 0
        obs, info = self.env.reset()
        obs = torch.FloatTensor(obs)


        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, truncated, _ = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs)

            self.agent.learn(obs, action, reward, next_obs, done, truncated)
            obs = next_obs

            total_reward += reward

            if not is_render:
                env.render()
            if done or truncated:
                break
        return total_reward

    def test_episode(self, is_render):
        env = gym.make('CartPole-v1', render_mode="human")

        total_reward = 0
        obs, info = env.reset()
        obs = torch.FloatTensor(obs)

        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, _, _ = env.step(action)
            next_obs = torch.FloatTensor(next_obs)

            obs = next_obs

            total_reward += reward
            if is_render:
                env.render()

            if done: break
        return total_reward

    def train(self):


        is_render = True
        for e in range(self.epo):
            ep_reward = self.train_eposode(is_render)
            print('Epsode %s:reward = %.1f'%(e, ep_reward))

        reward = self.test_episode( is_render)
        print('test reward = %.1f' % (reward))

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    tm = TrainMenager(env)
    tm.train()