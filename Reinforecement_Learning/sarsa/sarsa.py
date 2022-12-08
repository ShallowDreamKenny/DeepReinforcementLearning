"""
# File       : sarsa.py
# Time       ：2022/12/8 9:46
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import time

import gym
import numpy as np

class SarsaAgent():
    def __init__(self, n_states, n_act, e_greed=0.1, lr=0.1, gamma=0.9):
        self.Q = np.zeros((n_states, n_act))
        self.e_greed = e_greed
        self.n_act = n_act
        self.lr = lr
        self.gamma = gamma

    def predict(self, state):
        Q_list = self.Q[state, :]
        action = np.random.choice(np.flatnonzero(Q_list==Q_list.max()))
        return action

    def act(self, state):
        if np.random.uniform(0,1) < self.e_greed:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(state)
        return action


    def learn(self, state, action, reward, next_state, next_action, done, truncated):
        cur_Q = self.Q[state, action]

        if done or truncated:
            self.Q[state, action] += self.lr * (reward - cur_Q)
        else:
            self.Q[state, action] += self.lr * (reward + self.gamma * (self.Q[next_state, next_action]) - cur_Q)

def train_eposode(env, agent, is_render):
    total_reward = 0
    state, info = env.reset()
    action = agent.act(state)

    while True:
        next_state, reward, done, truncated, _ = env.step(action)
        next_action = agent.act(next_state)

        agent.learn(state, action, reward, next_state, next_action, done, truncated)
        action = next_action
        state = next_state

        total_reward += reward

        if not is_render:
            env.render()
        if done or truncated:
            break
    return total_reward

def test_episode(agent, is_render):
    env = gym.make('CliffWalking-v0', render_mode="human")
    total_reward = 0
    state, info = env.reset()


    while True:
        action = agent.predict(state)
        next_state, reward, done, _, _ = env.step(action)


        state = next_state

        total_reward += reward
        if is_render:
            env.render()
        time.sleep(0.5)
        if done: break
    return total_reward

def train(env, epi=500, lr=0.1, gamma=0.9, e_greed=0.1):
    agent = SarsaAgent(
        n_states=env.observation_space.n,
        n_act=env.action_space.n,
        lr = lr,
        gamma = gamma,
        e_greed = e_greed
    )

    is_render = True
    for e in range(epi):
        ep_reward = train_eposode(env, agent, is_render)
        print('Epsode %s:reward = %.1f'%(e, ep_reward))

    reward = test_episode(agent, is_render)
    print('test reward = %.1f' % (reward))

if __name__ == '__main__':
    env = gym.make('CliffWalking-v0', render_mode="rgb_array")
    train(env)