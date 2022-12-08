from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set train mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))   # to tensor
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam) # 创建概率分布
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()

def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # 更新参数theta
    return loss

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)  # 随机梯度下降

    for epi in range(300):
        state, info = env.reset()
        for t in range(200):
            # 最大时间步为200
            action = pi.act(state)
            state, reward, done, finish, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done or finish:
                break

        loss = train(pi, optimizer) # 每一幕结束后进行参数更新
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()
        print(f"Episode{epi}, loss:{loss}, "
              f"total_reward:{total_reward}, solved: {solved}")