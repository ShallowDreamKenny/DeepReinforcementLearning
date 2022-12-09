import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle

from dqn import Agent

env = gym.make('LunarLander-v2')
env.seed(0)
print('状态空间的形状: ', env.observation_space.shape)
print('动作空间中的动作数量: ', env.action_space.n)

agent = Agent(state_size=8, action_size=4, seed=0, buffer_size=int(20e5), batch_size=256)


def train_agent(n_episodes=5000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """实现智能体与环境交互并学习这一过程

    Params
    ======
        n_episodes (int): 任务执行次数
        max_t (int): 每次任务的最大执行步数
        eps_start (float): 起始探索率
        eps_end (float): 最小的探索率
        eps_decay (float): 探索率的下降因子
    """
    scores = []  # 存储每次任务的回报(Return)
    scores_window = deque(maxlen=100)  # 最新100次任务的回报
    eps = eps_start  # 初始化探索率
    for episode_i in range(1, n_episodes + 1):
        traj = []  # 一条轨迹
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            if episode_i % 100 == 0:
                env.render()

            transition = agent.buffer.transition(state, action, reward, next_state, done)
            traj.append(transition)

            state = next_state  # 这一步很关键，别忘了更新
            score += reward
            if done:
                break

        # 将轨迹存入经验池
        agent.add_traj(traj)

        # 学习
        for _ in range(int(len(traj) / 4)):
            agent.learn()

        # 保存得分
        scores_window.append(score)
        scores.append(score)

        eps = max(eps_end, eps_decay * eps)  # 探索率衰减

        print('\rEpisode {}\t平均得分: {:.2f}'.format(episode_i, np.mean(scores_window)), end="")
        if episode_i % 100 == 0:
            print('\rEpisode {}\t平均得分: {:.2f}'.format(episode_i, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')  # 保存模型
        if np.mean(scores_window) >= 200.0 and len(scores_window) >= 100:
            print('\n任务已经成功完成！总共经过 {:d} 次任务的训练。!\t最近100次任务的平均得分为: {:.2f}'.format(episode_i - 100,
                                                                                 np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint1.pth')  # 保存模型
            break
    return scores


scores = train_agent()  # 保存得分变化情况，方便后续画图

with open('scores.pkl', 'wb') as file:
    pickle.dump(scores, file)

with open('scores.pkl', 'rb') as file:
    scores = pickle.load(file)

# 画图
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# 从保存的文件加载网络模型参数
agent.qnetwork_local.load_state_dict(torch.load('checkpoint1.pth'))

for i in range(300):
    state = env.reset()
    score = 0
    for j in range(1000):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print(j, ':', score)
            break

env.close()
