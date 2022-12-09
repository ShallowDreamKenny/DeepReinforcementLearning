import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """DQN智能体，实现能够与环境交互并学习."""

    def __init__(self, state_size, action_size, seed, buffer_size=int(1e5),
                 batch_size=64, gamma=0.99, tau=1e-3, lr=1e-3):
        """初始化智能体.

        参数
        ======
            state_size (int): （观测）状态维度大小
            action_size (int): 动作数量
            seed (int): 随机种子
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.BUFFER_SIZE = buffer_size  # 经验池大小
        self.BATCH_SIZE = batch_size  # 训练用的批大小
        self.GAMMA = gamma  # 折扣因子
        self.TAU = tau  # 目标网络更新参数
        self.LR = lr  # 优化器学习参数

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)  # 用于和环境交互的神经网络
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)  # 目标神经网络
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # 经验池
        self.buffer = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        # 初始化交互次数计数器
        self.t_step = 0

    def add_traj(self, traj):
        """将一条轨迹存入经验池中
        """
        for transition in traj:
            self.buffer.memory.append(transition)

    def act(self, state, eps=0.):
        """完成从状态到动作的映射.
        Params
        ======
            state (array_like): （观测）状态
            eps (float): 即epsilon, 贪婪策略的探索率
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # 按照贪婪策略选择动作
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """使用经验数据更新两个价值网络的参数。
           这是DQN的核心部分。

        Params
        ======
            experiences (Tuple[torch.Tensor]): 按照(s, a, r, s', done) 这样的格式组织的经验数据
            gamma (float): 折扣因子
        """
        if len(self.buffer) < 10 * self.BATCH_SIZE:
            return 0
        experiences = self.buffer.sample()
        states, actions, rewards, next_states, dones = experiences

        # 使用目标网络估计下个状态的价值
        Q_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        y = rewards + (self.GAMMA * Q_next * (1 - dones))  # 使用单步TD估计当前状态的价值

        Q = self.qnetwork_local(states).gather(1, actions)  # 使用局部网络直接估计当前状态的价值
        loss = F.mse_loss(Q, y)  # 计算TD误差，作为损失函数

        # 用优化器优化神经网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- 更新目标网络参数 ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """两个神经网络参数之间的软更新:
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): 用于和环境交互的网络模型
            target_model (PyTorch model): 目标网络模型
            tau (float): 更新步长
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """用于存储智能体与环境交互的经验"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): 动作数量
            buffer_size (int): 经验池大小
            batch_size (int): 训练
            seed (int): 随机种子
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def sample(self):
        """从经验池中随机采样一定数量的样本."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):  # 方便别的地方使用len()获取经验池汇中transition的数量
        return len(self.memory)


class QNetwork(nn.Module):
    """价值估计网络."""

    def __init__(self, state_size, action_size, seed, hidden_size_1=256, hidden_size_2=128):
        """初始化价值网络.
        Params
        ======
            state_size (int): 状态的维度大小
            action_size (int): 动作空维度数量
            seed (int): 随机种子
            hidden_size_1 (int): 第隐藏曾1的神经元数量
            hidden_size_2 (int): 第隐藏曾2的神经元数量
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(state_size, hidden_size_1)

        self.hidden_layer = nn.Linear(hidden_size_1, hidden_size_2)
        self.output_layer = nn.Linear(hidden_size_2, action_size)

    def forward(self, state):
        """数据的前向传播"""
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)
