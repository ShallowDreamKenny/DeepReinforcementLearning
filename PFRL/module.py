"""
# File       : module.py
# Time       ：2022/12/8 12:39
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import torch
import pfrl
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