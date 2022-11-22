"""
# File       : RL_brain.py
# Time       ：2022/11/21 19:27
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import pandas as pd
import numpy as np


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decat=0.9, e_greed=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decat
        self.epsilon = e_greed

        self.q_table = pd.DataFrame(columns=self.actions)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass

class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decat=0.9, e_greed=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decat, e_greed)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decat=0.9, e_greed=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decat, e_greed)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_!='terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decat=0.9, e_greed=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decat, e_greed)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not  in self.q_table.index:
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state
            )

            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        self.eligibility_trace.loc[s, :] *=0
        self.eligibility_trace.loc[s, a] = 1

        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_

