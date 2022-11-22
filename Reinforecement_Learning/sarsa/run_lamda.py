"""
# File       : run.py
# Time       ：2022/11/21 19:27
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
from RL_brain import *
from maze_env import *


def update():
    for episode in range(100):
        observation = env.reset()
        # print(observation)
        action = RL.choose_action(str(observation))

        while True:
            env.render()
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_
            action = action_

            if done:
                break

    print('game over!')

if __name__ == '__main__':
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()