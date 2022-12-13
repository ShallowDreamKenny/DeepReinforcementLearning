"""
# File       : atari_Tennis_dqn.py
# Time       ：2022/12/13 15:09
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from atari.atari_network import DQN
from atari.atari_wrapper import make_atari_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import  IntrinsicCuriosityModule

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='TennisNoFrameskip-v4')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-obs", type=int, default=0) # 是否标准化输入数据


    # 探索噪声 类似于greedy的概率
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument("--eps-train-final", type=float, default=0.05)

    # ReplayBuffer的大小
    parser.add_argument("--buffer-size", type=int, default=100000)

    # 训练参数
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)  # 向前看几步（estimation_step）
    parser.add_argument("--target-update-freq", type=int, default=500) # 每进行多少部更新target网络

    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step-per-epoch", type=int, default=100000)

    #  the number of times the policy network would be updated per transition after (step_per_collect)
    #  transitions are collected,
    #  e.g., if update_per_step set to 0.3, and step_per_collect is 256 ,
    #  policy will be updated round(256 * 0.3 = 76.8) = 77 times
    #           after 256 transitions are collected by the collector.
    parser.add_argument("--step-per-collect", type=int, default=10) # 收集多少步之后开始训练
    parser.add_argument("--update-per-step", type=float, default=0.1)    #  Default to 1.

    parser.add_argument("--batch-size", type=int, default=32) # 每次更新拿多少步的数据

    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--frames-stack", type=int, default=8) # 堆栈，即每次返回返回atari图像中最近的4帧的图像

    # 导入原始模型继续训练
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)

    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)

    return parser.parse_args()


def test_Tennis(args=get_args()):
    env, train_envs, test_envs = make_atari_env(
        args.env,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #  定义模型
    net = DQN(*args.state_shape, args.action_shape, args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack
    )

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy,test_envs,exploration_noise=True)

    # logger
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "dqn"
    log_name = os.path.join(args.env, args.algo_name, str(args.seed), now)
    log_path = os.path.join("../../tf-logs", log_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif "Pong" in args.env:
            return mean_rewards >= 20
        else:
            return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        return ckpt_path

    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    pprint.pprint(result)

if __name__ == '__main__':
    test_Tennis()