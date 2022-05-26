import time
import warnings

import numpy as np
import os
import argparse
from collections import deque
import gym

import torch

from models import *
from utils import *

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="parameter setting for mujoco")
parser.add_argument('--env_name', type=str, default="Hopper-v2")
parser.add_argument('--method', choices=['ppo', 'sac'])
parser.add_argument('--n_episode', type=int, default=10)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def demo(n_episode=1):
    env = gym.make(args.env_name)
    policy = PPOActor(env.observation_space.shape[0],
                      env.action_space.shape[0], 256).to(device) if args.method == 'ppo' else \
             SACActor(env.observation_space.shape[0], env.action_space.shape[0], 256).to(device)
    policy_state_dict = torch.load(os.path.join('models', args.env_name, 'policy.pth'))
    policy.load_state_dict(policy_state_dict)
    grad_false(policy)

    def exploit(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = policy(state)
            if args.method == 'sac':
                mu = torch.tanh(mu)
        return mu.detach().cpu().numpy().reshape(-1)

    e_rewards = []
    for _ in range(n_episode):
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            if n_episode <= 1:
                env.render()
            action = exploit(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        e_rewards.append(episode_reward)
    print("Average reward of " + args.env_name + " is %.1f" % (np.mean(e_rewards)))
    print("Average std of " + args.env_name + " is %.1f" % (np.std(e_rewards)))

demo(args.n_episode)

