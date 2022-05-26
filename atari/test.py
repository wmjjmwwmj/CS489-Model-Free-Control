import time

import numpy as np
import os
import argparse
from collections import deque

import torch

from models import *
from atari_wrappers import wrap_deepmind, make_atari

parser = argparse.ArgumentParser(description="parameter setting for atari")
parser.add_argument('--env_name', type=str, default="BreakoutNoFrameskip-v4")
parser.add_argument('--is_dueling', action='store_true')
parser.add_argument('--is_double', action='store_true')
parser.add_argument('--n_episode', type=int, default=10)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def demo(n_episode=1):
    env_raw = make_atari(args.env_name)
    env = wrap_deepmind(env_raw)
    c, h, w = fp(env.reset()).shape
    action_dim = env.action_space.n

    if args.is_dueling:
        policy_net = DuelingQNet(h, w, action_dim, device).to(device)
    else:
        policy_net = QNet(h, w, action_dim, device).to(device)

    policy_state_dict = torch.load(os.path.join('models', args.env_name, 'policy.pth'))
    policy_net.load_state_dict(policy_state_dict)

    policy_net.eval()
    sa = ActionSelector(0., 0., 10000, policy_net, action_dim, device)
    frame_q = deque(maxlen=5)
    e_rewards = []
    for episode in range(n_episode):
        print(f"Demo episode {episode+1}/{n_episode}...")
        env.reset()
        e_reward = 0
        for _ in range(5): # noop
            n_frame, _, done, _ = env.step(0)
            n_frame = fp(n_frame)
            frame_q.append(n_frame)

        while not done:
            if n_episode <= 1:
                env.render()
                time.sleep(0.02)
            state = torch.cat(list(frame_q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state)
            n_frame, reward, done, _ = env.step(action)
            n_frame = fp(n_frame)
            frame_q.append(n_frame)
            e_reward += reward

        e_rewards.append(e_reward)
    avg_reward = float(sum(e_rewards))/float(n_episode)
    std = np.array(e_rewards).std()
    env.close()
    print(f"Average reward of {args.env_name} is {avg_reward:.1f}")
    print(f"Std of {args.env_name} is {std:.1f}")

demo(args.n_episode)




