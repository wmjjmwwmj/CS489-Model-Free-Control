import numpy as np
import os
import argparse
from collections import deque
from datetime import datetime

import torch.optim as optim

from models import *
from atari_wrappers import wrap_deepmind, make_atari

parser = argparse.ArgumentParser(description="parameter setting for atari")
parser.add_argument('--env_name', type=str, default="BreakoutNoFrameskip-v4")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--is_dueling', action='store_true')
parser.add_argument('--is_double', action='store_true')
parser.add_argument('--is_per', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_dir = os.path.join('local', args.env_name,
   f'seed{args.seed}-duel{args.is_dueling}-double{args.is_double}-per{args.is_per}-{datetime.now().strftime("%Y%m%d-%H%M")}')
model_dir = os.path.join(local_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# model hyper-parameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
NUM_STEPS = 10000000
M_SIZE = 200000
POLICY_UPDATE = 4
EVALUATE_FREQ = 25000
LR = 0.0000625

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if args.seed:
    fix_seed(args.seed)

# reset environment
env_raw = make_atari(args.env_name)
env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)
c, h, w = fp(env.reset()).shape
action_dim = env.action_space.n

if args.is_dueling:
    policy_net = DuelingQNet(h, w, action_dim, device).to(device)
    target_net = DuelingQNet(h, w, action_dim, device).to(device)
else:
    policy_net = QNet(h, w, action_dim, device).to(device)
    target_net = QNet(h, w, action_dim, device).to(device)
policy_net.apply(policy_net.init_weights)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR, eps=1.5e-4)
memory = PrioritizedReplay(M_SIZE, [5,h,w], device) if args.is_per else ReplayMemory(M_SIZE, [5,h,w], device)
sa = ActionSelector(EPS_START, EPS_END, EPS_DECAY, policy_net, action_dim, device)

best_reward = 0.

def optimize_model(train):
    if not train:
        return None, None, None
    if args.is_per:
        state_batch, action_batch, reward_batch, n_state_batch, done_batch, indices, weights = memory.sample(BATCH_SIZE)
    else:
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(BATCH_SIZE)
        weights = 1.

    q = policy_net(state_batch).gather(1, action_batch)
    if args.is_double:
        greedy_act = policy_net(n_state_batch).max(dim=1)[1].unsqueeze(1)
        nq = target_net(n_state_batch).gather(1, greedy_act).squeeze().detach()
    else:
        nq = target_net(n_state_batch).max(dim=1)[0].detach()

    # compute the expected Q values
    target_q_value = (nq * GAMMA * (1. - done_batch[:,0]) + reward_batch[:,0]).unsqueeze(1)

    # Compute loss
    td_error = target_q_value - q
    loss = torch.mean(F.smooth_l1_loss(q, target_q_value, reduction='none') * weights)

    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    if args.is_per:
        memory.update_priority(indices, td_error.detach().cpu())

    return loss, torch.mean(q), torch.mean(target_q_value)

def evaluate(step, policy_net, device, env, action_dim, loss, q, target_q, eps=0.01, n_episode=5):
    global best_reward
    env = wrap_deepmind(env)
    sa = ActionSelector(eps, eps, EPS_DECAY, policy_net, action_dim, device)
    e_rewards = []
    frame_q = deque(maxlen=5)
    for _ in range(n_episode):
        env.reset()
        e_reward = 0
        for _ in range(5): # noop
            n_frame, _, done, _ = env.step(0)
            n_frame = fp(n_frame)
            frame_q.append(n_frame)

        while not done:
            state = torch.cat(list(frame_q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, train)
            n_frame, reward, done, _ = env.step(action)
            n_frame = fp(n_frame)
            frame_q.append(n_frame)

            e_reward += reward
        e_rewards.append(e_reward)

    f = open(os.path.join(local_dir, "testing_rewards.csv"), 'a')
    avg_reward = float(sum(e_rewards))/float(n_episode)
    min_reward = min(e_rewards)
    max_reward = max(e_rewards)
    print(f"The average reward is {avg_reward:.5f}")
    if avg_reward > best_reward:
        print("New best reward, save model to disk.")
        torch.save(policy_net.state_dict(), os.path.join(model_dir, "policy.pth"))
        best_reward = avg_reward
    f.write("%f, %f, %f, %f, %f, %f, %d, %d\n" % (avg_reward, min_reward, max_reward, loss, q, target_q, step, n_episode))
    f.close()

frame_q = deque(maxlen=5)
reward_q = deque(maxlen=2)
done = True
eps = 0
episodes = 0
episode_len = 0
episode_reward = 0
training_rewards = deque(maxlen=10)

for step in range(NUM_STEPS):
    if done:
        episodes += 1
        training_rewards.append(episode_reward)
        # print(f'episode: {episodes:<4}  '
        #       f'episode steps: {episode_len:<4}  '
        #       f'reward: {np.mean(training_rewards):<5.1f}')

        env.reset()
        episode_len = 0
        episode_reward = 0
        img, _, _, _ = env.step(1) # for BREAKOUT
        for i in range(5): # noop
            n_frame, reward, _, _ = env.step(0)
            n_frame = fp(n_frame)
            frame_q.append(n_frame)
            reward_q.append(reward)

    train = len(memory) > 500
    # select and act
    state = torch.cat(list(frame_q))[1:].unsqueeze(0)
    action, eps = sa.select_action(state, train)
    n_frame, reward, done, info = env.step(action)
    n_frame = fp(n_frame)

    # update memory
    frame_q.append(n_frame)
    reward_q.append(reward)
    memory.push(torch.cat(list(frame_q)).unsqueeze(0), action, reward_q[0], done)
    episode_len += 1
    episode_reward += reward

    # perform one step of optimization
    if (step+1) % POLICY_UPDATE == 0:
        loss, q, target_q = optimize_model(train)
    # update target network
    if (step+1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    # evaluate current model performance
    if (step+1) % EVALUATE_FREQ == 0:
        evaluate(step, policy_net, device, env_raw, action_dim, loss, q, target_q, eps=0.05, n_episode=10)







