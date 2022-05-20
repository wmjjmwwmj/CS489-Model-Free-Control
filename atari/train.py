import numpy as np
import os
import argparse
from collections import deque

import torch.optim as optim

from models import *
from atari_wrappers import wrap_deepmind, make_atari

parser = argparse.ArgumentParser(description="parameter setting for atari")
parser.add_argument('--env_name', type=str, default="VideoPinball-ramNoFrameskip-v4")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--is_dueling', action='store_true')
parser.add_argument('--is_double', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model hyper-parameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
NUM_STEPS = 15000000
M_SIZE = 200000
POLICY_UPDATE = 4
EVALUATE_FREQ = 50000
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
memory = ReplayMemory(M_SIZE, [5,h,w], device)
sa = ActionSelector(EPS_START, EPS_END, EPS_DECAY, policy_net, action_dim, device)

best_reward = 0.

def optimize_model(train):
    if not train:
        return
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(BATCH_SIZE)
    q = policy_net(state_batch).gather(1, action_batch)
    if args.is_double:
        greedy_act = policy_net(n_state_batch).max(dim=1, keepdim=True)[1]
        nq = target_net(n_state_batch).gather(1, greedy_act).squeeze()
    else:
        nq = target_net(n_state_batch).max(dim=1)[0]

    # compute the expected Q values
    target_q_value = (nq * GAMMA)*(1. - done_batch[:,0]) + reward_batch[:,0]

    # Compute loss
    loss = F.smooth_l1_loss(q, target_q_value.unsqueeze(1))

    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def evaluate(step, policy_net, device, env, action_dim, n_episode=5):
    global best_reward
    os.makedirs("models", exist_ok=True)
    env = wrap_deepmind(env)
    sa = ActionSelector(0., 0., EPS_DECAY, policy_net, action_dim, device) # use greedy alg during testing
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
            action, eps = sa.select_action(state)
            n_frame, reward, done, _ = env.step(action)
            n_frame = fp(n_frame)
            frame_q.append(n_frame)

            e_reward += reward
        e_rewards.append(e_reward)

    f = open(args.env_name+".csv", 'a')
    avg_reward = float(sum(e_rewards))/float(n_episode)
    std = np.array(e_rewards).std()
    print(f"The average reward is {avg_reward:.5f}")
    if avg_reward > best_reward:
        print("New best reward, save model to disk!!!")
        checkpoint_name = f"DQN_{args.env_name}_best.pth"
        if args.is_dueling:
            checkpoint_name = "Dueling" + checkpoint_name
        if args.is_double:
            checkpoint_name = "Double" + checkpoint_name
        torch.save(policy_net.state_dict(), "models/"+checkpoint_name)
        best_reward = avg_reward
    f.write("%f, %f, %d, %d\n" % (avg_reward, std, step, n_episode))
    f.close()

frame_q = deque(maxlen=5)
reward_q = deque(maxlen=2)
done = True
eps = 0
episode_len = 0

for step in range(NUM_STEPS):
    if done:
        env.reset()
        sum_reward = 0
        episode_len = 0
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

    # perform one step of optimization
    if step % POLICY_UPDATE == 0:
        optimize_model(train)
    # update target network
    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    # evaluate current model performance
    if step % EVALUATE_FREQ == 0:
        evaluate(step, policy_net, device, env_raw, action_dim, n_episode=5)







