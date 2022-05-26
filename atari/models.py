import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, h, w, action_dim, device):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, action_dim)
        self.device = device

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

class DuelingQNet(nn.Module):
    def __init__(self, h, w, action_dim, device):
        super(DuelingQNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.adv = nn.Linear(512, action_dim)
        self.val = nn.Linear(512, 1)
        self.device = device

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        adv = self.adv(x)
        val = self.val(x).expand(adv.size())
        out = val + adv - adv.mean().expand(adv.size())
        return out

class ActionSelector:
    def __init__(self, initial_eps, final_eps, eps_decay, policy_net, action_dim, device):
        self.eps = initial_eps
        self.final_eps = final_eps
        self.initial_eps = initial_eps
        self.policy_net = policy_net
        self.eps_decay = eps_decay
        self.action_dim = action_dim
        self.device = device

    def select_action(self, state, training=False):
        sample = random.random()
        if training:
            self.eps -= (self.initial_eps - self.final_eps) / self.eps_decay
            self.eps = max(self.eps, self.final_eps)
        if sample > self.eps:
            with torch.no_grad():
                a = self.policy_net(state).max(1)[1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self.action_dim)]], device='cpu', dtype=torch.long)

        return a.numpy()[0,0].item(), self.eps

class ReplayMemory:
    def __init__(self, capacity, state_shape, device):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        self.m_states[self.position] = state
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, batch_size):
        indices = torch.randint(0, high=self.size, size=(batch_size,))
        bs = self.m_states[indices, :4] # state
        bns = self.m_states[indices, 1:] # next state
        ba = self.m_actions[indices].to(self.device)
        br = self.m_rewards[indices].to(self.device).float()
        bd = self.m_dones[indices].to(self.device).float()
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size

class PrioritizedReplay:
    def __init__(self, capacity, state_shape, device, alpha=0.6, beta_start=0.4, beta_annealing=0.001, epsilon=1e-4):
        c, h, w = state_shape
        self.alpha = alpha
        self.capacity = capacity
        self.device = device
        self.beta = beta_start
        self.epsilon = epsilon
        self.beta_annealing = beta_annealing
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.m_priority = torch.zeros((capacity,), dtype=torch.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        self.m_states[self.position] = state
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done

        max_prio = self.m_priority.max() if self.size else 1.0
        self.m_priority[self.position] = max_prio

        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, batch_size):
        prios = self.m_priority.squeeze().numpy()[:self.size]
        probs = prios ** self.alpha
        P = probs/probs.sum()

        indices = np.random.choice(self.size, batch_size, p=P)
        bs = self.m_states[indices, :4]  # state
        bns = self.m_states[indices, 1:]  # next state
        ba = self.m_actions[indices].to(self.device)
        br = self.m_rewards[indices].to(self.device).float()
        bd = self.m_dones[indices].to(self.device).float()
        self.beta = min(1.-self.epsilon, self.beta+self.beta_annealing)

        # compute importance sampling weight
        weights = (self.size * P[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(np.array(weights, dtype=np.float32),dtype=torch.float32).to(self.device)

        return bs, ba, br, bns, bd, indices, weights

    def update_priority(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.m_priority[idx] = torch.abs(prio)+self.epsilon

    def __len__(self):
        return self.size


def fp(n_frame):
    n_frame = torch.from_numpy(n_frame)
    h = n_frame.shape[-2]
    return n_frame.view(1, h, h)

class FrameProcessor:
    def __init__(self, im_size=84):
        self.im_size = im_size

    def process(self, frame):
        im_size = self.im_size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[46:160+46, :]

        frame = cv2.resize(frame, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
        frame = frame.reshape((1, im_size, im_size))

        x = torch.from_numpy(frame)
        return x