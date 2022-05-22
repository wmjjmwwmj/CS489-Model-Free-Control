import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.memory = deque(maxlen=capacity)
        self.device = device

    def append(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        action = np.expand_dims(action, 0)

        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = random.sample(self.memory, k=batch_size)
        samples = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.tensor(np.concatenate(states)).to(self.device)
        next_states = torch.tensor(np.concatenate(next_states)).to(self.device)
        actions = torch.tensor(np.concatenate(actions)).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class PrioritizedReplay:
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_annealing=0.0001, epsilon=1e-4):
        self.alpha = alpha
        self.device = device
        self.beta = beta_start
        self.epsilon = epsilon
        self.beta_annealing = beta_annealing
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done, error):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        action = np.expand_dims(action, 0)

        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(abs(error)+self.epsilon)

    def sample(self, batch_size):
        N = len(self.memory)

        prios = np.array(list(self.priorities)[:N])
        probs = prios ** self.alpha
        P = probs/probs.sum()

        indices = np.random.choice(N, batch_size, p=P)
        samples = [self.memory[idx] for idx in indices]

        self.beta = min(1.-self.epsilon, self.beta+self.beta_annealing)

        # compute importance sampling weight
        weights = (N * P[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.tensor(np.concatenate(states)).to(self.device)
        next_states = torch.tensor(np.concatenate(next_states)).to(self.device)
        actions = torch.tensor(np.concatenate(actions)).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        weights = torch.tensor(weights).to(self.device)
        batch = states, actions, rewards, next_states, dones

        return batch, indices, weights

    def update_priority(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)+self.epsilon

    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2
        self.eps = 1e-6
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.log_std.weight)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, states):
        mus, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(mus, stds)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        log_probs = normals.log_prob(xs) - torch.log(1-actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(mus)

class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(QCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.q.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

class TwinnedQCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(TwinnedQCritic, self).__init__()
        self.Q1 = QCritic(state_dim, action_dim, hidden_size)
        self.Q2 = QCritic(state_dim, action_dim, hidden_size)

    def reset_parameters(self):
        self.Q1.reset_parameters()
        self.Q2.reset_parameters()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2

class VCritic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(VCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.v.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v
