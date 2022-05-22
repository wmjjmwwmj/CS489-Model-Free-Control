from collections import deque
import numpy as np
import math
import torch

def to_batch(state, action, reward, next_state, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done

def update_params(optimizer, network, loss, grad_clip=None, retain_graph=False):
    optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optimizer.step()

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

def hard_update(target, source):
    target.load_state_dict(source.state_dict())

def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False

def make_mini_batch(*value):
    mini_batch_size = value[0]
    full_batch_size = len(value[1])
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    for i in range(full_batch_size // mini_batch_size):
        indices = full_indices[mini_batch_size * i: mini_batch_size * (i + 1)]
        yield [x[indices] for x in value[1:]]

class RunningMeanStats:
    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)