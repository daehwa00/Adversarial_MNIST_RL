import random
from typing import Tuple
from math import floor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
num_seeds = 10
action_size = 2
RL_epoch = 100


env_name = "MnistEnv-v1"
agent_name = "Reinforce"


class Reinforce(nn.Module):
    def __init__(self, opt):
        super(Reinforce, self).__init__()

        self.opt = opt

        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")

        self.fc1 = nn.Linear(3136, 1024)  # 7 * 7 * 64 = 3136
        self.fc2 = nn.Linear(1024, 1024)

        self.fc_mu = torch.nn.Linear(1024, action_size)
        self.fc_std = torch.nn.Linear(1024, action_size)

        self.x_range = torch.linspace(-1, 1, 28)
        self.y_range = torch.linspace(-1, 1, 28)
        self.x, self.y = torch.meshgrid(
            self.x_range, self.y_range, indexing="xy")
        self.x = self.x.expand([opt.batch_size, 1, -1, -1])
        self.y = self.y.expand([opt.batch_size, 1, -1, -1])
        self.normalized_coord = torch.cat([self.x, self.y], dim=1).to(
            opt.device)  # [batch_size,2,28,28]

    def forward(self, state):
        state = self.conv1(
            torch.cat([state, self.normalized_coord], dim=1))  # conv1
        state = F.relu(state)
        state = F.max_pool2d(state, 2)
        state = self.conv2(state)  # conv2
        state = F.relu(state)
        state = F.max_pool2d(state, 2)
        state = torch.flatten(state, 1)
        state = self.fc1(state)  # fc1
        state = F.relu(state)
        state = self.fc2(state)  # fc2
        state = F.relu(state)
        mu = self.fc_mu(state)
        std = self.fc_std(state)

        # sigmoid를 통해 0~1사이의 값으로 만들어주며, softplus를 통해 0~무한대의 값으로 만들어준다.
        return torch.sigmoid(mu), F.softplus(std)
