#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Actor (Policy) Model
    """

    def __init__(self, state_size, action_size, seed, fc1_units=256):
        """
        Initialize parameters and build model.

        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(9216, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """
        Build a network that maps state to action values.
        """
        x = F.relu(self.conv1(state), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x), inplace=True)
        return self.fc2(x)
