import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, states, actions):
        x = torch.cat((states, actions), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(dim_state, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, dim_action)

    def forward(self, states):
        x = self.fc1(states)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x