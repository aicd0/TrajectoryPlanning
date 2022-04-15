from cv2 import mean
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.configuration import global_configs as configs
from models.utils import fanin_init

class Actor (nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        h1 = 512
        h2 = 512
        w = configs.get(config.Model.InitialWeight_)

        self.fc1 = nn.Linear(dim_state, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.mean = nn.Linear(h2, dim_action)
        self.mean.weight.data.uniform_(-w, w)
        
        self.log_std = nn.Linear(h2, dim_action)
        self.log_std.weight.data.uniform_(-w, w)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std

class Critic (nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        h1 = 512
        h2 = 512
        w = configs.get(config.Model.InitialWeight_)

        self.fc1 = nn.Linear(dim_state + dim_action, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-w, w)

    def forward(self, states, actions):
        x = torch.cat((states, actions), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x