import config
import torch
import torch.nn.functional as F
from framework.configuration import global_configs as configs
from models.utils import fanin_init
from torch import nn

class Actor (nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        h1 = 512
        h2 = 512
        h3 = 512
        h4 = 512
        w = configs.get(config.Model.InitialWeight_)

        self.fc1 = nn.Linear(dim_state, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, h3)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(h3, h4)
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

        self.fc5 = nn.Linear(h4, dim_action)
        self.fc5.weight.data.uniform_(-w, w)

        self.drop_out = nn.Dropout(0.2)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

class Critic (nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        h1 = 512
        h2 = 512
        h3 = 512
        h4 = 512
        w = configs.get(config.Model.InitialWeight_)

        self.fc1 = nn.Linear(dim_state + dim_action, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, h3)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(h3, h4)
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

        self.fc5 = nn.Linear(h4, 1)
        self.fc5.weight.data.uniform_(-w, w)

        self.drop_out = nn.Dropout(0.2)

    def forward(self, states, actions):
        x = torch.cat((states, actions), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x