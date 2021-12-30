import layer
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, num_input):
        super(Critic, self).__init__()
        self.fc = layer.FullConnect(num_input, [50, 50, 1])

    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Actor(nn.Module):
    def __init__(self, num_input, num_output):
        super(Actor, self).__init__()
        self.fc = layer.FullConnect(num_input, [50, 50, num_output])

    def forward(self, x):
        x = self.fc.forward(x)
        return x