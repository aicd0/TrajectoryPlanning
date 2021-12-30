import torch.nn as nn
import torch.nn.functional as F

class FullConnect():
    def __init__(self, num_input, num_outputs, activation=F.relu):
        self.fc = []
        for num_output in num_outputs:
            fc = nn.Linear(num_input, num_output)
            self.fc.append(fc)
            num_input = num_output
        
        self.activation = activation

    def forward(self, x):
        for fc in self.fc:
            x = fc(x)
            x = self.activation(x)
        return x