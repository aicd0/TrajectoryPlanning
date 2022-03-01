import numpy as np
import torch

# MATLAB
MatlabSessionFile = '../../outputs/MatlabLauncher/session.txt'

# Trainning
#  Data type
NumpyDType = np.float32
TorchDType = torch.float32

#  Checkpoint directory
CheckpointDir = '../../outputs/Checkpoint'

#  Hyperparameters
class DDPG:
    ActionNoise = 0.1
    BatchSize = 32
    Gamma = 0.99
    Iterations = 256
    LRActor = 0.001
    LRCritic = 0.001
    MaxEpisode = 8192
    MaxStep = 2048
    ReplayBuffer = 4096
    Tau = 0.001

class HER:
    Enable = True
    K = 8

# Testing
class Test:
    MaxStep = 256
    