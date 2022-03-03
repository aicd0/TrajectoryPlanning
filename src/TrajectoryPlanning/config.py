import numpy as np
import torch

# MATLAB
MatlabSessionFile = '../../outputs/MatlabLauncher/session.txt'

# Trainning
class DataType:
    Numpy = np.float32
    Torch = torch.float32

#  Checkpoint directory
CheckpointDir = '../../outputs/Checkpoint'

#  Hyperparameters
class DDPG:
    BatchSize = 32
    Gamma = 0.99
    LRActor = 0.001
    LRCritic = 0.0001
    MaxEpisode = 8192
    MaxIterations = 2048
    MinSteps = 512
    NoiseAmount = 1.0
    NoiseEnabled = True
    ReplayBuffer = 8192
    Tau = 0.001

class HER:
    Enable = False
    K = 4

# Testing
class Test:
    DetachAgent = False
    MaxStep = 512
    NoiseEnabled = False
    