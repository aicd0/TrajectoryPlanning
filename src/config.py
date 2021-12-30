import numpy as np
import torch

# Data type
NumpyDType = np.float32
TorchDType = torch.float32

# Simulator
SessionName = 'MATLAB_13448'

# Trainning
class DDPG:
    ActionNoise = 0.03
    BatchSize = 64
    Gamma = 0.99
    Iterations = 128
    LRActor = 0.005
    LRCritic = 0.005
    MaxEpisode = 8192
    MaxStep = 512
    ReplayBuffer = 2048
    Tau = 0.005

class HER:
    Enable = False
    K = 8