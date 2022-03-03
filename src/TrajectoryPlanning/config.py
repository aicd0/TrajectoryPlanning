import numpy as np
import torch

# MATLAB
MatlabSessionFile = '../../outputs/MatlabLauncher/session.txt'

# Trainning
class DataType:
    Numpy = np.float32
    Torch = torch.float32

#  Evaluator
class Evaluator:
    MaxIterations = 10000

# Model
class Model:
    ActorHidden1 = 400
    ActorHidden2 = 300
    CheckpointDir = '../../outputs/Checkpoint'
    CriticHidden1 = 400
    CriticHidden2 = 300
    InitialWeight = 0.003
    SaveStepInterval = 1000

#  Hyper-parameters
class DDPG:
    BatchSize = 64
    Epsilon = 50000
    Gamma = 0.99
    LRActor = 0.0001
    LRCritic = 0.001
    MaxEpisodes = 200000
    MaxIterations = 10000
    NoiseEnabled = True
    ReplayBuffer = 8192
    Tau = 0.001
    Warmup = 100

    class OUNoise:
        Mu = 0.0
        Sigma = 0.2
        Theta = 0.15

    class UniformNoise:
        Max = 1.0
        Min = -1.0

    class Evaluation:
        MinStepInterval = 2000
        MaxEpisodes = 20

class HER:
    Enabled = False
    K = 8

# Testing
class Test:
    DetachAgent = False
    NoiseEnabled = False
    