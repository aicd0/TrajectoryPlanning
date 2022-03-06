import numpy as np
import torch

class DataType:
    Numpy = np.float32
    Torch = torch.float32

class Evaluator:
    MaxIterations = 10000
    OutputLocation = 'output/results'

class Model:
    ActorHidden1 = 600
    ActorHidden2 = 600
    CheckpointDir = 'output/checkpoint'
    CriticHidden1 = 600
    CriticHidden2 = 600
    InitialWeight = 0.03
    SaveStepInterval = 1000

class DDPG:
    BatchSize = 64
    Epsilon = 50000
    Gamma = 0.99
    LRActor = 0.0001
    LRCritic = 0.001
    MaxEpisodes = 200000
    MaxIterations = 10000
    NoiseEnabled = True
    ReplayBuffer = 50000
    Tau = 0.001
    Warmup = 1000

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

class Test:
    DetachAgent = False
    MaxEpisodes = 1
    NoiseEnabled = True
    
class Simulator:
    class Gym:
        # Supported environment:
        # CartPole-v0
        # CartPole-v1
        # FetchReach-v1
        # Pendulum-v1
        Environment = 'FetchReach-v1'
        MujocoLibPath = 'C:/Users/stdcn/.mujoco/mjpro150/bin'

    class MATLAB:
        SessionFile = '../../output/MatlabLauncher/session.txt'
        OutputLocation = 'output/matlab'