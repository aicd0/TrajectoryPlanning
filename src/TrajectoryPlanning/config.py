import numpy as np
import torch

class DataType:
    Numpy = np.float32
    Torch = torch.float32

class Model:
    InitialWeight = 0.03

class Train:
    LoadFromPreviousSession = False

    class DDPG:
        BatchSize = 64
        Epsilon = 50000
        Gamma = 0.99
        LRActor = 0.0001
        LRCritic = 0.001
        MaxEpoches = 200000
        MaxIterations = 10000
        MinLogStepInterval = 500
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

    class HER:
        Enabled = False
        K = 2

class Test:
    DetachAgent = False
    NoiseEnabled = True
    MaxEpoches = 100
    MaxIterations = 10000

class Evaluator:
    CheckpointLocation = 'output/checkpoint'
    OutputLocation = 'output/results'
    EpochWindowSize = 20

    class Figure:
        DPI = 300
        Height = 5
        Width = 9
        MaxSaveEpochInterval = 10

class Simulator:
    class Gym:
        # These environment have been tested and work well:
        #  CartPole-v0
        #  CartPole-v1
        #  FetchReach-v1
        #  Pendulum-v1
        Environment = 'FetchReach-v1'

        # This option is only for Windows user.
        MujocoLibPath = 'C:/Users/stdcn/.mujoco/mjpro150/bin'

    class MATLAB:
        SessionFile = '../../output/MatlabLauncher/session.txt'
        OutputLocation = 'output/matlab'