import numpy as np
import torch

# Valid targets:
# - train
# - test
# - debug
Target = 'train'

class DataType:
    Numpy = np.float32
    Torch = torch.float32

class Model:
    InitialWeight = 0.03

class Train:
    LoadFromPreviousSession = True

    class DDPG:
        BatchSize = 64
        Epsilon = 20000
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
            Sigma = 0.5
            Theta = 0.15

        class UniformNoise:
            Max = 1.0
            Min = -1.0

    class HER:
        Enabled = True
        K = 2

class Test:
    DetachAgent = False
    NoiseEnabled = False
    MaxEpoches = 100
    MaxIterations = 10000

class Evaluator:
    CheckpointLocation = 'output/checkpoint'
    OutputLocation = 'output/results'
    EpochWindowSize = 20
    MinSaveStepInterval = 1000

    class Figure:
        DPI = 300
        Height = 5
        Width = 9
        MaxSaveEpochInterval = 10

class Simulator:
    # Valid platforms:
    # - matlab
    # - gym
    # - ros
    Platform = 'ros'

    class Gym:
        # These environments have been tested and work fine:
        # - CartPole-v0
        # - CartPole-v1
        # - FetchReach-v1
        # - Pendulum-v1
        Environment = 'FetchReach-v1'

        # This option is only for Windows user.
        MujocoLibPath = 'C:/Users/stdcn/.mujoco/mjpro150/bin'

    class MATLAB:
        SessionFile = '../../output/MatlabLauncher/session.txt'
        OutputLocation = 'output/matlab'

    class ROS:
        DynamicEnabled = False
        ActionAmp = 0.1

        # Only make a difference when dynamic is enabled.
        StepIterations = 50