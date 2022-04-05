import numpy as np
import torch

# Valid targets:
# - train
# - test
# - debug
Target = 'debug'

ConfigDir = 'output/configs'

class DataType:
    Numpy = np.float32
    Torch = torch.float32

class Model:
    InitialWeight = 0.03

class Train:
    LoadFromPreviousSession = False

    class DDPG:
        MinLogStepInterval = 500

        DefaultBatchSize = 64
        DefaultEpsilon = 40000
        DefaultGamma = 0.99
        DefaultLRActor = 0.0001
        DefaultLRCritic = 0.001
        DefaultMaxEpoches = 200000
        DefaultMaxIterations = 10000
        DefaultNoiseEnabled = True
        DefaultReplayBuffer = 50000
        DefaultTau = 0.001
        DefaultWarmup = 2000

        class PER:
            DefaultEnabled = True
            DefaultAlpha = 0.5
            DefaultK = 0.01

        class OUNoise:
            DefaultMu = 0.0
            DefaultSigma = 1.0
            DefaultTheta = 0.15

        class UniformNoise:
            DefaultMax = 1.0
            DefaultMin = -1.0

    class HER:
        DefaultEnabled = True
        DefaultK = 4

class Test:
    DetachAgent = False
    NoiseEnabled = False
    MaxEpoches = 100
    MaxIterations = 10000

class Evaluator:
    SaveDir = 'output/checkpoint'
    StatisticDir = 'output/statistics'

    DefaultEpochWindowSize = 20
    DefaultMinSaveStepInterval = 1000

    class Figure:
        DefaultDPI = 300
        DefaultHeight = 5
        DefaultWidth = 9
        DefaultMaxSaveEpochInterval = 10

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
        DynamicEnabled = False # has to be False
        
        DefaultActionAmp = 0.1
        DefaultStepIterations = 50 # has to be consistent to sensor frequencies