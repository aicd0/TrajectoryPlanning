import numpy as np
import torch
import utils.string_utils

OutputDir = 'output/gym'
ConfigDir = 'configs'

# Valid targets:
# - train
# - test
# - debug
Target = 'train'

class DataType:
    Numpy = np.float32
    Torch = torch.float32

class Model:
    FieldInitialWeight = ('Model/InitialWeight', 0.03)

class Train:
    LoadFromPreviousSession = False

    class DDPG:
        MinLogStepInterval = 500
        FieldBatchSize = ('Train/DDPG/BatchSize', 64)
        FieldEpsilon = ('Train/DDPG/Epsilon', 50000)
        FieldGamma = ('Train/DDPG/Gamma', 0.99)
        FieldLRActor = ('Train/DDPG/LRActor', 0.0001)
        FieldLRCritic = ('Train/DDPG/LRCritic', 0.001)
        FieldMaxEpoches = ('Train/DDPG/MaxEpoches', 200000)
        FieldMaxIterations = ('Train/DDPG/MaxIterations', 10000)
        FieldNoiseEnabled = ('Train/DDPG/NoiseEnabled', True)
        FieldReplayBuffer = ('Train/DDPG/ReplayBuffer', 50000)
        FieldTau = ('Train/DDPG/Tau', 0.001)
        FieldWarmup = ('Train/DDPG/Warmup', 1000)

        class OUNoise:
            FieldMu = ('Train/DDPG/OUNoise/Mu', 0.0)
            FieldSigma = ('Train/DDPG/OUNoise/Sigma', 0.2)
            FieldTheta = ('Train/DDPG/OUNoise/Theta', 0.15)

        class UniformNoise:
            FieldMax = ('Train/DDPG/UniformNoise/Max', 1.0)
            FieldMin = ('Train/DDPG/UniformNoise/Min', -1.0)

    class HER:
        FieldEnabled = ('Train/HER/Enabled', True)
        FieldK = ('Train/HER/K', 2)

    class PER:
        FieldEnabled = ('Train/PER/Enabled', False)
        FieldAlpha = ('Train/PER/Alpha', 0.8)
        FieldK = ('Train/PER/K', 0.01)

class Test:
    DetachAgent = False
    NoiseEnabled = False
    MaxEpoches = 100
    MaxIterations = 10000

class Evaluator:
    SaveDir = 'checkpoint'
    StatisticDir = 'statistics'
    FieldEpochWindowSize = ('Evaluator/EpochWindowSize', 20)
    FieldMinSaveStepInterval = ('Evaluator/MinSaveStepInterval', 1000)

    class Figure:
        FieldDPI = ('Evaluator/Figure/DPI', 300)
        FieldHeight = ('Evaluator/Figure/Height', 5)
        FieldWidth = ('Evaluator/Figure/Width', 9)
        FieldMaxSaveEpochInterval = ('Evaluator/Figure/MaxSaveEpochInterval', 10)

class Simulator:
    # Valid platforms:
    # - gym
    # - matlab
    # - ros
    FieldPlatform = ('Simulator/Platform', 'gym')

    class Gym:
        MujocoLibPath = 'C:/Users/stdcn/.mujoco/mjpro150/bin' # only for Windows users

        # These environments have been tested and work fine:
        # - CartPole-v0
        # - CartPole-v1
        # - FetchReach-v1
        # - Pendulum-v1
        FieldEnvironment = ('Simulator/Gym/Environment', 'Pendulum-v1')

    class MATLAB:
        SessionFile = '../../output/MatlabLauncher/session.txt'
        OutputDir = 'MATLAB'
        FieldActionAmp = ('Simulator/MATLAB/ActionAmp', 0.1)

    class ROS:
        ROSLibPath = 'C:/opt/ros/noetic/x64/Lib/site-packages' # only for Windows users
        FieldActionAmp = ('Simulator/ROS/ActionAmp', 0.1)
        FieldStepIterations = ('Simulator/ROS/StepIterations', 50) # has to be adapted to sensor frequencies

# Post-initialization
OutputDir = utils.string_utils.to_folder_path(OutputDir)
ConfigDir = utils.string_utils.to_folder_path(OutputDir + ConfigDir)
Evaluator.SaveDir = utils.string_utils.to_folder_path(OutputDir + Evaluator.SaveDir)
Evaluator.StatisticDir = utils.string_utils.to_folder_path(OutputDir + Evaluator.StatisticDir)
Simulator.MATLAB.OutputDir = utils.string_utils.to_folder_path(OutputDir + Simulator.MATLAB.OutputDir)