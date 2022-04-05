import numpy as np
import torch
import utils.string_utils

# Valid targets:
# - train
# - test
# - debug
Target = 'train'

OutputDir = 'output/gym'
ConfigDir = 'configs'

class DataType:
    Numpy = np.float32
    Torch = torch.float32

class Model:
    InitialWeight = 0.03

class Train:
    LoadFromPreviousSession = False

    class DDPG:
        MinLogStepInterval = 500
        FieldBatchSize = ('Train/DDPG/BatchSize', 64)
        FieldEpsilon = ('Train/DDPG/Epsilon', 20000)
        FieldGamma = ('Train/DDPG/Gamma', 0.99)
        FieldLRActor = ('Train/DDPG/LRActor', 0.0001)
        FieldLRCritic = ('Train/DDPG/LRCritic', 0.001)
        FieldMaxEpoches = ('Train/DDPG/MaxEpoches', 200000)
        FieldMaxIterations = ('Train/DDPG/MaxIterations', 10000)
        FieldNoiseEnabled = ('Train/DDPG/NoiseEnabled', True)
        FieldReplayBuffer = ('Train/DDPG/ReplayBuffer', 50000)
        FieldTau = ('Train/DDPG/Tau', 0.001)
        FieldWarmup = ('Train/DDPG/Warmup', 1000)

        class PER:
            FieldEnabled = ('Train/DDPG/PER/Enabled', False)
            FieldAlpha = ('Train/DDPG/PER/Alpha', 0.5)
            FieldK = ('Train/DDPG/PER/K', 0.01)

        class OUNoise:
            FieldMu = ('Train/DDPG/OUNoise/Mu', 0.0)
            FieldSigma = ('Train/DDPG/OUNoise/Sigma', 0.4)
            FieldTheta = ('Train/DDPG/OUNoise/Theta', 0.15)

        class UniformNoise:
            FieldMax = ('Train/DDPG/UniformNoise/Max', 1.0)
            FieldMin = ('Train/DDPG/UniformNoise/Min', -1.0)

    class HER:
        FieldEnabled = ('Train/HER/Enabled', True)
        FieldK = ('Train/HER/K', 4)

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
    # - matlab
    # - gym
    # - ros
    FieldPlatform = ('Simulator/Platform', 'gym')

    class Gym:
        # These environments have been tested and work fine:
        # - CartPole-v0
        # - CartPole-v1
        # - FetchReach-v1
        # - Pendulum-v1
        FieldEnvironment = ('Simulator/Gym/Environment', 'FetchReach-v1')

        # This option is only for Windows user.
        MujocoLibPath = 'C:/Users/stdcn/.mujoco/mjpro150/bin'

    class MATLAB:
        SessionFile = '../../output/MatlabLauncher/session.txt'
        OutputDir = 'MATLAB'

    class ROS:
        DynamicEnabled = False # has to be False
        FieldActionAmp = ('Simulator/ROS/ActionAmp', 0.1)
        FieldStepIterations = ('Simulator/ROS/StepIterations', 50) # has to be consistent to sensor frequencies

# Initializations
OutputDir = utils.string_utils.to_folder_path(OutputDir)
ConfigDir = utils.string_utils.to_folder_path(OutputDir + ConfigDir)
Evaluator.SaveDir = utils.string_utils.to_folder_path(OutputDir + Evaluator.SaveDir)
Evaluator.StatisticDir = utils.string_utils.to_folder_path(OutputDir + Evaluator.StatisticDir)
Simulator.MATLAB.OutputDir = utils.string_utils.to_folder_path(OutputDir + Simulator.MATLAB.OutputDir)