import numpy as np
import torch
import utils.string_utils

class Common:
    OutputDir = 'output/gazebo_sac'
    ConfigDir = 'configs'
    Target_ = ('Common/Target', 'train')

    class DataType:
        Numpy = np.float32
        Torch = torch.float32

class Environment:
    Platform_ = ('Environment/Platform', 'ros')
    MaxIterations_ = ('Environment/MaxIterations', 10000)

    class Gym:
        MujocoLibPath = 'C:/Users/stdcn/.mujoco/mjpro150/bin' # only for Windows
        Environment_ = ('Environment/Gym/Environment', 'HalfCheetah-v2')

    class MATLAB:
        OutputDir = 'MATLAB'
        SessionFile = '../../output/MatlabLauncher/session.txt'
        ActionAmp_ = ('Environment/MATLAB/ActionAmp', 0.1)

    class ROS:
        ROSLibPath = 'C:/opt/ros/noetic/x64/Lib/site-packages' # only for Windows
        ProjectLibPath = '../RobotSimulator/devel/lib/site-packages' # only for Windows
        ActionAmp_ = ('Environment/ROS/ActionAmp', 0.1)
        MaxIterations_ = ('Environment/ROS/MaxIterations', 150)
        StepIterations_ = ('Environment/ROS/StepIterations', 50) # make sure corresponding to sensor frequency

class Model:
    InitialWeight_ = ('Model/InitialWeight', 0.03)
    ModelGroup_ = ('Model/ModelGroup', 'sac/l3')

class Training:
    MinLogStepInterval = 500
    LoadFromPreviousSession_ = ('Training/LoadFromPreviousSession', False)
    MaxEpoches_ = ('Training/MaxEpoches', 200000)
    ProtectedEpoches_ = ('Training/ProtectedEpoches', 20)

    class Agent:
        Algorithm_ = ('Training/Agent/Algorithm', 'sac')
        BatchSize_ = ('Training/Agent/BatchSize', 64)
        Gamma_ = ('Training/Agent/Gamma', 0.99)
        LRActor_ = ('Training/Agent/LRActor', 0.0001)
        LRCritic_ = ('Training/Agent/LRCritic', 0.001)
        ReplayBuffer_ = ('Training/Agent/ReplayBuffer', 200000)
        Tau_ = ('Training/Agent/Tau', 0.001)
        Warmup_ = ('Training/Agent/Warmup', 1000)

        class ActionNoise:
            class Normal:
                Enabled_ = ('Training/Agent/ActionNoise/Normal/Enabled', True)
                Epsilon_ = ('Training/Agent/ActionNoise/Normal/Epsilon', 50000)
                Mu_ = ('Training/Agent/ActionNoise/Normal/Mu', 0.0)
                Sigma_ = ('Training/Agent/ActionNoise/Normal/Sigma', 0.2)
                Theta_ = ('Training/Agent/ActionNoise/Normal/Theta', 0.15)

        class HER:
            Enabled_ = ('Training/Agent/HER/Enabled', True)
            K_ = ('Training/Agent/HER/K', 2)

        class PER:
            Enabled_ = ('Training/Agent/PER/Enabled', False)
            Alpha_ = ('Training/Agent/PER/Alpha', 0.8)
            K_ = ('Training/Agent/PER/K', 0.01)

        class SAC:
            AutoEntropyTuning_ = ('Training/Agent/SAC/AutoEntropyTuning', True)
            LRAlpha_ = ('Training/Agent/SAC/LRAlpha', 0.0001)

class Testing:
    DetachAgent = False
    NoiseEnabled = False
    MaxEpoches = 100
    MaxIterations = 10000

class Evaluator:
    SaveDir = 'checkpoint'
    FigureDir = 'figures'
    EpochWindowSize_ = ('Evaluator/EpochWindowSize', 20)
    MinSaveStepInterval_ = ('Evaluator/MinSaveStepInterval', 1000)

    class Figure:
        DPI_ = ('Evaluator/Figure/DPI', 300)
        Height_ = ('Evaluator/Figure/Height', 5)
        Width_ = ('Evaluator/Figure/Width', 9)
        MaxSaveEpochInterval_ = ('Evaluator/Figure/MaxSaveEpochInterval', 10)

# Post-initialization
Common.OutputDir = utils.string_utils.to_folder_path(Common.OutputDir)
Common.ConfigDir = utils.string_utils.to_folder_path(Common.OutputDir + Common.ConfigDir)
Environment.MATLAB.OutputDir = utils.string_utils.to_folder_path(Common.OutputDir + Environment.MATLAB.OutputDir)
Environment.ROS.ProjectLibPath = utils.string_utils.to_folder_path(Environment.ROS.ProjectLibPath)
Evaluator.SaveDir = utils.string_utils.to_folder_path(Common.OutputDir + Evaluator.SaveDir)
Evaluator.FigureDir = utils.string_utils.to_folder_path(Common.OutputDir + Evaluator.FigureDir)