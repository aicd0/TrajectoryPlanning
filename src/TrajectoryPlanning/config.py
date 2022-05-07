import numpy as np
import torch
from utils.string_utils import to_folder_path as todir

class Common:
    ProjectDir = 'main'
    ConfigDir = 'configs'
    Target_ = ('Common/Target', 'rl_train')

    class DataType:
        Numpy = np.float32
        Torch = torch.float32

class Environment:
    MaxIterations_ = ('Environment/MaxIterations', 10000)

    class Gazebo:
        ROSLibPath = 'C:/opt/ros/noetic/x64/Lib/site-packages' # only for Windows
        ProjectLibPath = '../RobotSimulator/devel/lib/site-packages' # only for Windows
        ActionAmp_ = ('Environment/Gazebo/ActionAmp', 0.1)
        MaxSteps_ = ('Environment/Gazebo/MaxSteps', 150)
        StepIterations_ = ('Environment/Gazebo/StepIterations', 50) # make sure corresponding to sensor frequency
        Workspace_ = ('Environment/Gazebo/Workspace', 'C2234')
        WorkspaceMinR_ = ('Environment/Gazebo/WorkspaceMinR', 0.05)

    class Gym:
        MujocoLibPath = 'C:/Users/stdcn/.mujoco/mjpro150/bin' # only for Windows
        Environment_ = ('Environment/Gym/Environment', 'HalfCheetah-v2')

    class MATLAB:
        OutputDir = 'MATLAB'
        SessionFile = '../../output/MatlabLauncher/session.txt'
        ActionAmp_ = ('Environment/MATLAB/ActionAmp', 0.1)

class Model:
    InitialWeight_ = ('Model/InitialWeight', 0.03)

class Agent:
    SaveDir = 'agents'
    BatchSize_ = ('Agent/BatchSize', 64)
    Gamma_ = ('Agent/Gamma', 0.99)
    LRActor_ = ('Agent/LRActor', 0.0001)
    LRCritic_ = ('Agent/LRCritic', 0.001)
    ReplayBuffer_ = ('Agent/ReplayBuffer', 400000)
    Tau_ = ('Agent/Tau', 0.001)
    Warmup_ = ('Agent/Warmup', 1000)

    class ActionNoise:
        class Normal:
            Enabled_ = ('Agent/ActionNoise/Normal/Enabled', True)
            Epsilon_ = ('Agent/ActionNoise/Normal/Epsilon', 50000)
            Mu_ = ('Agent/ActionNoise/Normal/Mu', 0.0)
            Sigma_ = ('Agent/ActionNoise/Normal/Sigma', 0.2)
            Theta_ = ('Agent/ActionNoise/Normal/Theta', 0.15)

    class HER:
        Enabled_ = ('Agent/HER/Enabled', True)
        K_ = ('Agent/HER/K', 2)

    class PER:
        Enabled_ = ('Agent/PER/Enabled', False)
        Alpha_ = ('Agent/PER/Alpha', 0.8)
        K_ = ('Agent/PER/K', 0.01)

    class SAC:
        AutoEntropyTuning_ = ('Agent/SAC/AutoEntropyTuning', True)
        LRAlpha_ = ('Agent/SAC/LRAlpha', 0.0001)

class Training:
    MinLogStepInterval = 500
    LoadFromPreviousSession_ = ('Training/LoadFromPreviousSession', False)
    MaxEpoches_ = ('Training/MaxEpoches', 200000)
    ProtectedEpoches_ = ('Training/ProtectedEpoches', 20)

class Testing:
    DetachAgent = False
    NoiseEnabled = False
    MaxEpoches = 20
    MaxIterations = 10000

class Evaluator:
    SaveDir = 'evaluators'
    FigureDir = 'figures'
    EpochWindowSize_ = ('Evaluator/EpochWindowSize', 20)
    MinSaveStepInterval_ = ('Evaluator/MinSaveStepInterval', 1000)

    class Figure:
        DPI_ = ('Evaluator/Figure/DPI', 300)
        Height_ = ('Evaluator/Figure/Height', 5)
        Width_ = ('Evaluator/Figure/Width', 9)
        MaxSaveEpochInterval_ = ('Evaluator/Figure/MaxSaveEpochInterval', 10)

class Workspace:
    SaveDir = 'workspace'
    ObstacleMargin_ = ('Workspace/ObstacleMargin', 0.1)

class ArtificialPotentialField:
    Eta_ = ('ArtificialPotentialField/Eta', 1.0)
    SampleCount_ = ('ArtificialPotentialField/SampleCount', 200)
    MaxStep_ = ('ArtificialPotentialField/MaxStep', 0.1)
    Zeta_ = ('ArtificialPotentialField/Zeta', 10.0)

class Export:
    SaveDir = 'export'

# Post-initialization
__output_dir = todir('output')
Common.ProjectDir = todir(__output_dir + Common.ProjectDir)
Common.ConfigDir = todir(Common.ProjectDir + Common.ConfigDir)
Environment.MATLAB.OutputDir = todir(Common.ProjectDir + Environment.MATLAB.OutputDir)
Environment.Gazebo.ProjectLibPath = todir(Environment.Gazebo.ProjectLibPath)
Agent.SaveDir = todir(Common.ProjectDir + Agent.SaveDir)
Evaluator.SaveDir = todir(Common.ProjectDir + Evaluator.SaveDir)
Evaluator.FigureDir = todir(Common.ProjectDir + Evaluator.FigureDir)
Workspace.SaveDir = todir(__output_dir + Workspace.SaveDir)
Export.SaveDir = todir(__output_dir + Export.SaveDir)