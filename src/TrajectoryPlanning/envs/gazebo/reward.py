import config
import numpy as np
import utils.math
from .state import GazeboState
from envs.reward import Reward
from framework.configuration import global_configs as configs
from framework.robot import Robot
from framework.workspace import Workspace
from math import exp

class GazeboReward(Reward):
    def __init__(self, robot: Robot, workspace: Workspace) -> None:
        super().__init__(configs.get(config.Environment.Gazebo.MaxSteps_))
        self.robot = robot
        self.workspace = workspace

    def _update(self, state: GazeboState, action: np.ndarray, next_state: GazeboState) -> None:
        self.done = False

        # Collision check.
        if next_state.collision:
            self.reward = 0
            if state.collision:
                self.done = True
            return

        # Target reached.
        d_target = utils.math.distance(next_state.achieved, next_state.desired)
        if d_target < 0.05:
            self.reward = 20
            last_d_target = utils.math.distance(state.achieved, state.desired)
            if last_d_target < 0.05:
                self.done = True
            return

        # Normal reward.
        d_obj = np.inf
        points = self.robot.collision_points(next_state.joint_position)
        for pos in points:
            d_obj = min(d_obj, pos[2])
            for obstacle in self.workspace.obstacles:
                d_obj = min(d_obj, obstacle.distance(pos))
        d_obj = max(d_obj, 0)
        reward = exp(-0.69 * d_target * (1 + 1 / (d_obj * 2 + 1e-5)))
        reward *= 10
        assert 0 <= reward <= 10
        self.reward = reward