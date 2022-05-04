import config
import numpy as np
import utils.math
from .state import GazeboState
from envs.reward import Reward
from framework.configuration import global_configs as configs
from framework.geometry import Geometry
from framework.robot import Robot
from math import exp

class GazeboReward(Reward):
    def __init__(self, robot: Robot, obstacles: list[Geometry]) -> None:
        super().__init__(configs.get(config.Environment.Gazebo.MaxSteps_))
        self.robot = robot
        self.obstacles = obstacles

    def _update(self, action: np.ndarray, state: GazeboState) -> None:
        self.reward = 0
        self.done = False
        if state.collision:
            return
        d = utils.math.distance(state.achieved, state.desired)
        self.reward = self.__eval_reward(state)

    def __eval_reward(self, state: GazeboState) -> float:
        d_target = utils.math.distance(state.achieved, state.desired)
        d_obj = np.inf
        points = self.robot.collision_points(state.joint_position)[3:]
        for pos in points:
            d_obj = min(d_obj, pos[2])
            for obstacle in self.obstacles:
                d_obj = min(d_obj, obstacle.distance(pos))
        d_obj = max(d_obj, 0)
        reward = 1 if d_target < 0.05 else 0
        reward += exp(-0.69 * d_target * (1 + 1 / (d_obj * 2 + 1e-5)))
        reward *= 10
        assert 0 <= reward <= 20
        return reward