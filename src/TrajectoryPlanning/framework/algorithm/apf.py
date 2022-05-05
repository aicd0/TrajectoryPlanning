import config
import numpy as np
import utils.math
from framework.configuration import global_configs as configs
from framework.geometry import Geometry
from framework.robot import Robot
from framework.planner import Planner

def __potential(points: list[np.ndarray], target_pos: np.ndarray, obstacles: list[Geometry]) -> float:
    eta = configs.get(config.ArtificialPotentialField.Eta_)
    zeta = configs.get(config.ArtificialPotentialField.Zeta_)

    d = utils.math.distance(points[-1], target_pos)
    potential = zeta * (d - 1 / (d + 1e-5))
    for pos in points:
        for obstacle in obstacles:
            d = obstacle.distance(pos)
            potential += eta / (d + 1e-5)
    return potential

def apf(robot: Robot, joint_position: np.ndarray, target_pos: np.ndarray, obstacles: list[Geometry]) -> np.ndarray:
    sample_count = configs.get(config.ArtificialPotentialField.SampleCount_)
    max_step = configs.get(config.ArtificialPotentialField.MaxStep_)

    dim = joint_position.shape[0]
    points = robot.collision_points(joint_position)
    step = min(max_step, utils.math.distance(points[-1], target_pos))
    actions = step * np.array([utils.math.random_point_on_hypersphere(dim) for _ in range(sample_count)])
    
    min_potential = __potential(points, target_pos, obstacles)
    ans = joint_position
    for action in actions:
        new_joint_position = robot.clip(joint_position + action)
        points = robot.collision_points(new_joint_position)
        potential = __potential(points, target_pos, obstacles)
        if potential < min_potential:
            min_potential = potential
            ans = new_joint_position
    return ans

class ArtificialPotentialFieldPlanner(Planner):
    def __init__(self, sim, **kwarg) -> None:
        super().__init__(sim, **kwarg)
    
    def _reach(self, position: np.ndarray) -> bool:
        while True:
            state = self.sim.state()
            d = utils.math.distance(state.achieved, position)
            if d < 0.05:
                success = True
                break
            joint_pos = apf(self.sim.robot, state.joint_position, position, self.sim.obstacles)
            if np.max(np.abs(joint_pos - state.joint_position)) < 1e-5:
                success = False
                break
            if not self._simple_reach(joint_pos):
                success = False
                break
        return success