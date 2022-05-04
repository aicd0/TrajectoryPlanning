import config
import numpy as np
import utils.math
from framework.configuration import global_configs as configs
from framework.planner import Planner

class ArtificialPotentialFieldPlanner(Planner):
    def __init__(self, sim, **kwarg) -> None:
        super().__init__(sim, **kwarg)
        self.eta = configs.get(config.ArtificialPotentialField.Eta_)
        self.samples = configs.get(config.ArtificialPotentialField.Samples_)
        self.step = configs.get(config.ArtificialPotentialField.Step_)
        self.zeta = configs.get(config.ArtificialPotentialField.Zeta_)
    
    def _reach(self, pos: np.ndarray) -> bool:
        while True:
            state = self.sim.state()
            d = utils.math.distance(state.achieved, pos)
            if d < 0.05:
                success = True
                break
            joint_pos = self.__next(state.joint_position, pos)
            if np.max(np.abs(joint_pos - state.joint_position)) < 1e-5:
                success = False
                break
            if not self._simple_reach(joint_pos):
                success = False
                break
        return success

    def __potential(self, joint_position: np.ndarray, target_pos: np.ndarray) -> float:
        points = self.sim.robot.collision_points(joint_position)
        d = utils.math.distance(points[-1], target_pos)
        potential = self.zeta * (d - 1 / (d + 1e-5))
        for pos in points:
            for obstacle in self.sim.obstacles:
                d = obstacle.distance(pos)
                potential += self.eta / (d + 1e-5)
        return potential

    def __next(self, joint_position: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        dim = joint_position.shape[0]
        actions = np.array([utils.math.random_point_on_hypersphere(dim) for _ in range(self.samples)]) * self.step
        min_potential = self.__potential(joint_position, target_pos)
        ans = joint_position
        for action in actions:
            new_joint_position = self.sim.robot.clip(joint_position + action)
            potential = self.__potential(new_joint_position, target_pos)
            if potential < min_potential:
                min_potential = potential
                ans = new_joint_position
        return ans