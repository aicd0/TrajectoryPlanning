import config
import numpy as np
import utils.math
from framework.planner import Planner
from framework.configuration import global_configs as configs

class ArtificialPotentialFieldPlanner(Planner):
    def __init__(self, sim, dim: int) -> None:
        super().__init__(sim)
        assert dim > 0
        self.eta = configs.get(config.ArtificialPotentialField.Eta_)
        self.samples = configs.get(config.ArtificialPotentialField.Samples_)
        self.step = configs.get(config.ArtificialPotentialField.Step_)
        self.zeta = configs.get(config.ArtificialPotentialField.Zeta_)
        self.dim = dim
        self.actions = np.array([utils.math.random_point_on_hypersphere(dim) for _ in range(self.samples)])
        self.actions *= self.step
    
    def reach(self, pos: np.ndarray) -> bool:
        while True:
            state = self.sim.state()
            d = utils.math.distance(state.achieved, pos)
            if d < 0.05:
                break
            joint_pos = self.__next(state.joint_position, pos)
            if not self._reach(joint_pos):
                return False
        return True

    def __potential(self, joint_position: np.ndarray, target_pos: np.ndarray) -> float:
        origins = self.sim.robot.origins(joint_position)
        d = utils.math.distance(origins[-1], target_pos)
        potential = self.zeta * (d - 1 / d)
        for pos in origins:
            for obstacle in self.sim.obstacles:
                d = obstacle.distance(pos)
                potential += self.eta / (d + 1e-5)
        return potential

    def __next(self, joint_position: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        min_potential = self.__potential(joint_position, target_pos)
        ans = joint_position
        for action in self.actions:
            new_joint_position = self.sim.robot.clip(joint_position + action)
            potential = self.__potential(new_joint_position, target_pos)
            if potential < min_potential:
                min_potential = potential
                ans = new_joint_position
        return ans