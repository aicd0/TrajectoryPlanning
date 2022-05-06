import config
import numpy as np
import utils.math
from copy import copy
from framework.configuration import global_configs as configs
from framework.geometry import Geometry
from framework.planner import Planner
from framework.robot import Robot
from framework.workspace import Workspace

def __potential_a(position: np.ndarray, target_position: np.ndarray) -> float:
    zeta = configs.get(config.ArtificialPotentialField.Zeta_)
    d = utils.math.distance(position, target_position)
    return zeta * (d - 1 / (d + 1e-5))

def __potential_b(points: list[np.ndarray], obstacles: list[Geometry]) -> float:
    eta = configs.get(config.ArtificialPotentialField.Eta_)
    potential = 0
    for pos in points:
        for obstacle in obstacles:
            d = obstacle.distance(pos)
            potential += eta / (d + 1e-5)
    return potential

def apf1(workspace: Workspace, robot: Robot, joint_position: np.ndarray,
         target_position: np.ndarray) -> np.ndarray:
    sample_count = configs.get(config.ArtificialPotentialField.SampleCount_)
    max_step = configs.get(config.ArtificialPotentialField.MaxStep_)

    dim = joint_position.shape[0]
    points = robot.collision_points(joint_position)
    actions = [utils.math.random_point_on_hypersphere(dim) for _ in range(sample_count)]
    actions.append(np.zeros(dim))
    actions = max_step * np.array(actions)
    
    min_potential = np.inf
    for action in actions:
        new_joint_position = robot.clip(joint_position + action)
        points = robot.collision_points(new_joint_position)
        potential_a = __potential_a(points[-1], target_position)
        potential_b = __potential_b(points, workspace.obstacles)
        potential = potential_a + potential_b
        if potential < min_potential:
            min_potential = potential
            opt_action = new_joint_position
    return opt_action

def apf2(workspace: Workspace, robot: Robot, joint_position: np.ndarray,
         target_position: np.ndarray) -> list[np.ndarray]:
    finish_nodes = workspace.nearest_positions(target_position)
    if len(finish_nodes) <= 0:
        return None
    path = []
    current_node = workspace.nearest_joint_position(joint_position)
    workspace_changed = False

    while True:
        path.append(current_node.joint_position)
        if current_node in finish_nodes:
            break
        min_potential = np.inf
        neighbours = copy(current_node.neighbours)
        neighbours.append(current_node)
        for node in neighbours:
            potential_a = __potential_a(node.position, target_position)
            potential_b = workspace.get_meta('potential', node)
            if potential_b is None:
                points = robot.collision_points(node.joint_position)
                potential_b = __potential_b(points, workspace.obstacles)
                workspace.set_meta('potential', node, potential_b)
                workspace_changed = True
            potential = potential_a + potential_b
            if potential < min_potential:
                min_potential = potential
                next_node = node
        if next_node is current_node:
            return None
        current_node = next_node
    if workspace_changed:
        workspace.save()
    return path

class ArtificialPotentialFieldPlanner(Planner):
    def __init__(self, sim, resampling=False, **kwarg) -> None:
        super().__init__(sim, **kwarg)
        self.resampling = resampling
    
    def _reach(self, position: np.ndarray) -> bool:
        if self.resampling:
            while True:
                state = self.sim.state()
                d = utils.math.distance(state.achieved, position)
                if d < 0.05:
                    success = True
                    break
                joint_pos = apf1(self.sim.workspace, self.sim.robot, state.joint_position, position)
                if np.max(np.abs(joint_pos - state.joint_position)) < 1e-5:
                    success = False
                    break
                if not self._simple_reach(joint_pos):
                    success = False
                    break
            return success
        else:
            state = self.sim.state()
            path = apf2(self.sim.workspace, self.sim.robot, state.joint_position, position)
            if path is None:
                return False
            for joint_position in path:
                if not self._simple_reach(joint_position):
                    return False
            return True