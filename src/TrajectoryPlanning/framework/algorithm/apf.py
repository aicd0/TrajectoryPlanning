import config
import numpy as np
import utils.math
from copy import copy
from framework.configuration import global_configs as configs
from framework.geometry import Geometry
from framework.planner import Planner
from framework.robot import Robot
from framework.workspace import Workspace
from typing import Generator

def potential_a(position: np.ndarray, target_position: np.ndarray) -> float:
    zeta = configs.get(config.ArtificialPotentialField.Zeta_)
    d = utils.math.distance(position, target_position)
    return zeta * (d - 1 / (d + 1e-5))

def potential_b(points: list[np.ndarray], obstacles: list[Geometry]) -> float:
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
    
    min_p = np.inf
    for action in actions:
        new_joint_position = robot.clip(joint_position + action)
        points = robot.collision_points(new_joint_position)
        pa = potential_a(points[-1], target_position)
        pb = potential_b(points, workspace.obstacles)
        p = pa + pb
        if p < min_p:
            min_p = p
            opt_action = new_joint_position
    return opt_action

def apf2(workspace: Workspace, robot: Robot, joint_position: np.ndarray,
         target_position: np.ndarray) -> list[np.ndarray] | None:
    finish_nodes = workspace.nearest_nodes_from_position_2(target_position, max_d=0.05)
    if len(finish_nodes) <= 0:
        return None
    current_node = workspace.nearest_node_from_joint_position_2(joint_position)
    path = []

    def eval_p(node):
        pa = potential_a(node.position, target_position)
        pb = workspace.get_meta('potential', node)
        return pa + pb

    current_p = eval_p(current_node)
    while True:
        path.append(current_node.joint_position)
        if current_node in finish_nodes:
            break
        min_p = np.inf
        for node in current_node.neighbours:
            p = eval_p(node)
            if p < min_p:
                min_p = p
                next_node = node
        if min_p >= current_p:
            return None
        current_node = next_node
        current_p = min_p
    return path

class ArtificialPotentialFieldPlanner(Planner):
    def __init__(self, sim, resampling=True, **kwarg) -> None:
        super().__init__(sim, **kwarg)
        self.resampling = resampling

        # Post initialization.
        workspace = self.sim.workspace
        robot: Robot = self.sim.robot

        if not self.resampling:
            workspace_changed = False
            for node in workspace.nodes:
                pb = workspace.get_meta('potential', node)
                if pb is None:
                    points = robot.collision_points(node.joint_position)
                    pb = potential_b(points, workspace.obstacles)
                    workspace.set_meta('potential', node, pb)
                    workspace_changed = True
            if workspace_changed:
                workspace.save()
    
    def _plan(self, position: np.ndarray) -> Generator[np.ndarray | None, None, None]:
        workspace: Workspace = self.sim.workspace
        robot: Robot = self.sim.robot
        state = self.sim.state()

        if self.resampling:
            current_joint_position = state.joint_position
            while True:
                yield current_joint_position

                # Reach target.
                points = robot.collision_points(current_joint_position)
                d = utils.math.distance(points[-1], position)
                if d < 0.05:
                    break
                
                # Generate next action.
                next_joint_position = apf1(workspace, robot, current_joint_position, position)
                
                # Check for local minima.
                if np.max(np.abs(next_joint_position - current_joint_position)) < 1e-5:
                    yield None
                    break

                # Move to next state.
                current_joint_position = next_joint_position
        else:
            track = apf2(workspace, robot, state.joint_position, position)
            if track is None:
                yield None
            else:
                for joint_position in track:
                    yield joint_position