import numpy as np
import utils.math
from copy import copy
from framework.agent import create_agent
from framework.algorithm.apf import apf1, apf2
from framework.planner import Planner
from framework.robot import Robot
from framework.workspace import Workspace
from math import pi

class UserPlanner1(Planner):
    def __init__(self, sim, **kwarg) -> None:
        super().__init__(sim, **kwarg)
        state = self.sim.reset()
        dim_action = self.sim.dim_action()
        dim_state = state.dim_state()
        self.agent = create_agent('sac', 'sac/h3', dim_state, dim_action, name='rl')
        self.agent.load(enable_learning=False)
    
    def _plan(self, position: np.ndarray) -> list[np.ndarray] | None:
        robot: Robot = self.sim.robot
        workspace: Workspace = self.sim.workspace
        state = copy(self.sim.state())
        last_joint_position = state.joint_position

        while True:
            current_joint_position = last_joint_position
            path = []
            local_minima = False
            collision = False

            while True:
                # Check collisions.
                d_obj = np.inf
                points = robot.collision_points(current_joint_position)
                for pos in points:
                    d_obj = min(d_obj, pos[2])
                    for obstacle in workspace.obstacles:
                        d_obj = min(d_obj, obstacle.distance(pos))
                d_obj = max(d_obj, 0)
                if d_obj < 0.05:
                    collision = True
                    break
                
                # Reach target.
                d_target = utils.math.distance(points[-1], position)
                if d_target < 0.05:
                    break

                # Generate next action.
                state.joint_position = current_joint_position
                state.achieved = points[-1]
                state.desired = position
                state.update()
                action = self.agent.sample_action(state, deterministic=True) * self.sim.action_amp
                next_joint_position = robot.clip(current_joint_position + action)
                
                # Check local minima.
                joint_delta = np.max(np.abs(next_joint_position - current_joint_position))
                if len(path) >= 50 or joint_delta < 2 * pi/180:
                    local_minima = True
                    break

                # Move to next state.
                path.append(next_joint_position)
                current_joint_position = next_joint_position

            if collision:
                finished = False
                idx = -1
                while True:
                    if idx >= len(path):
                        yield None
                        return

                    # Generate next action.
                    current_joint_position = last_joint_position if idx < 0 else path[idx]
                    next_joint_position = apf1(workspace, robot, current_joint_position, position)
                    
                    # Reach target.
                    points = robot.collision_points(next_joint_position)
                    d_target = utils.math.distance(points[-1], position)
                    if d_target < 0.05:
                        finished = True
                        break
                    
                    # Check local minima.
                    if not np.max(np.abs(next_joint_position - current_joint_position)) < 0.5 * pi/180:
                        break
                    idx += 1

                for joint_position in path[:idx + 1]:
                    yield joint_position
                yield next_joint_position
                last_joint_position = next_joint_position
                if finished:
                    break
                continue

            for joint_position in path:
                yield joint_position
                last_joint_position = joint_position

            if local_minima:
                current_joint_position = last_joint_position
                while True:
                    # Generate next action.
                    next_joint_position = apf1(workspace, robot, current_joint_position, position)

                    # Check local minima.
                    if np.max(np.abs(next_joint_position - current_joint_position)) < 0.5 * pi/180:
                        yield None
                        return
                    
                    # Perform the action.
                    yield next_joint_position
                    last_joint_position = next_joint_position

                    # Reach target.
                    points = robot.collision_points(next_joint_position)
                    d_target = utils.math.distance(points[-1], position)
                    if d_target < 0.05:
                        return

                    # Move to next state.
                    current_joint_position = next_joint_position
                # continue
            return