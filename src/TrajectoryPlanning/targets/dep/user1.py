import numpy as np
import utils.math
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
        self.agent = create_agent('sac', 'sac/l3', dim_state, dim_action, name='rl')
        self.agent.load(enable_learning=False)
    
    def _reach(self, position: np.ndarray) -> bool:
        robot: Robot = self.sim.robot
        workspace: Workspace = self.sim.workspace
        available = [True, True]
        prefer = [True, True]
        method_count = 2

        while True:
            state = self.sim.state()
            d_target = utils.math.distance(state.achieved, position)
            if d_target < 0.05:
                success = True
                break
            state.desired = position

            d_obj = np.inf
            points = robot.collision_points(state.joint_position)[3:]
            for pos in points:
                d_obj = min(d_obj, pos[2])
                for obstacle in workspace.obstacles:
                    d_obj = min(d_obj, obstacle.distance(pos))
            d_obj = max(d_obj, 0)

            if d_obj < 0.15:
                prefer[0] = False
            elif d_obj > 0.3:
                prefer[0] = True

            # Determine which method to use.
            method = None
            for i in range(method_count):
                if available[i] and prefer[i]:
                    method = i
                    break
            if method is None:
                for i in range(method_count):
                    if available[i]:
                        method = i
                        break
            if method is None:
                success = False
                break

            if method == 0:
                action = self.agent.sample_action(state, deterministic=True)
                if not self._simple_act(action, preamp=False):
                    success = False
                    break
                new_state = self.sim.state()
                if np.max(np.abs(new_state.joint_position - state.joint_position)) < 0.5 * pi/180:
                    available[0] = False
                prefer[1] = True
            else:
                new_joint_position = apf1(workspace, robot, state.joint_position, position)
                if not self._simple_reach(new_joint_position):
                    success = False
                    break
                new_state = self.sim.state()
                if np.max(np.abs(new_state.joint_position - state.joint_position)) < 0.5 * pi/180:
                    available[1] = False
        return success