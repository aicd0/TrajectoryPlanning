import numpy as np
import utils.math
from framework.agent import create_agent
from framework.algorithm.apf import apf
from framework.geometry import Geometry
from framework.planner import Planner
from framework.robot import Robot
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
        obstacles: list[Geometry] = self.sim.obstacles
        agent_available = True
        apf_available = True

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
                for obstacle in obstacles:
                    d_obj = min(d_obj, obstacle.distance(pos))
            d_obj = max(d_obj, 0)

            # Determine which method to use.
            if d_obj < 0.15:
                agent_available = False
            elif d_target > 0.2:
                agent_available = True

            if agent_available:
                use_agent = True
            elif apf_available:
                use_agent = False
            else:
                use_agent = True

            if use_agent:
                action = self.agent.sample_action(state, deterministic=True)
                if not self._simple_act(action, preamp=False):
                    success = False
                    break
                new_state = self.sim.state()
                if np.max(np.abs(new_state.joint_position - state.joint_position)) < 2 * pi/180:
                    agent_available = False
                apf_available = True
            else:
                new_joint_position = apf(robot, state.joint_position, position, obstacles)
                if not self._simple_reach(new_joint_position):
                    success = False
                    break
                new_state = self.sim.state()
                if np.max(np.abs(new_state.joint_position - state.joint_position)) < 0.5 * pi/180:
                    apf_available = False

        return success