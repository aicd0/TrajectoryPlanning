import numpy as np
import utils.math
from copy import copy
from framework.agent import create_agent
from framework.planner import Planner
from framework.robot import Robot
from typing import Generator

class ReinforcementLearningPlanner(Planner):
    def __init__(self, sim, **kwarg) -> None:
        super().__init__(sim, **kwarg)
        state = self.sim.reset()
        dim_action = self.sim.dim_action()
        dim_state = state.dim_state()
        self.agent = create_agent('sac', 'sac/h3', dim_state, dim_action, name='rl')
        self.agent.load(enable_learning=False)
    
    def _plan(self, position: np.ndarray) -> Generator[np.ndarray | None, None, None]:
        robot: Robot = self.sim.robot
        state = copy(self.sim.state())
        current_joint_position = state.joint_position
        step = 0
        
        while True:
            step += 1
            yield current_joint_position
            
            # Reach target.
            points = robot.collision_points(current_joint_position)
            d = utils.math.distance(points[-1], position)
            if d < 0.05:
                break

            # Generate next action.
            state.joint_position = current_joint_position
            state.achieved = points[-1]
            state.desired = position
            state.update()
            action = self.agent.sample_action(state, deterministic=True) * self.sim.action_amp
            next_joint_position = robot.clip(current_joint_position + action)
            
            # Check for local minima.
            if step >= 150 or np.max(np.abs(next_joint_position - current_joint_position)) < 1e-5:
                yield None
                break

            # Move to next state.
            current_joint_position = next_joint_position