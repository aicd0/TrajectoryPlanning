import numpy as np
import utils.math
from framework.agent import create_agent
from framework.planner import Planner

class ReinforcementLearningPlanner(Planner):
    def __init__(self, sim, **kwarg) -> None:
        super().__init__(sim, **kwarg)
        state = self.sim.reset()
        dim_action = self.sim.dim_action()
        dim_state = state.dim_state()
        self.agent = create_agent('sac', 'sac/l3', dim_state, dim_action, name='rl')
        self.agent.load(enable_learning=False)
    
    def _reach(self, position: np.ndarray) -> bool:
        while True:
            state = self.sim.state()
            d = utils.math.distance(state.achieved, position)
            if d < 0.05:
                success = True
                break
            state.desired = position
            action = self.agent.sample_action(state, deterministic=True)
            if not self._simple_act(action, preamp=False):
                success = False
                break
            new_state = self.sim.state()
            if np.max(np.abs(new_state.joint_position - state.joint_position)) < 1e-5:
                success = d < 0.2
                break
        return success