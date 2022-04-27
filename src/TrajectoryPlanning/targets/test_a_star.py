import config
import utils.print
from envs.simulator import create_simulator
from framework.agent import AgentBase, create_agent
from framework.algorithm.a_star import AStarPlanner
from framework.configuration import global_configs as configs

def main():
    # Load environment.
    sim = create_simulator('ros')
    state = sim.reset()
    dim_action = sim.dim_action()
    dim_state = state.dim_state()

    # Load agent.
    iks: AgentBase = create_agent('sac', dim_state, dim_action, name='iks')
    iks.load(enable_learning=False)

    # Load planner.
    planner = AStarPlanner(sim, iks)
    planner.plot = True
    
    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        sim.plot_reset()
        success = planner.reach(state.desired)
        utils.print.put('Succeeded' if success else 'Failed')

    sim.close()