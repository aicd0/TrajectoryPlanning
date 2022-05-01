import config
import utils.print
from envs import create_environment
from framework.algorithm.a_star import AStarPlanner
from framework.configuration import global_configs as configs

def main():
    # Load environment.
    sim, _ = create_environment('gazebo')

    # Load planner.
    planner = AStarPlanner(sim)
    planner.plot = True
    
    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        sim.plot_reset()
        success = planner.reach(state.desired)
        utils.print.put('Succeeded' if success else 'Failed')

    sim.close()