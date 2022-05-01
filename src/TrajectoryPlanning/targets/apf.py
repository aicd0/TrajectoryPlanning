import config
import utils.print
from envs.simulator import create_simulator
from framework.algorithm.apf import ArtificialPotentialFieldPlanner
from framework.configuration import global_configs as configs

def main():
    # Load environment.
    sim = create_simulator('ros')

    # Load planner.
    planner = ArtificialPotentialFieldPlanner(sim, sim.dim_action())
    planner.plot = True
    
    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        sim.plot_reset()
        success = planner.reach(state.desired)
        utils.print.put('Succeeded' if success else 'Failed')

    sim.close()