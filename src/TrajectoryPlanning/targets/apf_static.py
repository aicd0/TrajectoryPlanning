import config
from envs import create_environment
from framework.algorithm.apf import ArtificialPotentialFieldPlanner
from framework.configuration import global_configs as configs

def main():
    sim, _ = create_environment('gazebo')
    planner = ArtificialPotentialFieldPlanner(sim, plot=True)
    
    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        sim.plot_reset()
        planner.reach(state.desired, verbose=True)
    sim.close()