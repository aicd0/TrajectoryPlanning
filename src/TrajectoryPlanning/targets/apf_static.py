import config
from envs import create_simulator
from framework.algorithm.apf import ArtificialPotentialFieldPlanner

def main():
    sim = create_simulator('gazebo')
    planner = ArtificialPotentialFieldPlanner(sim, plot=True)
    
    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        sim.plot_reset()
        planner.reach(state.desired, verbose=True)
    sim.close()