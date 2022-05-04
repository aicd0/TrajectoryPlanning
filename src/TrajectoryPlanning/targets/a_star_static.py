import config
from envs import create_environment
from framework.algorithm.a_star import AStarPlanner

def main():
    sim, _ = create_environment('gazebo')
    planner = AStarPlanner(sim, plot=True)
    
    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        sim.plot_reset()
        planner.reach(state.desired, verbose=True)
    sim.close()