import config
from .dep.rl import ReinforcementLearningPlanner
from envs import create_environment

def main():
    sim, _ = create_environment('gazebo')
    planner = ReinforcementLearningPlanner(sim, plot=True)
    
    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        sim.plot_reset()
        planner.reach(state.desired, verbose=True)
    sim.close()