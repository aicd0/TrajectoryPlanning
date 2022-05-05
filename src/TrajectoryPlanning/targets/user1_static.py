import config
from .dep.user1 import UserPlanner1
from envs import create_simulator

def main():
    sim = create_simulator('gazebo')
    planner = UserPlanner1(sim, plot=True)
    
    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        sim.plot_reset()
        planner.reach(state.desired, verbose=True)
    sim.close()