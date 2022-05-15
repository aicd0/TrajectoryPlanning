from .dep.benchmark import benchmark
from .dep.rl import ReinforcementLearningPlanner
from envs import create_simulator

def main():
    sim = create_simulator('gazebo')
    planner = ReinforcementLearningPlanner(sim, plot=True)
    benchmark(sim, planner)
    sim.close()