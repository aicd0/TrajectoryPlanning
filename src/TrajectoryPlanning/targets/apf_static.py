from .dep.benchmark import benchmark
from envs import create_simulator
from framework.algorithm.apf import ArtificialPotentialFieldPlanner

def main():
    sim = create_simulator('gazebo')
    planner = ArtificialPotentialFieldPlanner(sim, resampling=False, plot=True)
    benchmark(sim, planner)
    sim.close()