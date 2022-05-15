from .dep.benchmark import benchmark
from envs import create_simulator
from framework.algorithm.a_star import AStarPlanner

def main():
    sim = create_simulator('gazebo')
    planner = AStarPlanner(sim, plot=True)
    benchmark(sim, planner)
    sim.close()