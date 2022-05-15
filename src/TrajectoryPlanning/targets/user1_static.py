from .dep.benchmark import benchmark
from .dep.user1 import UserPlanner1
from envs import create_simulator

def main():
    sim = create_simulator('gazebo')
    planner = UserPlanner1(sim, plot=True)
    benchmark(sim, planner)
    sim.close()