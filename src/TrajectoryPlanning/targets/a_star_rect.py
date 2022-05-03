import config
import numpy as np
import utils.export
import utils.print
from envs import create_environment
from framework.algorithm.a_star import AStarPlanner
from framework.configuration import global_configs as configs

def main():
    # Load environment.
    sim, _ = create_environment('gazebo')

    # Load planner.
    planner = AStarPlanner(sim)
    planner.plot = True
    
    track = [
        np.array([-0.2, 0.4, 0.3]),
        np.array([0.2, 0.4, 0.3]),
        np.array([0.2, 0.4, 0.7]),
        np.array([-0.2, 0.4, 0.7]),
    ]

    sim.reset()
    assert planner.reach(track[0])
    sim.record = True

    for position in track:
        assert planner.reach(position)
    sim.close()

    states = [s.joint_position for s in sim.records]
    utils.export.make_rect('a_star', states, 0.014296)