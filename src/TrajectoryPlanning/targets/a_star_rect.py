import numpy as np
import utils.export
import utils.print
from envs import create_environment
from framework.algorithm.a_star import AStarPlanner
    
track = [
    np.array([-0.4, -0.2, 0.3]),
    np.array([-0.4,  0.2, 0.3]),
    np.array([-0.4,  0.2, 0.7]),
    np.array([-0.4, -0.2, 0.7]),
    np.array([-0.4, -0.2, 0.3]),
]

def main():
    sim, _ = create_environment('gazebo')
    planner = AStarPlanner(sim, plot=True)

    sim.reset()
    assert planner.reach(track[0], verbose=True)
    sim.record = True

    for position in track:
        assert planner.reach(position, verbose=True)
    sim.close()

    states = [s.joint_position for s in sim.records]
    utils.export.make_rect('a_star', states, 0.014296)