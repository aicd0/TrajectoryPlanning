from framework.configuration import global_configs as configs
from envs.simulator import create_simulator
from framework.robot import Robot1

def main():
    # Load environment.
    sim = create_simulator('ros')

    while True:
        state = sim.reset()
        joint_position = state.joint_position
        robot = Robot1()
        origins = robot.origins(joint_position)
        predict = origins[-1]
        actual = state.achieved
        pass