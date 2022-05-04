import numpy as np
import utils.print
from envs import create_environment
from framework.robot import Robot1

def main():
    sim, _ = create_environment('gazebo')
    robot = Robot1()

    while True:
        state = sim.reset()
        joint_position = state.joint_position
        origins = robot.origins(joint_position)
        actual = state.achieved
        predict = origins[-1]
        utils.print.put('Actual:')
        utils.print.put(actual)
        utils.print.put('Predict:')
        utils.print.put(predict)
        utils.print.put('Max error:')
        utils.print.put(np.max(np.abs(actual - predict)), end='')
        input()
        utils.print.put('=' * 30)