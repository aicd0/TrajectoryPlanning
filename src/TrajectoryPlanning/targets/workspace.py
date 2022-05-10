import utils.print
from framework.robot import Robot1
from framework.workspace import Workspace
from typing import Callable
from utils.stopwatch import Stopwatch

def test(func: Callable, times: int):
    stopwatch = Stopwatch()
    for _ in range(times):
        stopwatch.start()
        func()
        stopwatch.pause()
    return stopwatch.span() / times

def main():
    workspace = Workspace('C2234')
    assert workspace.load()
    robot = Robot1()

    utils.print.put('Testing nearest position...')
    position = workspace.sample_cartesian_space()
    x1 = workspace.nearest_nodes_from_position_1(position, max_d=0.05)
    x2 = workspace.nearest_nodes_from_position_2(position, max_d=0.05)
    assert len(x1) == len(x1.intersection(x2)) == len(x2)
    baseline = test(lambda: workspace.nearest_nodes_from_position_1(position, max_d=0.05), 10)
    v2 = test(lambda: workspace.nearest_nodes_from_position_2(position, max_d=0.05), 10)
    utils.print.put('v2: %.2fx (%fs)' % (baseline / v2, v2))

    utils.print.put('Testing nearest joint position...')
    joint_position = robot.random_joint_position()
    x1 = workspace.nearest_node_from_joint_position_1(joint_position)
    x2 = workspace.nearest_node_from_joint_position_2(joint_position)
    assert x1 is x2
    baseline = test(lambda: workspace.nearest_node_from_joint_position_1(joint_position), 10)
    v2 = test(lambda: workspace.nearest_node_from_joint_position_2(joint_position), 10)
    utils.print.put('v2: %.2fx (%fs)' % (baseline / v2, v2))