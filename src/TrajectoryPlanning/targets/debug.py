import numpy as np
from simulator.targets import GameState
from framework.replay_buffer import ReplayBuffer, Transition

def main():
    rb = ReplayBuffer(5)

    st1 = GameState()
    st1.achieved = np.array([1., 2.])
    st1.desired = np.array([3., 4.])
    st1.collision = False
    st1.joint_states = np.array([5., 6., 7.])

    st2 = GameState()
    st2.achieved = np.array([8., 9.])
    st2.desired = np.array([8., 9.])
    st2.collision = False
    st2.joint_states = np.array([8., 9., 8.])

    a = Transition(st1, np.array([1., 2.]), 12., st2)
    a2 = Transition(st2, np.array([1., 2.]), 12., st1)
    rb.append(a)
    rb.append(a2)
    rb.append(a)
    rb.append(a2)
    rb.append(a)
    rb.append(a2)

    d = rb.to_serializable()
    e = ReplayBuffer.from_serializable(d, 3)
    d = e.to_serializable()
    pass