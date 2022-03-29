import numpy as np
from simulator.targets import Game, GameState, Simulator

def main():
    sim = Simulator()
    dim_action = sim.dim_action()

    action = np.random.uniform(-0.7, 0.7, dim_action)
    state = sim.reset()
    state = sim.step(action)
    state = sim.step(action)
    state = sim.step(action)

    action = np.random.uniform(-0.7, 0.7, dim_action)
    state = sim.reset()
    state = sim.step(action)
    state = sim.step(action)
    state = sim.step(action)

    action = np.random.uniform(-0.7, 0.7, dim_action)
    state = sim.reset()
    state = sim.step(action)
    state = sim.step(action)
    state = sim.step(action)

    sim.close()