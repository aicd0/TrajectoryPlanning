import config
import time
from framework.ddpg import Agent, Transition
from framework.game import GameState
from simulator.simulator import Simulator

def main():
    # Initialize the simulator
    sim = Simulator()
    # sim.eng.simPlotInit(nargout=0)
    
    # Reset simulator to get sizes of states and actions.
    state = sim.reset()
    dim_action = len(state.config)
    dim_state = len(state.as_input)

    # Load the agent from local checkpoints.
    agent = Agent(dim_state, dim_action)
    agent.try_load(config.CheckpointDir)

    # Initialize the game recorder.
    game = GameState()
    game.reset()

    # Sample an initial state.
    state = sim.reset()

    # Plot initialization.
    sim.plot_reset()

    # Logging.
    last_update_time = time.time()

    for step in range(1, config.Test.MaxStep + 1):
        action = agent.sample_action(state, noise=config.Test.NoiseEnabled, detach=config.Test.DetachAgent)
        state = sim.step(action)
        sim.plot_step()

        # Calculate reward.
        game.update(action, state)

        if game.game_over:
            break
        elif game.stage_over:
            state = sim.stage()

        if time.time() - last_update_time > 1:
            print('Step %d\r' % step, end='')
            last_update_time = time.time()

    print('Total steps: %d' % step)
    game.summary()
    