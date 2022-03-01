import config
import time
from framework.ddpg import Agent, Transition
from framework.game import GameState
from simulator.simulator import Simulator

def main():
    # Initialize the simulator
    sim = Simulator()
    
    # Reset simulator to get sizes of states and actions.
    state = sim.reset()
    dim_action = len(state.config)
    dim_state = len(state.as_input)

    # Load the agent from local checkpoints.
    agent = Agent(dim_state, dim_action)
    agent.load(config.CheckpointDir)

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
        action = agent.sample_action(state, noise=True)
        state = sim.step(action)
        sim.plot_step()

        # Calculate reward.
        game.update(action, state)

        if game.game_over:
            print('Collision detected.')
            break
        elif game.stage_over:
            print('Goal achieved.')
            break

        if time.time() - last_update_time > 1:
            print('Step %d\r' % step, end='')
            last_update_time = time.time()

    print('Simulation completed. ' + game.summary())
    