import config
import random
import simulator.simulator as sim
import time
from copy import copy
from framework.ddpg import Agent, Transition
from framework.game import GameState
from simulator.engine import Connector

def augment_replay_buffer(agent: Agent, replay_buffer: list[Transition], game: GameState):
    for i, trans in enumerate(replay_buffer):
        # Sample k transitions after this transition as new goals.
        sample_src = replay_buffer[i + 1:]
        sample_count = min(config.HER.K, len(sample_src))
        sampled_trans = random.sample(sample_src, sample_count)
        
        # Generate a new transition for each goal.
        for trans_goal in sampled_trans:
            new_state = copy(trans.state)
            new_state.desired = trans_goal.state.achieved
            new_state.from_matlab() # update auto-generated variables.

            new_next_state = copy(trans.next_state)
            new_next_state.desired = trans_goal.state.achieved
            new_next_state.from_matlab() # update auto-generated variables.

            new_action = trans.action
            game.update(new_action, new_next_state, test=True)
            new_trans = Transition(new_state, new_action, game.reward, new_next_state)
            agent.replay_buffer.append(new_trans)

def main():
    # Attach to a running session.
    connector = Connector()
    assert connector.connect()
    eng = connector.engine()

    # Initialize simulator.
    sim.initialize(eng)
    
    # Reset simulator to get sizes of states and actions.
    state = sim.reset(eng)
    dim_action = len(state.config)
    dim_state = len(state.as_input)

    # Initialize the agent.
    agent = Agent(dim_state, dim_action)

    # Initialize the game recorder.
    game = GameState()

    for episode in range(1, config.DDPG.MaxEpisode + 1):
        print('========== episode %d ==========' % episode)

        # Initialize replay buffer.
        replay_buffer: list[Transition] = []

        # Sample an initial state.
        state = sim.reset(eng)

        # Logging.
        total_reward = 0
        last_update_time = time.time()

        for step in range(config.DDPG.MaxStep):
            # Sample an action.
            action = agent.sample_action(state)

            # Execute the action.
            next_state = sim.step(eng, action)

            # Calculate reward.
            game.update(action, next_state)
            total_reward += game.reward

            # Add to replay buffer.
            trans = Transition(state, action, game.reward, next_state)
            agent.replay_buffer.append(trans)
            replay_buffer.append(trans)

            if game.game_over:
                next_state = sim.reset(eng)
            elif game.stage_over:
                next_state = sim.stage(eng)

            state = next_state

            if time.time() - last_update_time > 1:
                print('Step %d\r' % step, end='')
                last_update_time = time.time()
        
        print('Total reward: %d' % total_reward)
        
        # Augment replay buffer.
        if config.HER.Enable:
            augment_replay_buffer(agent, replay_buffer, game)
        
        # Optimize agent.
        agent.learn(config.DDPG.Iterations)