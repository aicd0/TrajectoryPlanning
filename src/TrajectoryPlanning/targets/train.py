import config
import random
import time
from copy import copy
from framework.ddpg import Agent, Transition
from framework.game import GameState
from simulator.simulator import Simulator

def augment_replay_buffer(replay_buffer: list[Transition]) -> None:
    augmented_replay_buffer: list[Transition] = []
    game = GameState()
    game.reset()

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
            game.update(new_action, new_next_state)
            new_trans = Transition(new_state, new_action, game.reward, new_next_state)
            augmented_replay_buffer.append(new_trans)
    
    replay_buffer.extend(augmented_replay_buffer)

def main():
    sim = Simulator()
    game = GameState()
    
    # Reset simulator to get sizes of states and actions.
    state = sim.reset()
    dim_action = len(state.config)
    dim_state = len(state.as_input)

    # Initialize the agent.
    agent = Agent(dim_state, dim_action)
    agent.try_load(config.CheckpointDir)

    for episode in range(1, config.DDPG.MaxEpisode + 1):
        print('========== episode %d ==========' % episode)
        last_update_time = time.time()
        step = 0

        while step < config.DDPG.MinSteps:
            replay_buffer: list[Transition] = []
            game.reset()
            state = sim.reset()

            while step < config.DDPG.MinSteps:
                step += 1

                # Sample and execute the action.
                action = agent.sample_action(state, noise=config.DDPG.NoiseEnabled)
                next_state = sim.step(action)

                # Calculate reward and add to replay buffer.
                game.update(action, next_state)
                trans = Transition(state, action, game.reward, next_state)
                replay_buffer.append(trans)

                if game.game_over:
                    break
                elif game.stage_over:
                    next_state = sim.stage()
                state = next_state

                if time.time() - last_update_time > 1:
                    print('Step %d\r' % step, end='')
                    last_update_time = time.time()
            
            game.summary()
            
            # Augment replay buffer.
            if config.HER.Enable:
                augment_replay_buffer(replay_buffer)
            
            # Add to agent replay buffer.
            for trans in replay_buffer:
                agent.replay_buffer.append(trans)
        
        # Optimize agent.
        agent.learn()
        agent.save(config.CheckpointDir)
        