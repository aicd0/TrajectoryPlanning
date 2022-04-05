import config
import random
import time
import utils.print
import utils.string_utils
from copy import copy
from framework.configuration import Configuration
from framework.ddpg import Agent
from framework.evaluator import Evaluator
from framework.replay_buffer import Transition
from simulator import Game, Simulator

def augment_replay_buffered(replay_buffer: list[Transition]) -> list[Transition]:
    res: list[Transition] = []
    game = Game()
    game.reset()

    for i, trans in enumerate(replay_buffer):
        # Sample k transitions after this transition as new goals.
        sample_src = replay_buffer[i + 1:]
        sample_count = min(config.Train.HER.K, len(sample_src))
        sampled_trans = random.sample(sample_src, sample_count)
        
        # Generate a new transition for each goal.
        for trans_goal in sampled_trans:
            new_state = copy(trans.state)
            new_state.desired = trans_goal.state.achieved
            new_state.update() # notify changes.

            new_next_state = copy(trans.next_state)
            new_next_state.desired = trans_goal.state.achieved
            new_next_state.update() # notify changes.

            new_action = trans.action
            reward, _ = game.update(new_action, new_next_state)
            new_trans = Transition(new_state, new_action, reward, new_next_state)
            res.append(new_trans)
    
    return res

def main(configs: Configuration):
    sim = Simulator()
    game = Game()
    
    # Reset simulator to get sizes of states and actions.
    state = sim.reset()
    dim_action = sim.dim_action()
    dim_state = state.dim_state()

    # Initialize the agent.
    agent = Agent(dim_state, dim_action, 'controller')

    # Load evaluator.
    evaluator = Evaluator(agent)
    if config.Train.LoadFromPreviousSession: evaluator.load()

    # Logging.
    last_update_time = time.time()
    last_log_step = 0

    # Load from configs.
    max_epoches = configs.get('MaxEpoches', config.Train.DDPG.DefaultMaxEpoches)
    max_iters = configs.get('MaxIterations', config.Train.DDPG.DefaultMaxIterations)
    noise_enabled = configs.get('NoiseEnabled', config.Train.DDPG.DefaultNoiseEnabled)
    warmup = configs.get('Warmup', config.Train.DDPG.DefaultWarmup)
    epsilon = configs.get('Epsilon', config.Train.DDPG.DefaultEpsilon)
    her_enabled = configs.get('HER/Enabled', config.Train.HER.DefaultEnabled)
    
    while evaluator.get_epoch() <= max_epoches:
        epoch_replay_buffer: list[Transition] = []
        game.reset()
        state = sim.reset()
        done = False

        while not done and evaluator.get_iteration() <= max_iters:
            step = evaluator.get_step()

            # Sample an action and perform.
            if step < warmup:
                action = agent.sample_random_action()
            else:
                if noise_enabled:
                    noise_amount = 1 - step / epsilon
                else:
                    noise_amount = -1
                action = agent.sample_action(state, noise_amount=noise_amount)

            next_state = sim.step(action)

            # Calculate reward and add to replay buffer.
            reward, done = game.update(action, next_state)
            trans = Transition(state, action, reward, next_state)
            agent.replay_buffer.append(trans)
            epoch_replay_buffer.append(trans)
    
            # [optional] Optimize & save the agent.
            if step >= warmup:
                agent.learn()
                
            state = next_state
            evaluator.step(reward)

            if time.time() - last_update_time > 1:
                utils.print.put('[Train] %s' %
                    utils.string_utils.dict_to_str(evaluator.summary(shortterm=True)), same_line=True)
                last_update_time = time.time()
        
        # [optional] Perform HER.
        if her_enabled:
            epoch_replay_buffer = augment_replay_buffered(epoch_replay_buffer)
            for trans in epoch_replay_buffer:
                agent.replay_buffer.append(trans)

        evaluator.epoch(save=step >= warmup * 2)

        # [optional] Evaluate.
        if (step - last_log_step) >= config.Train.DDPG.MinLogStepInterval:
            last_log_step = step
            utils.print.put('[Evaluate] ' + utils.string_utils.dict_to_str(evaluator.summary()))
    
    sim.close()