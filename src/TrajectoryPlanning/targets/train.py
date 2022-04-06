import config
import framework.her
import time
import utils.print
import utils.string_utils
from framework.configuration import global_configs as configs
from framework.ddpg import Agent
from framework.evaluator import Evaluator
from framework.replay_buffer import Transition
from simulator import Game, Simulator

def main():
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
    max_epoches = configs.get(config.Train.DDPG.FieldMaxEpoches)
    max_iters = configs.get(config.Train.DDPG.FieldMaxIterations)
    noise_enabled = configs.get(config.Train.DDPG.FieldNoiseEnabled)
    warmup = configs.get(config.Train.DDPG.FieldWarmup)
    epsilon = configs.get(config.Train.DDPG.FieldEpsilon)
    her_enabled = configs.get(config.Train.HER.FieldEnabled)
    her_k = configs.get(config.Train.HER.FieldK)
    
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
            epoch_replay_buffer = framework.her.augment_replay_buffer(epoch_replay_buffer, her_k)
            for trans in epoch_replay_buffer:
                agent.replay_buffer.append(trans)

        evaluator.epoch(allow_save=step >= warmup * 2)

        # [optional] Evaluate.
        if (step - last_log_step) >= config.Train.DDPG.MinLogStepInterval:
            last_log_step = step
            utils.print.put('[Evaluate] ' + utils.string_utils.dict_to_str(evaluator.summary()))
    
    sim.close()