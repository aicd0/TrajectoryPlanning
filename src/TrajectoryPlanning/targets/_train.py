import config
import framework.her
import time
import utils.print
import utils.string_utils
from copy import copy
from framework.configuration import global_configs as configs
from framework.ddpg import Agent
from framework.evaluator import Evaluator
from framework.replay_buffer import Transition
from simulator import Game, Simulator

def main():
    sim = Simulator()
    controller_game = Game()
    planner_game = Game()
    
    # Reset simulator to get sizes of states and actions.
    state = sim.reset()
    dim_state = state.dim_state()
    dim_controller_action = sim.dim_action()

    # Initialize the agents.
    controller_agent = Agent(dim_state, dim_controller_action, 'controller', 'controller')
    planner_agent = Agent(dim_state, len(state.desired), 'planner', 'planner')

    # Load evaluators.
    controller_eval = Evaluator(controller_agent)
    planner_eval = Evaluator(planner_agent)

    if config.Train.LoadFromPreviousSession:
        controller_eval.load()
        planner_eval.load()

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

    while controller_eval.get_epoch() <= max_epoches:
        epoch_replay_buffer: list[Transition] = []
        controller_game.reset()
        planner_game.reset()
        planner_state = sim.reset()
        done = False

        while not done and controller_eval.get_iteration() <= max_iters:
            step = controller_eval.get_step()

            # Sample an action and perform.
            if step < warmup:
                planner_action = planner_agent.sample_random_action()
                controller_action = controller_agent.sample_random_action()
            else:
                if noise_enabled:
                    noise_amount = 1 - step / epsilon
                else:
                    noise_amount = -1
                planner_action = planner_agent.sample_action(planner_state, noise_amount=noise_amount)

                controller_state = copy(planner_state)
                controller_state.desired = planner_action
                controller_action = controller_agent.sample_action(controller_state, noise_amount=noise_amount)

            planner_next_state = sim.step(controller_action)
            controller_next_state = copy(planner_next_state)
            controller_next_state.desired = planner_action

            # Calculate rewards and add to replay buffer.
            planner_rwd, _ = game.update(planner_action, planner_next_state)
            planner_trans = Transition(planner_state, planner_action, planner_rwd, planner_next_state)
            planner_agent.replay_buffer.append(planner_trans)

            controller_rwd, done = controller_game.update(controller_action, controller_next_state)
            controller_trans = Transition(controller_state, controller_action, controller_rwd, controller_next_state)

            epoch_replay_buffer.append(controller_trans)

            # [optional] Optimize & save the agent.
            if step >= config.Train.DDPG.Warmup:
                planner_agent.learn()
                controller_agent.learn()
                
            planner_state = planner_next_state
    
            # Logging.
            planner_eval.step(planner_rwd)
            controller_eval.step(planner_rwd)

            if time.time() - last_update_time > 1:
                utils.print.put('[Train] %s' %
                    utils.string_utils.dict_to_str(controller_eval.summary(shortterm=True)), same_line=True)
                last_update_time = time.time()
        
        # [optional] Perform HER.
        if config.Train.HER.Enabled:
            epoch_replay_buffer = framework.her.augment_replay_buffer(epoch_replay_buffer)
            for trans in epoch_replay_buffer:
                controller_agent.replay_buffer.append(trans)

        # Logging.
        planner_eval.epoch(save=step >= config.Train.DDPG.Warmup * 2)
        controller_eval.epoch(save=step >= config.Train.DDPG.Warmup * 2)

        if (step - last_log_step) >= config.Train.DDPG.MinLogStepInterval:
            last_log_step = step
            utils.print.put('[Evaluate] ' + utils.string_utils.dict_to_str(controller_eval.summary()))
    
    sim.close()