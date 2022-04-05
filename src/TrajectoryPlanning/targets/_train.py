import config
import random
import time
import utils.print
import utils.string_utils
from copy import copy
from framework.controller.model import Critic as ControllerCritic
from framework.controller.model import Actor as ControllerActor
from framework.ddpg import Agent
from framework.evaluator import Evaluator
from framework.planner.game import Game as PlannerGame
from framework.models.planner import Critic as PlannerCritic
from framework.models.planner import Actor as PlannerActor
from framework.replay_buffer import Transition
from simulator import Game as ControllerGame
from simulator import Simulator

def augment_replay_buffered(replay_buffer: list[Transition]) -> list[Transition]:
    res: list[Transition] = []
    game = ControllerGame()
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

def generate_controller_agent(dim_state, dim_action) -> Agent:
    critic = ControllerCritic(dim_state, dim_action)
    critic_targ = ControllerCritic(dim_state, dim_action)
    actor = ControllerActor(dim_state, dim_action)
    actor_targ = ControllerActor(dim_state, dim_action)
    return Agent(critic, actor, critic_targ, actor_targ, 'controller')

def generate_planner_agent(dim_state, dim_action) -> Agent:
    critic = PlannerCritic(dim_state, dim_action)
    critic_targ = PlannerCritic(dim_state, dim_action)
    actor = PlannerActor(dim_state, dim_action)
    actor_targ = PlannerActor(dim_state, dim_action)
    return Agent(critic, actor, critic_targ, actor_targ, 'planner')

def main():
    sim = Simulator()
    controller_game = ControllerGame()
    planner_game = PlannerGame()
    
    # Reset simulator to get sizes of states and actions.
    state = sim.reset()
    dim_state = state.dim_state()
    dim_controller_action = sim.dim_action()

    # Initialize the agents.
    controller_agent = generate_controller_agent(dim_state, dim_controller_action)
    planner_agent = generate_planner_agent(dim_state, len(state.desired))

    # Load evaluators.
    controller_eval = Evaluator(controller_agent)
    if config.Train.LoadFromPreviousSession: controller_eval.load()

    planner_eval = Evaluator(planner_agent)
    if config.Train.LoadFromPreviousSession: planner_eval.load()

    # Logging.
    last_update_time = time.time()
    last_log_step = 0

    while controller_eval.get_epoch() <= config.Train.DDPG.MaxEpoches:
        epoch_replay_buffer: list[Transition] = []
        done = False

        planner_state = sim.reset()

        controller_game.reset()
        planner_game.reset()

        while not done and controller_eval.get_iteration() <= config.Train.DDPG.MaxIterations:
            step = controller_eval.get_step()

            # Sample an action and perform.
            if step < config.Train.DDPG.Warmup:
                planner_action = planner_agent.sample_random_action()
                controller_action = controller_agent.sample_random_action()
            else:
                if config.Train.DDPG.NoiseEnabled:
                    noise_amount = 1 - step / config.Train.DDPG.Epsilon
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
            planner_rwd, _ = planner_game.update(planner_action, planner_next_state)
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
            epoch_replay_buffer = augment_replay_buffered(epoch_replay_buffer)
            for trans in epoch_replay_buffer:
                controller_agent.replay_buffer.append(trans)

        # Logging.
        planner_eval.epoch(save=step >= config.Train.DDPG.Warmup * 2)
        controller_eval.epoch(save=step >= config.Train.DDPG.Warmup * 2)

        if (step - last_log_step) >= config.Train.DDPG.MinLogStepInterval:
            last_log_step = step
            utils.print.put('[Evaluate] ' + utils.string_utils.dict_to_str(controller_eval.summary()))
    
    sim.close()