import config
import framework.algorithm.her
import numpy as np
import time
import utils.print
import utils.string_utils
from framework.algorithm.ddpg import DDPG as Agent
from framework.configuration import global_configs as configs
from framework.evaluator import Evaluator
from framework.noise.ou import OrnsteinUhlenbeckProcess
from framework.noise.uniform import UniformNoise
from framework.replay_buffer import Transition
from simulator import Game, Simulator

def main():
    # Load from configs.
    max_epoches = configs.get(config.Train.DDPG.FieldMaxEpoches)
    max_iters = configs.get(config.Train.DDPG.FieldMaxIterations)
    noise_enabled = configs.get(config.Train.DDPG.FieldNoiseEnabled)
    warmup = configs.get(config.Train.DDPG.FieldWarmup)
    epsilon = configs.get(config.Train.DDPG.FieldEpsilon)
    her_enabled = configs.get(config.Train.HER.FieldEnabled)
    her_k = configs.get(config.Train.HER.FieldK)

    # Initialize environment.
    sim = Simulator()
    game = Game()
    state = sim.reset()
    dim_action = sim.dim_action()
    dim_state = state.dim_state()

    # Initialize agent.
    agent = Agent(dim_state, dim_action, 'ddpg/l5')

    # Initialize noises.
    random_policy = UniformNoise(dim_action,
        configs.get(config.Train.DDPG.UniformNoise.FieldMin),
        configs.get(config.Train.DDPG.UniformNoise.FieldMax))
    action_noise = OrnsteinUhlenbeckProcess(dim_action,
        theta=configs.get(config.Train.DDPG.OUNoise.FieldTheta),
        mu=configs.get(config.Train.DDPG.OUNoise.FieldMu),
        sigma=configs.get(config.Train.DDPG.OUNoise.FieldSigma))

    # Load evaluator.
    evaluator = Evaluator(agent)
    if config.Train.LoadFromPreviousSession: evaluator.load()

    # Logging.
    last_update_time = time.time()
    last_log_step = 0
    
    while evaluator.get_epoch() <= max_epoches:
        epoch_replay_buffer: list[Transition] = []
        game.reset()
        state = sim.reset()
        done = False

        while not done and evaluator.get_iteration() <= max_iters:
            step = evaluator.get_step()

            # Sample an action and perform.
            if step < warmup:
                action = random_policy.sample()
            else:
                action = agent.sample_action(state)
                
                if noise_enabled:
                    noise_amount = 1 - step / epsilon
                else:
                    noise_amount = 0
                noise_amount = max(noise_amount, 0)
                action += action_noise.sample() * noise_amount
            
            action = np.clip(action, -1, 1)
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
            epoch_replay_buffer = framework.algorithm.her.augment_replay_buffer(epoch_replay_buffer, her_k)
            for trans in epoch_replay_buffer:
                agent.replay_buffer.append(trans)

        evaluator.epoch(allow_save=step >= warmup * 2)

        # [optional] Evaluate.
        if (step - last_log_step) >= config.Train.DDPG.MinLogStepInterval:
            last_log_step = step
            utils.print.put('[Evaluate] ' + utils.string_utils.dict_to_str(evaluator.summary()))
    
    sim.close()