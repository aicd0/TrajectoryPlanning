import config
import framework.algorithm
import numpy as np
import utils.print
import utils.string_utils
from envs import Game, Simulator
from framework.configuration import global_configs as configs
from framework.evaluator import Evaluator

def main():
    # Load from configs.
    algorithm = configs.get(config.Training.Agent.Algorithm_)
    model_group = configs.get(config.Model.ModelGroup_)

    # Initialize environment.
    sim = Simulator()
    game = Game()
    state = sim.reset()
    dim_action = sim.dim_action()
    dim_state = state.dim_state()

    # Initialize agent.
    agent = framework.algorithm.create_agent(algorithm, dim_state, dim_action, model_group)

    # Load evaluator.
    evaluator = Evaluator(agent)
    evaluator.load(enable_learning=False)

    # Logging.
    step_rewards = []
    epoch_rewards = []

    for _ in range(config.Testing.MaxEpoches):
        state = sim.reset()
        game.reset()
        sim.plot_reset()
        done = False
        iteration = 0
        epoch_reward = 0

        while not done and iteration < config.Testing.MaxIterations:
            iteration += 1
            action = agent.sample_action(state, deterministic=True)
            state = sim.step(action)
            reward, done = game.update(action, state)
            epoch_reward += reward
            step_rewards.append(reward)
            sim.plot_step()
        
        epoch_rewards.append(epoch_reward)

    sim.close()

    step_rewards = np.array(step_rewards, dtype=config.Common.DataType.Numpy)
    epoch_rewards = np.array(epoch_rewards, dtype=config.Common.DataType.Numpy).reshape(-1, 1)
    
    res = {}
    res['EpRwd'] = np.mean(epoch_rewards)
    res['EpRwdStd'] = np.std(epoch_rewards)
    res['StpRwd'] = np.mean(step_rewards)
    res['StpRwdStd'] = np.std(step_rewards)
    utils.print.put('[Test] ' + utils.string_utils.dict_to_str(res))